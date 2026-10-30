# -*- coding: utf-8 -*-
"""MuJoCo 连续控制上的 SAC：双 Q 网络、高斯策略与温度参数。
"""
import copy

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace

from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
from rl_server.core.optim import assign_gradients, get_cached_adam, trainable_parameters
from rl_server.core.noisy import NoisyLinear
from rl_server.core.buffers import Experience, ExperienceBuffer

TRAIN_ENVS = {
    'Swimmer': SimpleNamespace(**{
        'ENV_NAME': "Swimmer-v4",
        'OBS_DIM': 8,
        'ACT_DIM': 2,
        'HIDDEN_DIM': 32,
        'USE_NOISE': True
    }),
    'HalfCheetah': SimpleNamespace(**{
        'ENV_NAME': "HalfCheetah-v4",
        'OBS_DIM': 17,
        'ACT_DIM': 6,
        'HIDDEN_DIM': 64,
        'USE_NOISE': True
    }),
    'Ant': SimpleNamespace(**{
        'ENV_NAME': "Ant-v4",
        'OBS_DIM': 27,
        'ACT_DIM': 8,
        'HIDDEN_DIM': 256,
        'USE_NOISE': True
    }),
    'Hopper': SimpleNamespace(**{
        'ENV_NAME': "Hopper-v4",
        'OBS_DIM': 11,
        'ACT_DIM': 3,
        'HIDDEN_DIM': 64,
        'USE_NOISE': True
    }),
    'Pusher': SimpleNamespace(**{
        'ENV_NAME': "Pusher-v5",
        'OBS_DIM': 23,
        'ACT_DIM': 7,
        'HIDDEN_DIM': 128,
        'USE_NOISE': True
    }),
    'Humanoid': SimpleNamespace(**{
        'ENV_NAME': "Humanoid-v4",
        'OBS_DIM': 376,
        'ACT_DIM': 17,
        'HIDDEN_DIM': 512,
        'USE_NOISE': True
    }),
    'Walker2d': SimpleNamespace(**{
        'ENV_NAME': "Walker2d-v4",
        'OBS_DIM': 17,
        'ACT_DIM': 6,
        'HIDDEN_DIM': 64,
        'USE_NOISE': True
    }),
    'Manipulator': SimpleNamespace(**{
        'ENV_NAME': "UR5e-v0",
        'OBS_DIM': 28,
        'ACT_DIM': 6,
        'HIDDEN_DIM': 256,
        'USE_NOISE': True
    }),
    'Reacher3D': SimpleNamespace(**{
        'ENV_NAME': "Reacher-v4",
        'OBS_DIM': 11,
        'ACT_DIM': 2,
        'HIDDEN_DIM': 128,
        'USE_NOISE': True
    }),
    'Quadruped': SimpleNamespace(**{
        'ENV_NAME': "AntMaze-v0",
        'OBS_DIM': 132,
        'ACT_DIM': 12,
        'HIDDEN_DIM': 512,
        'USE_NOISE': True
    }),
    'Bronchoscope': SimpleNamespace(**{
        'ENV_NAME': "Bronchoscope-v0",
        'OBS_DIM': 45,
        'ACT_DIM': 3,
        'HIDDEN_DIM': 256,
        'USE_NOISE': True
    }),
    'HumanoidStandup': SimpleNamespace(**{
        'ENV_NAME': "HumanoidStandup-v4",
        'OBS_DIM': 376,
        'ACT_DIM': 17,
        'HIDDEN_DIM': 512,
        'USE_NOISE': True
    })
}

current_env_name = 'Swimmer'

# Training parameters
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['BATCH_SIZE'] = 256
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
# 熵温度 α（固定值）。原实现完全没有 actor/α 项，策略参数拿不到梯度。
# 初始熵温度（之后由 log_alpha 学习；目标熵取 -act_dim，见 calculator）
TRAIN_CONFIG['ALPHA'] = 0.2
# 目标网络软更新系数
TRAIN_CONFIG['TAU'] = 0.005
# 熵温度通常用比策略更大的学习率单独优化
TRAIN_CONFIG['ALPHA_LEARNING_RATE'] = 3e-4

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32
MODEL_CONFIG['NUM_STEPS'] = 1000
MODEL_CONFIG['OBS_SPACE'] = (8,)
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32)
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_CONFIG['MAX_ACTION'] = 1.0


class MujocoNormalQNet(AlgoBaseNet):
    """策略网络 + 双 Q 网络 + 目标 Q；用于连续动作熵正则化 RL。"""

    def __init__(self):
        super(MujocoNormalQNet, self).__init__()

        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.alpha_noisy_layers = [
                NoisyLinear(hide_dim, hide_dim),
                NoisyLinear(hide_dim, act_dim),
            ]
            self.beta_noisy_layers = [
                NoisyLinear(hide_dim, hide_dim),
                NoisyLinear(hide_dim, act_dim),
            ]

            self.mu = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                self.alpha_noisy_layers[0],
                nn.ReLU(),
                self.alpha_noisy_layers[1]
            )

            self.log_std = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                self.beta_noisy_layers[0],
                nn.ReLU(),
                self.beta_noisy_layers[1],
            )
        else:
            self.mu = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, act_dim))
            )

            self.log_std = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, act_dim)),
            )

        self.q1_value = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, 1)),
        )

        self.q2_value = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, 1))
        )

        # 可学习的熵温度（log 空间保证 α>0）。它和其它参数一样参与梯度聚合，
        # 因此 checkpoint 里会多出 ``log_alpha`` 一个键（旧检查点用 strict=False 兼容加载）。
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(TRAIN_CONFIG['ALPHA']))))

        # 目标 critic 放在网络内部：随 checkpoint 一起持久化、由 update_state 统一做
        # Polyak 更新，避免每个 trainer 进程各持一份而互相漂移。它们 requires_grad=False，
        # 因此不会进入梯度聚合（见 core.optim.trainable_parameters）。
        self.target_q1_value = copy.deepcopy(self.q1_value)
        self.target_q2_value = copy.deepcopy(self.q2_value)
        for param in self.target_q1_value.parameters():
            param.requires_grad_(False)
        for param in self.target_q2_value.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def sync_targets(self):
        """用在线 critic 覆盖目标 critic（旧检查点缺目标网络时调用）。"""
        for target, online in (
            (self.target_q1_value, self.q1_value),
            (self.target_q2_value, self.q2_value),
        ):
            target.load_state_dict(online.state_dict())

    @property
    def alpha(self) -> torch.Tensor:
        """当前熵温度 α = exp(log_alpha)。"""
        return self.log_alpha.exp()

    def get_distributions(self, states):
        mus = self.mu(states)
        log_stds = self.log_std(states).clamp(-20, 2)
        dists = Normal(mus, torch.exp(log_stds))
        return dists

    def forward(self, states):
        mus = self.mu(states)
        return torch.tanh(mus)

    def get_sample_data(self, states):
        dists = self.get_distributions(states)
        actions = dists.sample()
        return torch.tanh(actions)

    def get_check_data(self, states):
        dists = self.get_distributions(states)
        mus = self.mu(states)
        return torch.tanh(mus), dists.entropy()

    def get_calculate_data(self, states, actions):
        values = self.q1_value(torch.cat((states, actions), -1))
        dists = self.get_distributions(states)
        log_probs = dists.log_prob(actions)
        return values, log_probs, dists.entropy()

    def update_state(self, version, grads_buffer):
        params = trainable_parameters(self)
        assign_gradients(params, grads_buffer)

        # 三组参数各用自己的优化器：策略、critic、熵温度（α 用更大学习率）。
        # 复用优化器实例，Adam 的矩估计才能跨版本累积。
        policy_params = list(self.mu.parameters()) + list(self.log_std.parameters())
        critic_params = list(self.q1_value.parameters()) + list(self.q2_value.parameters())
        alpha_params = [self.log_alpha]

        for name, group, lr in (
            ('policy', policy_params, TRAIN_CONFIG['LEARNING_RATE']),
            ('critic', critic_params, TRAIN_CONFIG['LEARNING_RATE']),
            ('alpha', alpha_params, TRAIN_CONFIG['ALPHA_LEARNING_RATE']),
        ):
            optimizer = get_cached_adam(self, name, group, lr)
            optimizer.step()

        for param in params:
            param.grad = None

        # Polyak 软更新目标 critic（不产生梯度）。
        tau = TRAIN_CONFIG['TAU']
        with torch.no_grad():
            for target, online in (
                (self.target_q1_value, self.q1_value),
                (self.target_q2_value, self.q2_value),
            ):
                for target_param, online_param in zip(target.parameters(), online.parameters()):
                    target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            for noise_layer in self.alpha_noisy_layers:
                noise_layer.sample_noise()
            for noise_layer in self.beta_noisy_layers:
                noise_layer.sample_noise()

    def get_q_values(self, states, actions):
        q_input = torch.cat((states, actions), -1)
        q1_values = self.q1_value(q_input)
        q2_values = self.q2_value(q_input)
        return q1_values, q2_values

    def get_mu_q_values(self, states):
        mus = self.mu(states)
        noise = torch.randn_like(mus)
        mu_actions = mus + torch.exp(self.log_std(states)) * noise
        q_input = torch.cat((states, mu_actions), -1)
        mu_q_values = self.q1_value(q_input)
        return mu_q_values


class MujocoNormalQAgent(AlgoBaseAgent):
    """SAC 采样智能体：从策略网络采样动作并写经验。"""

    def __init__(self, sample_net: MujocoNormalQNet, is_checker):
        super(MujocoNormalQAgent, self).__init__()
        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.device = MODEL_CONFIG['DEVICE']
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS']

        env_name = TRAIN_ENVS[current_env_name].ENV_NAME

        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("MujocoNormalQ check mujoco env is", env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]

    def sample_multi_envs(self, model_dict):
        exps = [[] for _ in range(self.num_envs)]

        for _ in range(self.num_steps):
            # 采样中途响应退出请求：否则一次 rollout（MuJoCo 数秒、MicroRTS 更久）
            # 期间 worker 无法退出，只能等满 grace 后被 terminate（env/JVM 清理全丢）。
            if model_dict.get('is_exit'):
                return None
            actions = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])
                if done_n:
                    next_state_n = self.envs[i].reset()[0]

                exps[i].append([self.states[i], actions[i], reward_n, done_n, model_dict['TRAIN_VERSION']])
                self.states[i] = next_state_n

        return exps

    def check_single_env(self):
        step_record_dict = dict()

        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []

        while True:
            # 评估中途让出：否则一次回合要跑满（MuJoCo 数秒、MicroRTS 更久），
            # worker 只能被 SIGTERM 强杀，env/JVM 与 semaphore 都来不及清理。
            if self._stop_requested():
                return None
            mu, entropy = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)

            steps += 1
            if is_done:
                break

        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['sum_entropys'] = np.sum(entropys)
        step_record_dict['average_mus'] = np.mean(mus)

        return step_record_dict

    @torch.no_grad()
    def _get_sample_actions(self, states):
        states_v = torch.Tensor(np.array(states))
        actions = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy()

    @torch.no_grad()
    def _get_single_action(self, state):
        state_v = torch.Tensor(np.array(state))
        mu, entropy = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(), entropy.cpu().numpy()


# Replay buffer config
MAX_BUFFER_SIZE = 100000
REPEAT_TIME = 1


class MujocoNormalQCalculate(AlgoBaseCalculate):
    """SAC 软贝尔曼目标与策略/α 损失，输出聚合梯度。"""

    def __init__(self, SHARE_MODEL: MujocoNormalQNet):
        super(MujocoNormalQCalculate, self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG
        self.share_model = SHARE_MODEL
        self.device = self.model_config['DEVICE']
        self.calculate_net = MujocoNormalQNet()

        self.batch_size = TRAIN_CONFIG['BATCH_SIZE']
        self.exps_buffer = ExperienceBuffer(capacity=MAX_BUFFER_SIZE)
        # 目标 critic 只活在 trainer 进程里（不进 checkpoint、不参与聚合），首次使用时
        # 用在线 critic 初始化，之后按 TAU 软更新。
        self.target_q1_value = None
        self.target_q2_value = None
        self.tau = TRAIN_CONFIG['TAU']

    def generate_grads(self, samples, model_dict):

        gamma = self.train_config['GAMMA']

        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])

        for state, action, reward, done, next_state in zip(s_states[:-1], s_actions[:-1], s_rewards[:-1], s_dones[:-1], s_states[1:]):
            exp = Experience(state, action, reward, done, next_state)
            self.exps_buffer.append(exp)

        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        self.calculate_net.to(self.device)
        train_version = model_dict['TRAIN_VERSION']

        self.calculate_net.zero_grad()

        # actor / α 项都使用"当前温度"的 detached 副本，温度本身单独由 alpha loss 学习。
        alpha = self.calculate_net.alpha.detach()
        # 标准 SAC 的目标熵：-动作维度
        target_entropy = -float(TRAIN_ENVS[current_env_name].ACT_DIM)
        q_loss_value = 0.0
        actor_loss_value = 0.0
        entropy_value = 0.0
        alpha_loss_value = 0.0

        for _ in range(REPEAT_TIME):
            states, actions, rewards, dones, next_states = self.exps_buffer.sample(self.batch_size)

            t_states = torch.Tensor(states).to(self.device)
            t_actions = torch.Tensor(actions).to(self.device)
            t_rewards = torch.Tensor(rewards).to(self.device)
            t_dones = torch.Tensor(dones).to(self.device)
            t_next_states = torch.Tensor(next_states).to(self.device)

            # ---- critic：软贝尔曼误差（终局转移必须把自举项置零）----
            q1_values, q2_values = self.calculate_net.get_q_values(t_states, t_actions)

            with torch.no_grad():
                next_actions = self.calculate_net.get_sample_data(t_next_states)
                # 用目标 critic 估值（原实现用在线 critic 自己给自己当 target）。
                next_q1 = self.calculate_net.target_q1_value(
                    torch.cat((t_next_states, next_actions), -1))
                next_q2 = self.calculate_net.target_q2_value(
                    torch.cat((t_next_states, next_actions), -1))
                next_q = torch.min(next_q1, next_q2)
                target_q = t_rewards.unsqueeze(1) + gamma * (1.0 - t_dones.unsqueeze(1)) * next_q

            q_loss = (F.mse_loss(q1_values, target_q) + F.mse_loss(q2_values, target_q)) / REPEAT_TIME
            q_loss.backward()

            # ---- actor：重参数化策略梯度 ----
            # 先把 critic 的叶子参数冻结：actor loss 只应给策略参数产生梯度，
            # 否则 critic 会被同时推向两个方向（原来这里压根没有 actor 项）。
            critic_params = list(self.calculate_net.q1_value.parameters()) + \
                list(self.calculate_net.q2_value.parameters())
            for param in critic_params:
                param.requires_grad_(False)

            dists = self.calculate_net.get_distributions(t_states)
            raw_actions = dists.rsample()
            pi_actions = torch.tanh(raw_actions)
            # tanh 压缩后的 log-prob 需要雅可比修正，否则策略梯度有偏。
            log_prob = dists.log_prob(raw_actions) - torch.log(1.0 - pi_actions.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            q1_pi, q2_pi = self.calculate_net.get_q_values(t_states, pi_actions)
            actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean() / REPEAT_TIME
            actor_loss.backward()

            for param in critic_params:
                param.requires_grad_(True)

            # ---- 温度：向目标熵 -act_dim 靠近 ----
            alpha_loss = -(self.calculate_net.log_alpha * (log_prob + target_entropy).detach()).mean()
            alpha_loss.backward()
            alpha_loss_value += float(alpha_loss.item())

            q_loss_value += float(q_loss.item())
            actor_loss_value += float(actor_loss.item())
            entropy_value += float(dists.entropy().sum(dim=-1).mean().item())

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in trainable_parameters(self.calculate_net)
        ]

        # 指标交给 trainer 收集（随梯度批次传给主进程 → TensorBoard + 进度行）。
        model_dict['trainer_metrics'] = {
            'q_loss': q_loss_value,
            'actor_loss': actor_loss_value,
            'alpha_loss': alpha_loss_value,
            'entropy': entropy_value,
            'alpha': float(self.calculate_net.alpha.detach()),
        }

        return [grads], train_version

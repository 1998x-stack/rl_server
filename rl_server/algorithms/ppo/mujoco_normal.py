# -*- coding: utf-8 -*-
"""MuJoCo 连续控制上的 PPO：高斯策略、价值头、GAE 与 clip 目标。

``TRAIN_ENVS`` 定义各任务的观测/动作维度。
"""
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

current_env_name = 'Pusher'

# Training parameters
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAE_LAMBDA'] = 0.95
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['CLIP_COEF'] = 0.2
TRAIN_CONFIG['MAX_CLIP_COEF'] = 2
TRAIN_CONFIG['ENT_COEF'] = 0.2
TRAIN_CONFIG['VLAUE_COEF'] = 4
TRAIN_CONFIG['IS_CLIP_VALUE_LOSS'] = False
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
# Ablation: which ratio/pg_loss variant to use for final policy gradient
# 'mixed' (default): ratio3 = (ratio1+ratio2)/2, the original PPO clipped objective
# 'discrete': ratio1 = exp(log_prob_new - log_prob_old) per action dim
# 'prod': ratio2 = exp(sum(log_prob_new) - sum(log_prob_old)) joint product
# 'avg_discrete_prod': pg_loss4 = (pg_loss1 + pg_loss2) / 2, average of discrete and prod ratios
TRAIN_CONFIG['PPO_LOSS_TYPE'] = 'mixed'

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32
MODEL_CONFIG['NUM_STEPS'] = 1000
MODEL_CONFIG['OBS_SPACE'] = (8,)
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32)
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MujocoNormalNet(AlgoBaseNet):
    """高斯策略（μ、σ）与状态价值网络；可选用 ``NoisyLinear`` 增强探索。"""

    def __init__(self):
        super(MujocoNormalNet, self).__init__()

        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.noise_layer_out = NoisyLinear(hide_dim, act_dim)
            self.noise_layer_hide = NoisyLinear(hide_dim, hide_dim)

            self.mu = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                self.noise_layer_hide,
                nn.ReLU(),
                self.noise_layer_out,
                nn.Tanh()
            )
        else:
            self.mu = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, act_dim)),
                nn.Tanh()
            )

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

        self.value = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, 1))
        )

    def get_distributions(self, states):
        mus = self.mu(states)
        dists = Normal(mus, torch.exp(self.log_std))
        return dists

    def forward(self, states):
        mus = self.mu(states)
        return mus

    def get_sample_data(self, states):
        dists = self.get_distributions(states)
        actions = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs

    def get_check_data(self, states):
        dists = self.get_distributions(states)
        mus = self.mu(states)
        return mus, dists.entropy()

    def get_calculate_data(self, states, actions):
        values = self.value(states)
        dists = self.get_distributions(states)
        log_probs = dists.log_prob(actions)
        return values, log_probs, dists.entropy()

    def update_state(self, version, grads_buffer):
        params = trainable_parameters(self)
        # 复用 Adam：每次新建会丢掉一阶/二阶矩。
        train_optim = get_cached_adam(self, 'main', params, TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        assign_gradients(params, grads_buffer)
        train_optim.step()

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()


class MujocoNormalAgent(AlgoBaseAgent):
    """多环境 rollout 采样连续动作；checker 模式下统计单回合回报与熵。"""

    def __init__(self, sample_net: MujocoNormalNet, is_checker):
        super(MujocoNormalAgent, self).__init__()
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
            print("MujocoNormal check mujoco env is", env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]

    def sample_multi_envs(self, model_dict):
        exps = [[] for _ in range(self.num_envs)]

        for _ in range(self.num_steps):
            # 采样中途响应退出请求：否则一次 rollout（MuJoCo 数秒、MicroRTS 更久）
            # 期间 worker 无法退出，只能等满 grace 后被 terminate（env/JVM 清理全丢）。
            if model_dict.get('is_exit'):
                return None
            actions, log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])
                if done_n:
                    next_state_n = self.envs[i].reset()[0]

                exps[i].append([self.states[i], actions[i], reward_n, done_n, log_probs[i], model_dict['TRAIN_VERSION']])
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
        actions, log_probs = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def _get_single_action(self, state):
        state_v = torch.Tensor(np.array(state))
        mu, entropy = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(), entropy.cpu().numpy()


class MujocoNormalCalculate(AlgoBaseCalculate):
    """按 PPO clip 损失从轨迹样本反传，生成供主进程聚合的梯度。"""

    def __init__(self, SHARE_MODEL: MujocoNormalNet):
        super(MujocoNormalCalculate, self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG
        self.share_model = SHARE_MODEL
        self.device = self.model_config['DEVICE']
        self.calculate_net = MujocoNormalNet()

    def generate_grads(self, samples, model_dict):

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        self.calculate_net.to(self.device)
        train_version = model_dict['TRAIN_VERSION']

        gamma = self.train_config['GAMMA']
        gae_lambda = self.train_config['GAE_LAMBDA']
        ent_coef = self.train_config['ENT_COEF']
        vf_coef = self.train_config['VLAUE_COEF']

        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])

        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)

        t_new_values, t_new_log_probs, t_entropys = self.calculate_net.get_calculate_data(t_states, t_actions)

        with torch.no_grad():
            rewards_t = torch.from_numpy(s_rewards[:-1]).float().to(self.device)
            dones_t = torch.from_numpy(s_dones[:-1]).to(self.device).bool()
            values_t = t_new_values[:-1].squeeze(-1)
            next_values_t = t_new_values[1:].squeeze(-1)

            deltas = rewards_t + gamma * next_values_t - values_t
            deltas[dones_t] = rewards_t[dones_t] - values_t[dones_t]

            advantages = torch.zeros(len(deltas) + 1, device=self.device)
            gae = 0.0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + gamma * gae_lambda * (~dones_t[t]).float() * gae
                advantages[t] = gae

            t_advantages = advantages.unsqueeze(-1)
            values_full = torch.cat([values_t, torch.zeros(1, device=self.device)])
            t_returns = (advantages + values_full).unsqueeze(-1)

        t_advantages = t_advantages.reshape(-1, 1)

        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # discrete ratio
        ratio1 = torch.exp(t_new_log_probs - old_log_probs)

        # prod ratio
        ratio2 = torch.exp(t_new_log_probs.sum(1) - old_log_probs.sum(1)).reshape(-1, 1).expand_as(ratio1)

        # mixed ratio
        ratio3 = (ratio1 + ratio2) / 2

        # --- Ablation: pg_loss variants (controlled by TRAIN_CONFIG['PPO_LOSS_TYPE']) ---
        pg_loss1 = self.get_pg_loss(ratio1, t_advantages)  # discrete
        pg_loss2 = self.get_pg_loss(ratio2, t_advantages)  # prod
        pg_loss3 = self.get_pg_loss(ratio3, t_advantages)  # mixed (default)
        pg_loss4 = (pg_loss1 + pg_loss2) / 2               # avg discrete+prod

        loss_type = self.train_config.get('PPO_LOSS_TYPE', 'mixed')
        pg_loss_map = {
            'discrete': pg_loss1,
            'prod': pg_loss2,
            'mixed': pg_loss3,
            'avg_discrete_prod': pg_loss4,
        }
        pg_loss = -pg_loss_map[loss_type].mean()

        v_loss = F.mse_loss(t_returns.reshape(-1, 1), t_new_values) * vf_coef

        e_loss = -torch.mean(t_entropys) * ent_coef

        loss = pg_loss + v_loss + e_loss

        self.calculate_net.zero_grad()

        loss.backward()

        # Compute gradient norm for monitoring
        total_norm = 0.0
        for p in self.calculate_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]

        # Write training metrics for TensorBoard logging
        model_dict['trainer_metrics'] = {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy': e_loss.item(),
            'gradient_norm': total_norm,
            'train_version': train_version,
        }

        return [grads], train_version

    def get_pg_loss(self, ratio, advantage):

        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']

        base_value = ratio * advantage
        clip_value = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage
        min_loss_policy = torch.min(base_value, clip_value)

        max_loss_policy = torch.max(min_loss_policy, max_clip_coef * advantage)

        return torch.where(advantage >= 0, min_loss_policy, max_loss_policy)

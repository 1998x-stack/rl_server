# -*- coding: utf-8 -*-
"""Atari DQN: Nature-style CNN + Dueling architecture, frame-stacked input, Double DQN."""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
from rl_server.core.optim import assign_gradients, get_cached_adam, trainable_parameters
from rl_server.core.buffers import Experience, ExperienceBuffer

TRAIN_ENVS = {
    'Pong': SimpleNamespace(**{
        'ENV_NAME': "ALE/Pong-v5",
        'ACT_DIM': 6,
    }),
    'Breakout': SimpleNamespace(**{
        'ENV_NAME': "ALE/Breakout-v5",
        'ACT_DIM': 4,
    }),
    'SpaceInvaders': SimpleNamespace(**{
        'ENV_NAME': "ALE/SpaceInvaders-v5",
        'ACT_DIM': 6,
    }),
    'Boxing': SimpleNamespace(**{
        'ENV_NAME': "ALE/Boxing-v5",
        'ACT_DIM': 18,
    }),
    'Qbert': SimpleNamespace(**{
        'ENV_NAME': "ALE/Qbert-v5",
        'ACT_DIM': 6,
    }),
    'BeamRider': SimpleNamespace(**{
        'ENV_NAME': "ALE/BeamRider-v5",
        'ACT_DIM': 9,
    }),
    'Enduro': SimpleNamespace(**{
        'ENV_NAME': "ALE/Enduro-v5",
        'ACT_DIM': 9,
    }),
    'Seaquest': SimpleNamespace(**{
        'ENV_NAME': "ALE/Seaquest-v5",
        'ACT_DIM': 18,
    }),
}

current_env_name = 'Pong'

TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['BATCH_SIZE'] = 32
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
TRAIN_CONFIG['epsilon_start'] = 1.0
TRAIN_CONFIG['epsilon_end'] = 0.1
TRAIN_CONFIG['epsilon_decay_steps'] = 100000
# Atari 像素输入的梯度量级很容易到 1e4~1e5（实测），不裁剪会污染梯度聚合。
# 0 表示关闭裁剪。
TRAIN_CONFIG['MAX_GRAD_NORM'] = 10.0

MODEL_CONFIG = dict()
MODEL_CONFIG['ENV_NAME'] = "Atari"
MODEL_CONFIG['NUM_ENVS'] = 4
MODEL_CONFIG['NUM_STEPS'] = 128
MODEL_CONFIG['OBS_SPACE'] = (84, 84, 4)
MODEL_CONFIG['ACTION_SHAPE'] = [TRAIN_ENVS[current_env_name].ACT_DIM]
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_BUFFER_SIZE = 500000


def make_atari_env(env_name):
    """Create an Atari env with standard preprocessing: frame skip, 84x84 grayscale, 4-frame stack."""
    import ale_py.registration
    env = gym.make(env_name, obs_type="rgb", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


class AtariDQNNet(AlgoBaseNet):
    """Nature DQN CNN with Dueling heads for Atari (84x84 grayscale, 4-frame stack)."""

    def __init__(self):
        super(AtariDQNNet, self).__init__()
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        self.value = layer_init(nn.Linear(512, 1))
        self.advantage = layer_init(nn.Linear(512, act_dim))

    def get_q_values(self, states):
        out = self.network(states)
        advantage = self.advantage(out)
        value = self.value(out)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, states):
        q_values = self.get_q_values(states)
        return torch.argmax(q_values, dim=-1)

    def update_state(self, version, grads_buffer):
        params = trainable_parameters(self)
        train_optim = get_cached_adam(self, 'main', params, TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        assign_gradients(params, grads_buffer)
        train_optim.step()


class AtariDQNAgent(AlgoBaseAgent):

    def __init__(self, sample_net: AtariDQNNet, is_checker):
        super(AtariDQNAgent, self).__init__()
        self.sample_net = sample_net
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS'] if not is_checker else 1
        self.act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        env_name = TRAIN_ENVS[current_env_name].ENV_NAME

        self.epsilon = TRAIN_CONFIG['epsilon_start']
        self.epsilon_end = TRAIN_CONFIG['epsilon_end']
        self.epsilon_decay = (self.epsilon - self.epsilon_end) / TRAIN_CONFIG['epsilon_decay_steps']
        self.total_steps = 0

        if not is_checker:
            self.envs = [make_atari_env(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            self.envs = make_atari_env(env_name)
            self.states = self.envs.reset()[0]

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.total_steps += 1

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
                if done_n or truncated:
                    next_state_n = self.envs[i].reset()[0]
                reward_n = np.clip(reward_n, -1.0, 1.0)
                exps[i].append([self.states[i], actions[i], reward_n, done_n or truncated, model_dict['TRAIN_VERSION']])
                self.states[i] = next_state_n
            self._decay_epsilon()
        return exps

    def check_single_env(self):
        rewards = []
        actions = []
        while True:
            # 评估中途让出：否则一次回合要跑满（MuJoCo 数秒、MicroRTS 更久），
            # worker 只能被 SIGTERM 强杀，env/JVM 与 semaphore 都来不及清理。
            if self._stop_requested():
                return None
            action = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(action)
            self.states = next_state_n
            rewards.append(reward_n)
            actions.append(action)
            if is_done or truncated:
                break
        return {'sum_rewards': np.sum(rewards), 'average_mus': np.mean(actions)}

    @torch.no_grad()
    def _get_sample_actions(self, states):
        t_states = torch.FloatTensor(np.array(states))
        if np.random.random() > self.epsilon:
            actions = self.sample_net(t_states).cpu().numpy()
        else:
            actions = np.random.choice(self.act_dim, size=len(states))
        return actions

    @torch.no_grad()
    def _get_single_action(self, state):
        return self.sample_net(torch.FloatTensor(np.array([state]))).cpu().numpy()[0]


class AtariDQNCalculate(AlgoBaseCalculate):

    def __init__(self, SHARE_MODEL: AtariDQNNet):
        super(AtariDQNCalculate, self).__init__()
        self.share_model = SHARE_MODEL
        self.calculate_net = AtariDQNNet()
        self.target_net = AtariDQNNet()
        self.exps_buffer = ExperienceBuffer(capacity=MAX_BUFFER_SIZE)

        self.gamma = TRAIN_CONFIG['GAMMA']
        self.batch_size = TRAIN_CONFIG['BATCH_SIZE']
        self.version_diff = 10
        self.num_repeat = 8
        self.update_version = 0

    def generate_grads(self, samples, model_dict):
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples], dtype=np.bool_)

        for state, action, reward, done, next_state in zip(
            s_states[:-1], s_actions[:-1], s_rewards[:-1], s_dones[:-1], s_states[1:]
        ):
            self.exps_buffer.append(Experience(state, action, reward, done, next_state))

        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        if self.update_version % self.version_diff == 0:
            self.target_net.load_state_dict(self.calculate_net.state_dict())

        self.calculate_net.zero_grad()

        for _ in range(self.num_repeat):
            b_states, b_actions, b_rewards, b_dones, b_next_states = self.exps_buffer.sample(self.batch_size)

            states_v = torch.FloatTensor(b_states)
            actions_v = torch.tensor(b_actions)
            # Atari 奖励未归一化时 Q 目标会爆炸（实测 q_loss≈1e5、裁剪前梯度范数≈2.5e7）。
            # 标准做法：奖励截断到 [-1, 1]。
            rewards_v = torch.FloatTensor(b_rewards).clamp(-1.0, 1.0)
            next_states_v = torch.FloatTensor(b_next_states)
            dones_v = torch.tensor(b_dones)

            q_values = self.calculate_net.get_q_values(states_v)
            q_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_q_values = self.calculate_net.get_q_values(next_states_v)
                expected_actions = torch.max(next_q_values, 1)[1]
                target_next_q_values = self.target_net.get_q_values(next_states_v)
                target_next_q_values = target_next_q_values.gather(1, expected_actions.unsqueeze(1)).squeeze(1)
                target_next_q_values[dones_v] = 0.0
                expected_q_values = rewards_v + self.gamma * target_next_q_values

            # Huber（smooth L1）比 MSE 对离群 TD 误差更稳，也是 Atari DQN 的常规选择。
            loss = F.smooth_l1_loss(q_values, expected_q_values) / self.num_repeat
            loss.backward()

        grad_norm_preclip = self._clip_gradients()
        max_norm = float(TRAIN_CONFIG.get('MAX_GRAD_NORM', 0.0) or 0.0)
        clip_frac = 1.0 if (max_norm > 0 and grad_norm_preclip > max_norm) else 0.0
        grad_scale = (grad_norm_preclip / max_norm) if max_norm > 0 else 1.0

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
        self.update_version += 1
        model_dict['trainer_metrics'] = {
            'q_loss': float(loss.item()),
            'grad_norm_preclip': grad_norm_preclip,
            # 裁剪是否真的在生效、以及原始梯度是裁剪上限的多少倍（尺度失控的直接证据）。
            'clip_frac': clip_frac,
            'grad_scale': grad_scale,
        }
        return [grads], model_dict['TRAIN_VERSION']

    def _clip_gradients(self) -> float:
        """按全局 L2 范数裁剪梯度，返回裁剪前的范数。

        Atari 的像素输入会让梯度范数达到 1e4~1e5（实测），而聚合端会把多个 worker 的
        梯度直接相加，量级失控会直接毁掉训练。裁剪在计算端做，日志里同时保留裁剪前的
        范数，便于判断是否真的触发了裁剪。
        """
        max_norm = float(TRAIN_CONFIG.get('MAX_GRAD_NORM', 0.0) or 0.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.calculate_net.parameters(), max_norm if max_norm > 0 else float('inf')
        )
        return float(grad_norm)

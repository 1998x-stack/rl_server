# -*- coding: utf-8 -*-
"""MuJoCo 连续控制上的 SAC：双 Q 网络、高斯策略与温度参数。
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
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()

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

        for _ in range(REPEAT_TIME):
            states, actions, rewards, dones, next_states = self.exps_buffer.sample(self.batch_size)

            t_states = torch.Tensor(states).to(self.device)
            t_actions = torch.Tensor(actions).to(self.device)
            t_rewards = torch.Tensor(rewards).to(self.device)
            t_next_states = torch.Tensor(next_states).to(self.device)

            # Q loss
            q1_values, q2_values = self.calculate_net.get_q_values(t_states, t_actions)

            with torch.no_grad():
                next_actions = self.calculate_net.get_sample_data(t_next_states)
                next_q1, next_q2 = self.calculate_net.get_q_values(t_next_states, next_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = t_rewards.unsqueeze(1) + gamma * next_q

            q_loss = (F.mse_loss(q1_values, target_q) + F.mse_loss(q2_values, target_q)) / REPEAT_TIME
            q_loss.backward()

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]

        return [grads], train_version

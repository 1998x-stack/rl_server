# -*- coding: utf-8 -*-
"""
PPO with Normal distribution for MuJoCo continuous control environments.
Copied from algo_envs/ppo_mujoco_normal.py with updated imports.
"""
import torch
import torch.nn as nn
import gym
import numpy as np
from gym.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace

from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
from rl_server.core.noisy import NoisyLinear

TRAIN_ENVS = {
    'Swimmer': SimpleNamespace(**{
        'ENV_NAME': "Swimmer-v3",
        'OBS_DIM': 8,
        'ACT_DIM': 2,
        'HIDDEN_DIM': 32,
        'USE_NOISE': True
    }),
    'HalfCheetah': SimpleNamespace(**{
        'ENV_NAME': "HalfCheetah-v3",
        'OBS_DIM': 17,
        'ACT_DIM': 6,
        'HIDDEN_DIM': 64,
        'USE_NOISE': True
    }),
    'Ant': SimpleNamespace(**{
        'ENV_NAME': "Ant-v3",
        'OBS_DIM': 111,
        'ACT_DIM': 8,
        'HIDDEN_DIM': 256,
        'USE_NOISE': True
    }),
    'Hopper': SimpleNamespace(**{
        'ENV_NAME': "Hopper-v3",
        'OBS_DIM': 11,
        'ACT_DIM': 3,
        'HIDDEN_DIM': 64,
        'USE_NOISE': True
    }),
    'Pusher': SimpleNamespace(**{
        'ENV_NAME': "Pusher-v2",
        'OBS_DIM': 23,
        'ACT_DIM': 7,
        'HIDDEN_DIM': 128,
        'USE_NOISE': True
    }),
    'Humanoid': SimpleNamespace(**{
        'ENV_NAME': "Humanoid-v3",
        'OBS_DIM': 376,
        'ACT_DIM': 17,
        'HIDDEN_DIM': 512,
        'USE_NOISE': True
    }),
    'Walker2d': SimpleNamespace(**{
        'ENV_NAME': "Walker2d-v3",
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
        'ENV_NAME': "Reacher-v3",
        'OBS_DIM': 16,
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
        'ENV_NAME': "HumanoidStandup-v2",
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

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32
MODEL_CONFIG['NUM_STEPS'] = 1000
MODEL_CONFIG['OBS_SPACE'] = (8,)
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32)
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')


class MujocoNormalNet(AlgoBaseNet):

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
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()


class MujocoNormalAgent(AlgoBaseAgent):

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

        s_advantages = [0]
        s_returns = [0]

        # Calculate advantages
        last_gae = 0.0
        with torch.no_grad():
            for value, next_value, reward, done in zip(reversed(t_new_values[:-1]), reversed(t_new_values[1:]),
                                                       reversed(s_rewards[:-1]), reversed(s_dones[:-1])):
                if done:
                    delta = reward - value
                    last_gae = delta
                else:
                    delta = reward + gamma * next_value - value
                    last_gae = delta + gamma * gae_lambda * last_gae

                s_advantages.append(last_gae)
                s_returns.append(last_gae + value)

        t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)
        t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)

        t_advantages = t_advantages.reshape(-1, 1)

        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # discrete ratio
        ratio1 = torch.exp(t_new_log_probs - old_log_probs)

        # prod ratio
        ratio2 = torch.exp(t_new_log_probs.sum(1) - old_log_probs.sum(1)).reshape(-1, 1).expand_as(ratio1)

        # mixed ratio
        ratio3 = (ratio1 + ratio2) / 2

        # Policy loss
        pg_loss1 = self.get_pg_loss(ratio1, t_advantages)
        pg_loss2 = self.get_pg_loss(ratio2, t_advantages)
        pg_loss3 = self.get_pg_loss(ratio3, t_advantages)
        pg_loss4 = (pg_loss1 + pg_loss2) / 2

        # Policy loss
        pg_loss = -pg_loss3.mean()

        v_loss = F.mse_loss(t_returns.reshape(-1, 1), t_new_values) * vf_coef

        e_loss = -torch.mean(t_entropys) * ent_coef

        loss = pg_loss + v_loss + e_loss

        self.calculate_net.zero_grad()

        loss.backward()
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]

        return [grads], train_version

    def get_pg_loss(self, ratio, advantage):

        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']

        base_value = ratio * advantage
        clip_value = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage
        min_loss_policy = torch.min(base_value, clip_value)

        max_loss_policy = torch.max(min_loss_policy, max_clip_coef * advantage)

        return torch.where(advantage >= 0, min_loss_policy, max_loss_policy)

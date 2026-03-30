# -*- coding: utf-8 -*-
"""
DQN network for Gym classic control environments.
Copied from algo_envs/dqn_gym_classic.py with updated imports.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from rl_server.core.base import AlgoBaseNet, layer_init

TRAIN_ENVS = {
    'CartPole': SimpleNamespace(**{
        'ENV_NAME': "CartPole-v1",
        'OBS_DIM': 4,
        'ACT_DIM': 2,
        'HIDDEN_DIM': 16
    }),
    'MountainCar': SimpleNamespace(**{
        'ENV_NAME': "MountainCar-v1",
        'OBS_DIM': 2,
        'ACT_DIM': 3,
        'HIDDEN_DIM': 32
    }),
    'Acrobot': SimpleNamespace(**{
        'ENV_NAME': "Acrobot-v1",
        'OBS_DIM': 6,
        'ACT_DIM': 3,
        'HIDDEN_DIM': 64
    }),
    'Pendulum': SimpleNamespace(**{
        'ENV_NAME': "Pendulum-v1",
        'OBS_DIM': 3,
        'ACT_DIM': 1,
        'HIDDEN_DIM': 64
    }),
    'LunarLander': SimpleNamespace(**{
        'ENV_NAME': "LunarLander-v2",
        'OBS_DIM': 8,
        'ACT_DIM': 4,
        'HIDDEN_DIM': 128
    })
}

current_env_name = 'CartPole'

# Training parameters
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['BATCH_SIZE'] = 256
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
TRAIN_CONFIG['epsilon'] = 0.01

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32
MODEL_CONFIG['NUM_STEPS'] = 512
MODEL_CONFIG['OBS_SPACE'] = (4,)
MODEL_CONFIG['ACTION_SHAPE'] = [2]
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')

MAX_BUFFER_SIZE = 100000


class DQNGymClassicNet(AlgoBaseNet):
    """Dueling DQN network for Gym classic control environments."""

    def __init__(self):
        super(DQNGymClassicNet, self).__init__()

        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM

        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
        )

        # Dueling DQN
        self.value = layer_init(nn.Linear(hide_dim, 1))
        self.advantage = layer_init(nn.Linear(hide_dim, act_dim))

    def get_q_values(self, states):
        out = self.network(states)
        advantage = self.advantage(out)
        value = self.value(out)
        return value + advantage - advantage.mean()

    def forward(self, states):
        q_values = self.get_q_values(states)
        action = torch.argmax(q_values, dim=-1)
        return action

    def update_state(self, version, grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()

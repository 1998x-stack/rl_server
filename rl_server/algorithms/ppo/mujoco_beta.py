# -*- coding: utf-8 -*-
"""MuJoCo 连续控制上的 PPO：Beta 分布策略（有界动作）与价值网络。

实现源自 ``algo_envs/ppo_mujoco_beta.py``。
"""
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from torch.nn import functional as F
from torch.distributions.beta import Beta
from types import SimpleNamespace

from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
from rl_server.core.noisy import NoisyLinear, GradCoef

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
TRAIN_CONFIG['GAE_LAMBDA'] = 0.95
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['CLIP_COEF'] = 0.2
TRAIN_CONFIG['MAX_CLIP_COEF'] = 100
TRAIN_CONFIG['ENT_COEF'] = 0.2
TRAIN_CONFIG['VLAUE_COEF'] = 2
TRAIN_CONFIG['IS_CLIP_VALUE_LOSS'] = False
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
TRAIN_CONFIG['RATIO_COEF'] = 1
# Ablation: which ratio/pg_loss variant to use for final policy gradient
# 'mixed' (default), 'discrete', 'prod', 'avg_discrete_prod'
TRAIN_CONFIG['PPO_LOSS_TYPE'] = 'mixed'

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32
MODEL_CONFIG['NUM_STEPS'] = 1000
MODEL_CONFIG['OBS_SPACE'] = (8,)
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32)
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_CONFIG['MAX_ACTION'] = 1.0


class MujocoBetaNet(AlgoBaseNet):
    """Beta 分布参数化策略与价值头；动作为 [0,1] 再线性映射到环境区间。"""

    def __init__(self):
        super(MujocoBetaNet, self).__init__()

        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM

        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.alpha_noisy_layers = [NoisyLinear(hide_dim, act_dim), NoisyLinear(hide_dim, hide_dim)]
            self.beta_noisy_layers = [NoisyLinear(hide_dim, act_dim), NoisyLinear(hide_dim, hide_dim)]

            self.alpha = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                self.alpha_noisy_layers[1],
                nn.ReLU(),
                self.alpha_noisy_layers[0],
                nn.Softplus()
            )

            self.beta = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                self.beta_noisy_layers[1],
                nn.ReLU(),
                self.beta_noisy_layers[0],
                nn.Softplus()
            )
        else:
            self.alpha = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, act_dim)),
                nn.Softplus()
            )

            self.beta = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hide_dim, act_dim)),
                nn.Softplus()
            )

        self.value = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, 1))
        )

    def get_distributions(self, states):
        alphas = self.alpha(states) + 1.0
        betas = self.beta(states) + 1.0
        dists = Beta(alphas, betas)
        return dists

    def forward(self, states):
        dists = self.get_distributions(states)
        mus = (dists.mean - 0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION']
        return mus

    def get_sample_data(self, states):
        dists = self.get_distributions(states)
        sample_actions = dists.sample()
        log_probs = dists.log_prob(sample_actions)
        actions = (sample_actions - 0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION']
        return sample_actions, actions, log_probs

    def get_check_data(self, states):
        dists = self.get_distributions(states)
        mus = (dists.mean - 0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION']
        log_probs = dists.log_prob(dists.mean)
        return mus, dists.entropy(), log_probs

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
            for noise_layer in self.alpha_noisy_layers:
                noise_layer.sample_noise()
            for noise_layer in self.beta_noisy_layers:
                noise_layer.sample_noise()


class MujocoBetaAgent(AlgoBaseAgent):
    """并行环境 rollout 与单环境评估，与 ``MujocoBetaNet`` 配套。"""

    def __init__(self, sample_net: MujocoBetaNet, is_checker):
        super(MujocoBetaAgent, self).__init__()
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
            print("MujocoBeta check mujoco env is", env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]

    def sample_multi_envs(self, model_dict):
        exps = [[] for _ in range(self.num_envs)]
        for _ in range(self.num_steps):
            sample_acitons, actions, log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])
                if done_n:
                    next_state_n = self.envs[i].reset()[0]

                exps[i].append([self.states[i], sample_acitons[i], reward_n, done_n, log_probs[i], model_dict['TRAIN_VERSION']])
                self.states[i] = next_state_n

        return exps

    def check_single_env(self):
        step_record_dict = dict()

        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []
        log_probs = []

        while True:
            mu, entropy, log_prob = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            log_probs.append(log_prob)

            steps += 1
            if is_done:
                break

        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_mus'] = np.mean(mus)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)

        return step_record_dict

    @torch.no_grad()
    def _get_sample_actions(self, states):
        states_v = torch.Tensor(np.array(states))
        sample_acitons, actions, log_probs = self.sample_net.get_sample_data(states_v)
        return sample_acitons.cpu().numpy(), actions.cpu().numpy(), log_probs.cpu().numpy()

    @torch.no_grad()
    def _get_single_action(self, state):
        state_v = torch.Tensor(np.array(state))
        mu, entropy, log_prob = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(), entropy.cpu().numpy(), log_prob.cpu().numpy()


class MujocoBetaCalculate(AlgoBaseCalculate):
    """PPO 目标下从样本轨迹计算策略/价值梯度。"""

    def __init__(self, SHARE_MODEL: MujocoBetaNet):
        super(MujocoBetaCalculate, self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG
        self.share_model = SHARE_MODEL
        self.device = self.model_config['DEVICE']
        self.calculate_net = MujocoBetaNet()
        self.calculate_net.to(self.device)

    def generate_grads(self, samples, model_dict):

        train_version = model_dict['TRAIN_VERSION']
        gamma = self.train_config['GAMMA']
        gae_lambda = self.train_config['GAE_LAMBDA']
        vf_coef = self.train_config['VLAUE_COEF']
        ent_coef = self.train_config['ENT_COEF']
        ratio_coef = TRAIN_CONFIG['RATIO_COEF']

        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])

        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        t_new_values, t_new_log_probs, t_entropys = self.calculate_net.get_calculate_data(t_states, t_actions)

        s_advantages = [0]
        s_returns = [0]

        # Calculate advantages
        last_gae = 0.0
        with torch.no_grad():
            for value, next_value, reward, done in zip(
                reversed(t_new_values[:-1]),
                reversed(t_new_values[1:]),
                reversed(s_rewards[:-1]),
                reversed(s_dones[:-1])
            ):
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

        t_advantages = t_advantages.reshape(-1, 1).expand_as(t_new_log_probs)
        t_returns = t_returns.reshape(-1, 1)

        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # discrete ratio
        ratio1 = torch.exp(t_new_log_probs - old_log_probs)

        # prod ratio
        ratio2 = ratio1.prod(1, keepdim=True).expand_as(ratio1)
        ratio2 = GradCoef.apply(ratio2, 1.0 / ratio2.shape[1])

        # mixed ratio
        ratio3 = (GradCoef.apply(ratio1, ratio_coef) + GradCoef.apply(ratio2, 2.0 - ratio_coef)) / 2

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
        pg_loss = -torch.mean(pg_loss_map[loss_type])

        v_loss = F.mse_loss(t_returns, t_new_values) * vf_coef

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

    def get_pg_loss(self, ratio, advantage, clip_max=False):

        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']

        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage, advantage)
        if clip_max:
            negtive = torch.where(ratio <= 1.0 - clip_coef, 0 * advantage, torch.where(ratio >= max_clip_coef, 0 * advantage, advantage))
        else:
            negtive = torch.where(ratio <= 1.0 - clip_coef, 0 * advantage, advantage)

        return torch.where(advantage >= 0, positive, negtive) * ratio

    def get_r_coef(self, ratio, advantage, clip_max=False):
        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']

        positive = torch.where(ratio >= 1.0 + clip_coef, 0, 1)
        if clip_max:
            negtive = torch.where(ratio <= 1.0 - clip_coef, 0, torch.where(ratio >= max_clip_coef, 0, 1))
        else:
            negtive = torch.where(ratio <= 1.0 - clip_coef, 0, 1)

        return torch.where(advantage >= 0, positive, negtive)

    def get_prod_ratio(self, ratio):
        act_dim = ratio.shape[1]
        select_ratio = [ratio.select(1, i) for i in range(act_dim)]

        prod_ratio = []

        for i in range(act_dim):
            prod_value = 1
            for j in range(act_dim):
                if i != j:
                    prod_value *= select_ratio[j].detach()
            prod_ratio.append(select_ratio[i] * prod_value)

        return torch.stack(prod_ratio, dim=1)

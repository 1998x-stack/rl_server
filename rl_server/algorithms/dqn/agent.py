# -*- coding: utf-8 -*-
"""经典 Gym 环境的 DQN 智能体：向量环境并行采样与单环境评估。

实现源自 ``algo_envs/dqn_gym_classic.py``。
"""
import gymnasium as gym
import numpy as np
import torch

from rl_server.core.base import AlgoBaseAgent
from rl_server.algorithms.dqn.network import (
    DQNGymClassicNet, TRAIN_ENVS, current_env_name, TRAIN_CONFIG, MODEL_CONFIG
)


class DQNGymClassicAgent(AlgoBaseAgent):
    """ε-贪心探索；支持多环境 rollout 与 checker 单环境评估。"""

    def __init__(self, sample_net: DQNGymClassicNet, is_checker):
        super(DQNGymClassicAgent, self).__init__()

        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS']
        self.epsilon = TRAIN_CONFIG['epsilon']

        self.ACT_DIM = TRAIN_ENVS[current_env_name].ACT_DIM
        env_name = TRAIN_ENVS[current_env_name].ENV_NAME

        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("DQNGymClassic check env is", env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]

    def sample_multi_envs(self, model_dict):
        """并行运行 ``NUM_STEPS`` 步，返回每个子环境一条转移序列（含 ``TRAIN_VERSION``）。"""
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
        """单回合直到 ``done``，返回奖励和与动作相关的统计字典。"""
        actions = []
        rewards = []
        is_done = False
        step_record_dict = dict()

        while True:
            action = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(action)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            rewards.append(reward_n)
            actions.append(action)

            if is_done:
                break

        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['average_mus'] = np.mean(actions)

        return step_record_dict

    @torch.no_grad()
    def _get_sample_actions(self, states):
        """批量状态 -> 贪心或随机动作（训练探索）。"""
        t_states = torch.Tensor(states)
        if np.random.random() > self.epsilon:
            actions = self.sample_net(t_states)
            actions = actions.cpu().numpy()
        else:
            actions = np.random.choice(self.ACT_DIM, size=t_states.shape[0])
        return actions

    @torch.no_grad()
    def _get_single_action(self, state):
        """单状态贪心动作（标量或向量）。"""
        action = self.sample_net(torch.Tensor(state))
        action = action.cpu().numpy()
        return action

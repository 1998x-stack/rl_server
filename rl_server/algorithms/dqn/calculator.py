# -*- coding: utf-8 -*-
"""DQN 梯度计算器：经验回放缓冲、目标网络与 MSE TD 损失。
"""
import numpy as np
import torch
import torch.nn.functional as F

from rl_server.core.base import AlgoBaseCalculate
from rl_server.core.buffers import Experience, ExperienceBuffer
from rl_server.algorithms.dqn.network import (
    DQNGymClassicNet, TRAIN_CONFIG, MODEL_CONFIG, MAX_BUFFER_SIZE
)


class DQNGymClassicCalculate(AlgoBaseCalculate):
    """维护在线网、目标网与经验池，从样本生成一步聚合梯度。"""

    def __init__(self, SHARE_MODEL: DQNGymClassicNet):
        super(DQNGymClassicCalculate, self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG
        self.share_model = SHARE_MODEL
        self.calculate_net = DQNGymClassicNet()
        self.target_net = DQNGymClassicNet()
        self.exps_buffer = ExperienceBuffer(capacity=MAX_BUFFER_SIZE)

        self.gamma = self.train_config['GAMMA']
        self.batch_size = TRAIN_CONFIG['BATCH_SIZE']
        self.version_diff = 10
        self.num_repeat = 64
        self.update_version = 0

    def generate_grads(self, samples, model_dict):
        """将样本写入经验池，采样若干 minibatch 重复反传，返回 ``[grads]`` 与当前训练版本。"""
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[4] for s in samples])

        for state, action, reward, done, next_state in zip(s_states[:-1], s_actions[:-1], s_rewards[:-1], s_dones[:-1], s_states[1:]):
            exp = Experience(state, action, reward, done, next_state)
            self.exps_buffer.append(exp)

        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        if self.update_version % self.version_diff == 0:
            self.target_net.load_state_dict(self.calculate_net.state_dict())

        self.calculate_net.zero_grad()

        for _ in range(self.num_repeat):
            s_states, s_actions, s_rewards, s_dones, s_next_states = self.exps_buffer.sample(self.batch_size)

            states_v = torch.Tensor(s_states)
            actions_v = torch.tensor(s_actions)
            rewards_v = torch.Tensor(s_rewards)
            next_states_v = torch.Tensor(s_next_states)

            q_values: torch.Tensor = self.calculate_net.get_q_values(states_v)
            q_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_q_values: torch.Tensor = self.calculate_net.get_q_values(next_states_v)
                expected_actions = torch.max(next_q_values, 1)[1]
                target_next_q_values: torch.Tensor = self.target_net.get_q_values(next_states_v)
                target_next_q_values = target_next_q_values.gather(1, expected_actions.unsqueeze(1)).squeeze(1)
                target_next_q_values[s_dones] = 0.0
                expected_q_values = rewards_v + self.gamma * target_next_q_values

            loss = F.mse_loss(q_values, expected_q_values) / self.num_repeat
            loss.backward()

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
        self.update_version = self.update_version + 1

        return [grads], model_dict['TRAIN_VERSION']

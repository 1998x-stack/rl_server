# -*- coding: utf-8 -*-
"""经验回放缓冲区：单步转移与整条轨迹两种存储方式。

从 ``libs/exps.py`` 迁移，已移除对 ``sys.path`` 的修改。
"""
import collections
import numpy as np
from typing import List

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'next_state'],
)


class ExperienceBuffer:
    """以「单条经验列表」为元素的 FIFO 缓冲区，支持均匀随机批次采样。"""

    def __init__(self, capacity):
        """初始化缓冲区。

        Args:
            capacity: 最大条目数；超出后丢弃最旧元素。
        """
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        """返回当前存储的条目数。"""
        return len(self.buffer)

    def append(self, exps: List[Experience]):
        """追加一条经验列表（一次 append 作为一个采样单元）。

        Args:
            exps: ``Experience`` 命名元组组成的列表。
        """
        self.buffer.append(exps)

    def sample(self, batch_size: int):
        """无放回随机采样若干条存储单元，并拆成五元组数组。

        Args:
            batch_size: 采样条数，需不大于当前长度。

        Returns:
            五元组 ``(states, actions, rewards, dones, next_states)``，均为 ``numpy`` 数组。
        """
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


class TrajectoryBuffer:
    """以「整条轨迹」为元素的缓冲区，采样返回轨迹列表。"""

    def __init__(self, capacity: int = 1000000):
        """初始化轨迹缓冲区。

        Args:
            capacity: 最大轨迹条数。
        """
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        """返回当前轨迹条数。"""
        return len(self.buffer)

    def append(self, trajectory: List):
        """追加一条轨迹。

        Args:
            trajectory: 算法约定的轨迹结构（通常为时间步列表）。
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int):
        """无放回随机采样若干条完整轨迹。

        Args:
            batch_size: 采样轨迹条数。

        Returns:
            长度为 ``batch_size`` 的轨迹列表。
        """
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        return [self.buffer[idx] for idx in indices]

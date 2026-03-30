# -*- coding: utf-8 -*-
"""
Experience buffers for RL training.
Copied from libs/exps.py with sys.path hack removed.
"""
import collections
import numpy as np
from typing import List

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exps: List[Experience]):
        self.buffer.append(exps)

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


class TrajectoryBuffer:
    def __init__(self, capacity: int = 1000000):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, trajectory: List):
        self.buffer.append(trajectory)

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        return [self.buffer[idx] for idx in indices]

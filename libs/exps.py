# -*- coding: utf-8 -*-
"""经验缓冲：单步 ``Experience`` 与轨迹缓冲（与 ``rl_server.core.buffers`` 同源）。"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import collections
import numpy as np
from typing import List

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'next_state'])
        
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
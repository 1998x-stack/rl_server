import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'next_state'])
        
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exps):
        self.buffer.append(exps)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)

class TrajectoryBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,replace=False)
        return [self.buffer[idx] for idx in indices]
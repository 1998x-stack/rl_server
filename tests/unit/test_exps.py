"""经验缓冲区单元测试。"""
import numpy as np
import pytest

from rl_server.core.buffers import Experience, ExperienceBuffer, TrajectoryBuffer


class TestExperienceBuffer:
    def test_append_and_len(self):
        buf = ExperienceBuffer(capacity=100)
        assert len(buf) == 0
        exp = Experience(state=np.zeros(4), action=1, reward=1.0, done=False, next_state=np.zeros(4))
        buf.append(exp)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ExperienceBuffer(capacity=3)
        for i in range(5):
            exp = Experience(state=np.array([i]), action=i, reward=float(i), done=False, next_state=np.array([i + 1]))
            buf.append(exp)
        assert len(buf) == 3

    def test_sample_returns_correct_shapes(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(20):
            exp = Experience(
                state=np.array([i, i + 1]),
                action=i % 3,
                reward=float(i),
                done=False,
                next_state=np.array([i + 1, i + 2])
            )
            buf.append(exp)
        states, actions, rewards, dones, next_states = buf.sample(5)
        assert states.shape == (5, 2)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert dones.shape == (5,)
        assert next_states.shape == (5, 2)

    def test_sample_raises_on_insufficient_data(self):
        buf = ExperienceBuffer(capacity=100)
        exp = Experience(state=np.zeros(2), action=0, reward=0.0, done=False, next_state=np.zeros(2))
        buf.append(exp)
        with pytest.raises(ValueError):
            buf.sample(5)


class TestTrajectoryBuffer:
    def test_append_and_len(self):
        buf = TrajectoryBuffer(capacity=10)
        buf.append([1, 2, 3])
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = TrajectoryBuffer(capacity=3)
        for i in range(5):
            buf.append([i])
        assert len(buf) == 3

    def test_sample(self):
        buf = TrajectoryBuffer(capacity=100)
        for i in range(10):
            buf.append([i, i * 2])
        samples = buf.sample(3)
        assert len(samples) == 3

    def test_sample_raises_on_insufficient_data(self):
        buf = TrajectoryBuffer(capacity=100)
        buf.append([1])
        with pytest.raises(ValueError):
            buf.sample(5)

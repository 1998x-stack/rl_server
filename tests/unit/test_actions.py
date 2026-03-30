"""动作选择器单元测试。"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from libs.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector, ProbabilityActionSelector, EpsilonTracker


class TestArgmaxActionSelector:
    def test_selects_max_index(self):
        selector = ArgmaxActionSelector()
        scores = np.array([[0.1, 0.9, 0.3], [0.7, 0.2, 0.1]])
        actions = selector(scores)
        assert actions[0] == 1
        assert actions[1] == 0

    def test_batch_output_shape(self):
        selector = ArgmaxActionSelector()
        scores = np.random.randn(5, 4)
        actions = selector(scores)
        assert actions.shape == (5,)


class TestEpsilonGreedyActionSelector:
    def test_zero_epsilon_is_greedy(self):
        selector = EpsilonGreedyActionSelector(epsilon=0.0)
        scores = np.array([[0.1, 0.9], [0.8, 0.2]])
        actions = selector(scores)
        assert actions[0] == 1
        assert actions[1] == 0

    def test_full_epsilon_is_random(self):
        np.random.seed(42)
        selector = EpsilonGreedyActionSelector(epsilon=1.0)
        scores = np.zeros((100, 4))
        actions = selector(scores)
        assert len(set(actions)) > 1


class TestProbabilityActionSelector:
    def test_samples_from_distribution(self):
        np.random.seed(42)
        selector = ProbabilityActionSelector()
        probs = np.array([[0.01, 0.01, 0.98]])
        actions = [selector(probs)[0] for _ in range(100)]
        assert actions.count(2) > 80

    def test_output_shape(self):
        selector = ProbabilityActionSelector()
        probs = np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]])
        actions = selector(probs)
        assert actions.shape == (3,)


class TestEpsilonTracker:
    def test_epsilon_decays(self):
        selector = EpsilonGreedyActionSelector(epsilon=1.0)
        # EpsilonTracker.__init__ calls frame(0), which sets epsilon = eps_start - 0/eps_frames = 1.0
        tracker = EpsilonTracker(selector, eps_start=1.0, eps_final=0.01, eps_frames=100)
        assert selector.epsilon == 1.0

        # formula: eps = eps_start - frame / eps_frames
        # frame(50): eps = 1.0 - 50/100 = 0.5
        tracker.frame(50)
        assert selector.epsilon == pytest.approx(0.5, abs=0.01)

        # frame(200): eps = 1.0 - 200/100 = -1.0, clamped to eps_final=0.01
        tracker.frame(200)
        assert selector.epsilon == 0.01

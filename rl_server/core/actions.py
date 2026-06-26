# -*- coding: utf-8 -*-
"""动作选择器：将 Q 值、 logits 或概率向量映射为离散动作。
"""
import numpy as np
from typing import Union


class ActionSelector:
    """动作选择器抽象基类：将「分数」转为环境动作索引。"""

    def __call__(self, scores):
        """根据分数张量选择动作（由子类实现）。

        Args:
            scores: 通常为形状 ``(batch, n_actions)`` 的 numpy 数组。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """贪心策略：对每个样本在动作维上取 argmax。"""

    def __call__(self, scores: np.ndarray):
        """返回每行最大 Q 值对应的动作索引。

        Args:
            scores: 形状 ``(batch_size, n_actions)`` 的数组。

        Returns:
            形状 ``(batch_size,)`` 的整型动作索引数组。
        """
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    """ε-贪心：以概率 ε 随机探索，否则使用内嵌选择器（默认 argmax）。"""

    def __init__(self, epsilon=0.05, selector=None):
        """初始化 ε-贪心选择器。

        Args:
            epsilon: 随机探索概率，范围 ``[0, 1]``。
            selector: 非随机时使用的选择器；若为 ``None`` 则使用 ``ArgmaxActionSelector``。
        """
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores: np.ndarray):
        """对批次应用 ε-贪心。

        Args:
            scores: 形状 ``(batch_size, n_actions)`` 的数组。

        Returns:
            形状 ``(batch_size,)`` 的动作索引数组。
        """
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """按每行概率分布独立采样动作（用于策略输出为概率时）。"""

    def __call__(self, probs: np.ndarray):
        """对每一行概率向量采样一个动作。

        Args:
            probs: 形状 ``(batch_size, n_actions)``，每行和为 1 的概率分布。

        Returns:
            形状 ``(batch_size,)`` 的整型动作数组。
        """
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """随帧数线性衰减 ε，用于训练过程中逐渐减小探索。"""

    def __init__(
        self,
        selector: EpsilonGreedyActionSelector,
        eps_start: Union[int, float],
        eps_final: Union[int, float],
        eps_frames: int,
    ):
        """初始化 ε 调度器。

        Args:
            selector: 被修改 ``epsilon`` 属性的 ``EpsilonGreedyActionSelector``。
            eps_start: 初始 ε 值。
            eps_final: ε 下限，不会低于该值。
            eps_frames: 从 ``eps_start`` 线性衰减到接近 ``eps_final`` 所用的帧数尺度。
        """
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        """根据当前帧号更新 ``selector.epsilon``。

        Args:
            frame: 当前全局帧索引；ε 按 ``eps_start - frame / eps_frames`` 线性下降并截断。
        """
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)

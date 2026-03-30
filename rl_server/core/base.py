# -*- coding: utf-8 -*-
"""强化学习算法抽象基类：网络、智能体与梯度计算器接口。

从 ``algo_envs/algo_base.py`` 抽取，供各具体环境与算法实现继承。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

from rl_server.core.noisy import NoisyLinear


def layer_init(
    linear_layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
    method: str = 'orthogonal',
) -> nn.Linear:
    """对线性层权重与偏置进行初始化。

    Args:
        linear_layer: 待初始化的 ``nn.Linear`` 模块。
        std: 正交初始化时的缩放系数，默认 ``sqrt(2)``。
        bias_const: 偏置常数初值。
        method: ``'orthogonal'`` 使用正交初始化；``'kaiming'`` 使用 Kaiming 正态初始化（ReLU）。

    Returns:
        与原模块同一对象，便于链式调用。
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(linear_layer.weight, std)
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, bias_const)
    else:
        nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)

    return linear_layer


class AlgoBaseNet(nn.Module):
    """策略/价值网络的算法无关基类。

    子类需实现前向传播与根据梯度缓冲更新参数的逻辑。
    """

    def __init__(self):
        super(AlgoBaseNet, self).__init__()

    def forward(self, states):
        """前向计算（由子类实现）。

        Args:
            states: 输入状态张量或批次。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError

    def update_state(self, version: int, grads_buffer: List):
        """使用聚合后的梯度缓冲更新网络参数（由子类实现）。

        Args:
            version: 当前全局训练版本号。
            grads_buffer: 梯度列表或与算法约定一致的结构。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError


class AlgoBaseAgent:
    """智能体基类：负责多环境采样与单环境评估（通常不占用 GPU）。

    采样逻辑与具体环境、算法绑定，由子类实现。
    """

    def __init__(self):
        pass

    def sample_multi_envs(self, model_dict: Dict):
        """在多个并行环境中采集经验（由子类实现）。

        Args:
            model_dict: 含 ``TRAIN_VERSION`` 等共享状态的字典。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError

    def check_single_env(self):
        """在单个环境中做评估/检查（由子类实现）。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError


class AlgoBaseCalculate:
    """梯度计算器基类：根据样本批次生成梯度（供 Trainer 使用）。"""

    def __init__(self):
        pass

    def generate_grads(self, samples: List, model_dict: Dict):
        """根据样本与共享模型状态计算梯度列表（由子类实现）。

        Args:
            samples: 算法约定的样本结构。
            model_dict: 训练侧共享状态。

        Raises:
            NotImplementedError: 基类未实现。
        """
        raise NotImplementedError

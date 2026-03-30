# -*- coding: utf-8 -*-
"""
Base classes for RL algorithm networks, agents, and gradient calculators.
Extracted from algo_envs/algo_base.py.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

from rl_server.core.noisy import NoisyLinear


def layer_init(linear_layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0, method: str = 'orthogonal') -> nn.Linear:
    """
    Initialize a linear layer's weights and biases.

    Args:
        linear_layer: The linear layer to initialize.
        std: Standard deviation for weight initialization (default sqrt(2)).
        bias_const: Initial constant value for biases (default 0.0).
        method: Initialization method ('orthogonal' or 'kaiming').

    Returns:
        The initialized linear layer.
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
    """Base class for RL algorithm networks."""

    def __init__(self):
        super(AlgoBaseNet, self).__init__()

    def forward(self, states):
        raise NotImplementedError

    def update_state(self, version: int, grads_buffer: List):
        raise NotImplementedError


class AlgoBaseAgent:
    """Base class for RL algorithm agents (sampling does not use GPU)."""

    def __init__(self):
        pass

    def sample_multi_envs(self, model_dict: Dict):
        raise NotImplementedError

    def check_single_env(self):
        raise NotImplementedError


class AlgoBaseCalculate:
    """Base class for RL algorithm gradient calculators."""

    def __init__(self):
        pass

    def generate_grads(self, samples: List, model_dict: Dict):
        raise NotImplementedError

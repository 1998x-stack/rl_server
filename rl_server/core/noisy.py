# -*- coding: utf-8 -*-
"""
Noisy layers and gradient scaling utilities for exploration.
Extracted from algo_envs/algo_base.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class NoisyLinear(nn.Linear):
    """
    Noisy linear layer that adds parametric noise to weights and biases
    during training to encourage exploration.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        # Initialize weight noise parameters
        w = torch.full(
            size=(out_features, in_features),
            fill_value=sigma_init
        )
        self.sigma_weight = nn.Parameter(w)

        # Register buffer for weight noise
        z = torch.zeros(out_features, in_features)
        self.register_buffer(name="epsilon_weight", tensor=z)

        # If using bias, initialize bias noise parameters
        if bias:
            w = torch.full(size=(out_features,), fill_value=sigma_init)
            self.sigma_bias = nn.Parameter(w)

            # Register buffer for bias noise
            z = torch.zeros(out_features)
            self.register_buffer(name="epsilon_bias", tensor=z)

        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset weights and biases using uniform distribution in [-std, std]."""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor):
        # If not training, use standard linear forward
        if not self.training:
            return super(NoisyLinear, self).forward(input)

        # Add bias noise if applicable
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        # Add weight noise and compute output
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        """Sample new noise for weights and biases."""
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()


class GradCoef(autograd.Function):
    """
    A custom autograd function to scale gradients during backpropagation.
    Leaves the forward pass unchanged but scales gradients by a coefficient
    during the backward pass.
    """

    @staticmethod
    def forward(ctx: autograd.Function, x: torch.Tensor, coeff: float):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: torch.Tensor):
        return ctx.coeff * grad_output, None

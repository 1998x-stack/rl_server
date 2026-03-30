# -*- coding: utf-8 -*-
"""带噪线性层与自定义梯度缩放：用于探索与分布式训练中的梯度处理。

从 ``algo_envs/algo_base.py`` 抽取。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class NoisyLinear(nn.Linear):
    """因子化高斯噪声线性层：训练时对权重与偏置注入可学习尺度的随机噪声以增强探索。

    推理模式（``eval()``）下退化为普通线性层。
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        """构建带噪线性层。

        Args:
            in_features: 输入维度。
            out_features: 输出维度。
            sigma_init: 噪声标准差参数的初始值。
            bias: 是否使用可学习偏置及偏置噪声。
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        w = torch.full(
            size=(out_features, in_features),
            fill_value=sigma_init
        )
        self.sigma_weight = nn.Parameter(w)

        z = torch.zeros(out_features, in_features)
        self.register_buffer(name="epsilon_weight", tensor=z)

        if bias:
            w = torch.full(size=(out_features,), fill_value=sigma_init)
            self.sigma_bias = nn.Parameter(w)

            z = torch.zeros(out_features)
            self.register_buffer(name="epsilon_bias", tensor=z)

        self.reset_parameters()

    def reset_parameters(self):
        """使用均匀分布 ``U(-std, std)`` 初始化主权重与偏置，其中 ``std = sqrt(3/in_features)``。"""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor):
        """前向传播：训练时使用当前采样的噪声；评估时使用确定性线性变换。

        Args:
            input: 输入张量，形状 ``(batch, in_features)``。

        Returns:
            线性输出，形状 ``(batch, out_features)``。
        """
        if not self.training:
            return super(NoisyLinear, self).forward(input)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        """为权重与偏置噪声缓冲区重新采样标准正态分布。"""
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()


class GradCoef(autograd.Function):
    """自定义 autograd：前向恒等，反向将梯度乘以给定系数。"""
    @staticmethod
    def forward(ctx: autograd.Function, x: torch.Tensor, coeff: float):
        """前向：原样返回 ``x``。

        Args:
            ctx: 上下文，用于保存 ``coeff``。
            x: 输入张量。
            coeff: 反向时乘以的标量系数。

        Returns:
            与 ``x`` 同形状的张量视图。
        """
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx: autograd.Function, grad_output: torch.Tensor):
        """反向：``grad_input = coeff * grad_output``。

        Args:
            ctx: 保存的上下文。
            grad_output: 上游梯度。

        Returns:
            ``(coeff * grad_output, None)``，不对 ``coeff`` 求导。
        """
        return ctx.coeff * grad_output, None

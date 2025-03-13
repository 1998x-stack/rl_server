# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import math 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class NoisyLinear(nn.Linear):
    """
    带有噪声的全连接层，继承自nn.Linear。
    该层在训练时会在权重和偏置上添加噪声，以增加模型的探索能力。
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        """
        初始化NoisyLinear层。

        参数:
        - in_features: 输入特征的数量。
        - out_features: 输出特征的数量。
        - sigma_init: 噪声的初始标准差，默认为0.017。
        - bias: 是否使用偏置项，默认为True。
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias) # 继承nn.Linear方法
        
        # 初始化权重噪声参数
        w = torch.full(
                size=(out_features, in_features), 
                fill_value=sigma_init
            )
        
        self.sigma_weight = nn.Parameter(w)
        
        # 注册缓冲区用于存储权重噪声
        z = torch.zeros(out_features, in_features)
        self.register_buffer(name="epsilon_weight", tensor=z)
        
        # 如果使用偏置，初始化偏置噪声参数
        if bias:
            w = torch.full(size=(out_features,), fill_value=sigma_init)
            self.sigma_bias = nn.Parameter(w)
            
            # 注册缓冲区用于存储偏置噪声
            z = torch.zeros(out_features)
            self.register_buffer(name="epsilon_bias", tensor=z)
            
        # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置权重和偏置的初始值。
        使用均匀分布初始化权重和偏置，范围在[-std, std]之间，其中std = sqrt(3 / in_features)。
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        """
        前向传播函数。

        参数:
        - input: 输入张量。

        返回:
        - 输出张量。
        """
        # 如果不是训练模式，直接调用父类的forward方法
        if not self.training:
            return super(NoisyLinear, self).forward(input)
        
        # 如果有偏置，添加偏置噪声
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data
            
        # 添加权重噪声并计算输出
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        """
        采样新的噪声。
        该方法会重新生成权重和偏置的噪声。
        """
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()

def layer_init(linear_layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0, method: str='orthogonal') -> nn.Linear:
    """
    初始化线性层的权重和偏置。

    参数:
    - layer (nn.Linear): 需要初始化的线性层。
    - std (float): 权重初始化的标准差，默认为 $\sqrt{2}$。
    - bias_const (float): 偏置的初始常数值，默认为 0.0。
    - method

    返回:
    - nn.Linear: 初始化后的线性层。
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(linear_layer.weight, std)  # 使用正交初始化权重
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, bias_const)  # 使用常数初始化偏置
    else:
        nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias, bias_const)  # 使用常数初始化偏置

    return linear_layer
        
class AlgoBaseNet(nn.Module):
    """RL算法基础的网络
    """
    def __init__(self):
        super(AlgoBaseNet,self).__init__()
              
    def forward(self, states):
        raise NotImplementedError
        
    def update_state(self, version, grads_buffer):
        raise NotImplementedError
    
#采样不使用GPU
class AlgoBaseAgent:
    """RL算法基础的智能体
    """
    def __init__(self):
        pass
    
    def sample_multi_envs(self, model_dict):
        raise NotImplementedError
    
    def check_single_env(self):
        raise NotImplementedError
        
class AlgoBaseCalculate:
    """RL算法基础的推理
    """
    def __init__(self):
        pass
    
    def generate_grads(self, samples, model_dict):
        raise NotImplementedError
    
class GradCoef(autograd.Function):
    """
    A custom autograd function to scale gradients during backpropagation.
    This function allows you to scale the gradients by a specified coefficient
    during the backward pass, while leaving the forward pass unchanged.
    Methods
    -------
    forward(ctx, x, coeff)
        Performs the forward pass of the function. Stores the coefficient in the context.
    backward(ctx, grad_output)
        Performs the backward pass of the function. Scales the gradient by the stored coefficient.
    Parameters
    ----------
    ctx : autograd.Function
        The context object that can be used to stash information for backward computation.
    x : torch.Tensor
        The input tensor.
    coeff : float
        The coefficient by which to scale the gradients during the backward pass.
    Returns
    -------
    torch.Tensor
        The output tensor, which is a view of the input tensor.
    """
    # 模型前向
    @staticmethod
    def forward(ctx: autograd.Function, x: torch.Tensor, coeff: float):
        # 将coeff存为ctx的成员变量           
        ctx.coeff = coeff
        return x.view_as(x)

    # 模型梯度反传
    @staticmethod
    def backward(ctx: autograd.Function, grad_output: torch.Tensor):
        # backward的输出个数，应与forward的输入个数相同
        # 此处coeff不需要梯度，因此返回None    
        return ctx.coeff * grad_output, None
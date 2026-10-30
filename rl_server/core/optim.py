# -*- coding: utf-8 -*-
"""参数与优化器工具：可训练参数筛选、Adam 复用。

两个容易踩的坑都在这里收口：

1. **每次 update_state 都新建 Adam** 会把一阶/二阶矩清零，Adam 退化成"带符号的 SGD"，
   学习率语义、偏差修正全部失效。优化器必须按网络实例缓存。
2. **不可训练参数**（例如 SAC 的目标 critic）不能进入梯度聚合：聚合契约要求
   ``len(grads) == len(params)``，混进目标网络会让整批梯度因长度不匹配被拒。
"""
from typing import Iterable, List, Sequence

import numpy as np
import torch


def trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """返回参与梯度聚合的参数（按 ``model.parameters()`` 的顺序，仅 ``requires_grad`` 者）。

    Args:
        model: 任意含有参数的模块。

    Returns:
        参数列表；顺序与 ``model.parameters()`` 一致，可直接与梯度列表 zip。
    """
    return [param for param in model.parameters() if param.requires_grad]


def assign_gradients(params: Sequence[torch.nn.Parameter], grads: Sequence) -> int:
    """把 numpy 梯度写入参数 ``.grad``。

    ``torch.FloatTensor(grad)`` 在 0 维参数（如 SAC 的 ``log_alpha``）上会抛
    ``TypeError``，这里统一用 ``torch.as_tensor`` 转换。

    Args:
        params: 目标参数（顺序需与 ``grads`` 一致）。
        grads: numpy 数组序列。

    Returns:
        写入的参数个数。

    Raises:
        ValueError: 长度不一致（聚合契约应已在更外层拦截）。
    """
    params = list(params)
    if len(params) != len(grads):
        raise ValueError(f"gradient count {len(grads)} != parameter count {len(params)}")
    for param, grad in zip(params, grads):
        param.grad = torch.as_tensor(np.asarray(grad), dtype=torch.float32).clone()
    return len(params)


def _optimizer_fingerprint(params: Sequence[torch.nn.Parameter], lr: float):
    """用于判断缓存是否需要重建：参数身份、设备与学习率变化都要重建。"""
    return (
        tuple(id(param) for param in params),
        tuple(str(param.device) for param in params),
        float(lr),
    )


def get_cached_adam(
    owner: torch.nn.Module,
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    **kwargs,
) -> torch.optim.Adam:
    """取（或创建）挂在 ``owner`` 上的 Adam 优化器，保持矩估计跨调用有效。

    Args:
        owner: 持有缓存的模块（通常是网络本身）。
        name: 缓存名，便于同一模块挂多个优化器（如 SAC 的 policy/critic/alpha）。
        params: 待优化参数。
        lr: 学习率。
        **kwargs: 传给 ``torch.optim.Adam`` 的其它参数（仅首次创建时生效）。

    Returns:
        缓存或新建的 ``torch.optim.Adam``。
    """
    params = list(params)
    cache = owner.__dict__.setdefault('_rl_optimizers', {})
    fingerprint = _optimizer_fingerprint(params, lr)

    cached = cache.get(name)
    if cached is not None:
        stored_fingerprint, optimizer = cached
        if stored_fingerprint == fingerprint:
            return optimizer

    optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
    cache[name] = (fingerprint, optimizer)
    return optimizer


def default_device(model: torch.nn.Module) -> torch.device:
    """返回模型首个参数所在设备（模型无参数时回退 CPU）。"""
    for param in model.parameters():
        return param.device
    return torch.device('cpu')

# -*- coding: utf-8 -*-
"""梯度批次契约：判断一批梯度能否用于当前模型版本，并说明拒绝原因。

本地主循环与 Redis 聚合服务器共用同一份判定，避免两边规则漂移。返回值同时给出
**可观测的原因**，让 "Rejected N stale or invalid gradient batches" 这类无法定位的
日志变成可统计的分项计数。
"""
import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch

# 允许的滞后上限（自动模式下用于兜底，避免接受极其陈旧的梯度）。
DEFAULT_LAG_CAP = 64

# ``max_gradient_lag`` 允许配置为该字符串，表示按实际吞吐自动推导。
AUTO_LAG = "auto"

# 拒绝原因常量：既用于日志聚合，也用于测试断言。
REASON_VERSION_TYPE = "version_type"
REASON_LAG_INVALID = "max_lag_invalid"
REASON_STALE = "stale_version"
REASON_FUTURE = "future_version"
REASON_NOT_SEQUENCE = "not_sequence"
REASON_LENGTH_MISMATCH = "length_mismatch"
REASON_NONE_GRAD = "none_gradient"
REASON_SHAPE_MISMATCH = "shape_mismatch"
REASON_BAD_DTYPE = "non_float_dtype"
REASON_NON_FINITE = "non_finite"


def is_auto_lag(configured) -> bool:
    """配置值是否表示"自动推导滞后上限"。"""
    return isinstance(configured, str) and configured.strip().lower() == AUTO_LAG


def resolve_max_gradient_lag(
    configured,
    window: Optional[float] = None,
    cap: int = DEFAULT_LAG_CAP,
) -> int:
    """把 ``max_gradient_lag`` 配置解析成具体整数。

    同一批样本往往一次产出多条梯度（PPO 按 minibatch、SAC 按 critic/actor 多次反传），
    而主循环每应用一条就把版本推高一格：只要 ``max_gradient_lag`` 小于这个条数，
    同批里靠后的梯度一出生就会被判为过期。手动填错就白烧算力，所以支持 ``auto``：

    Args:
        configured: 整数或 ``"auto"``。
        window: 一个采样/聚合窗口内预期产生的梯度条数（auto 模式使用）。
        cap: auto 模式下的上限。

    Returns:
        实际使用的 ``max_gradient_lag``（非负整数）。
    """
    if not is_auto_lag(configured):
        if configured is None:
            return 0
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise TypeError(f"max_gradient_lag must be int or 'auto', got {configured!r}")
        return configured
    if window is None:
        return 1
    return min(int(cap), max(1, int(math.ceil(float(window))) - 1))


def gradient_batch_rejection(
    grads: Sequence[np.ndarray],
    version: int,
    params: Sequence[torch.nn.Parameter],
    current_version: int,
    max_lag: int = 0,
) -> Optional[str]:
    """检查一批梯度是否可用于 ``current_version``。

    Args:
        grads: 梯度数组序列，长度需与 ``params`` 一致。
        version: 该批次生成时对应的训练版本。
        params: 目标模型参数，用于校验长度、形状与数据类型。
        current_version: 当前模型版本。
        max_lag: 允许的最大滞后版本数；``0`` 表示只接受当前版本的梯度。

    Returns:
        ``None`` 表示可用；否则返回 :data:`REASON_*` 常量之一。
    """
    if type(version) is not int:
        return REASON_VERSION_TYPE
    if max_lag < 0:
        return REASON_LAG_INVALID
    if version > current_version:
        return REASON_FUTURE
    if version < current_version - max_lag:
        return REASON_STALE
    if not isinstance(grads, (list, tuple)):
        return REASON_NOT_SEQUENCE
    if len(grads) != len(params):
        return REASON_LENGTH_MISMATCH
    for grad, param in zip(grads, params):
        if grad is None:
            return REASON_NONE_GRAD
        array = np.asarray(grad)
        if array.shape != tuple(param.shape):
            return REASON_SHAPE_MISMATCH
        if not np.issubdtype(array.dtype, np.floating):
            return REASON_BAD_DTYPE
        if not np.isfinite(array).all():
            return REASON_NON_FINITE
    return None


def accepts_gradient_batch(
    grads: Sequence[np.ndarray],
    version: int,
    params: Sequence[torch.nn.Parameter],
    current_version: int,
    max_lag: int = 0,
) -> bool:
    """``gradient_batch_rejection`` 的布尔封装，保持旧调用方不变。"""
    return gradient_batch_rejection(
        grads, version, params, current_version, max_lag
    ) is None

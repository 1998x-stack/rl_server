# -*- coding: utf-8 -*-
"""Redis 传输层序列化：使用 pickle 与 zlib 压缩减小载荷。"""
import io
import pickle
import zlib
from typing import Any, List

import numpy as np


def serialize(obj: Any) -> bytes:
    """将任意可 pickle 对象序列化为压缩字节串。

    Args:
        obj: 待序列化对象。

    Returns:
        zlib 压缩后的 pickle 字节。
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(data)


def deserialize(data: bytes) -> Any:
    """解压并反序列化 ``serialize`` 产生的字节串。

    Args:
        data: 压缩后的 pickle 字节。

    Returns:
        原始 Python 对象。

    Raises:
        各类 pickle/zlib 相关异常：数据损坏或格式非法时。
    """
    return pickle.loads(zlib.decompress(data))


def serialize_gradients(grads_list: List[np.ndarray]) -> bytes:
    """将梯度数组列表用 ``np.savez_compressed`` 序列化（比 pickle+zlib 小约 30%）。

    Args:
        grads_list: 梯度 ndarray 列表。

    Returns:
        压缩后的字节串。
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, *grads_list)
    return buf.getvalue()


def deserialize_gradients(data: bytes) -> List[np.ndarray]:
    """反序列化 ``serialize_gradients`` 产生的字节串。

    Args:
        data: ``serialize_gradients`` 的输出。

    Returns:
        ndarray 列表。
    """
    buf = io.BytesIO(data)
    with np.load(buf, allow_pickle=False) as loaded:
        return [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

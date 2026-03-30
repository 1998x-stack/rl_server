# -*- coding: utf-8 -*-
"""Redis 传输层序列化：使用 pickle 与 zlib 压缩减小载荷。"""
import pickle
import zlib
from typing import Any


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

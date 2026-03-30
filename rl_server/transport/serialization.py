# -*- coding: utf-8 -*-
"""Serialization utilities for Redis transport."""
import pickle
import zlib
from typing import Any


def serialize(obj: Any) -> bytes:
    """Serialize object using pickle + zlib compression."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(data)


def deserialize(data: bytes) -> Any:
    """Deserialize object from compressed pickle bytes."""
    return pickle.loads(zlib.decompress(data))

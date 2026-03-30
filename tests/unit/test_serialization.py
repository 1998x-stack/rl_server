# -*- coding: utf-8 -*-
"""Tests for rl_server.transport.serialization module."""
import numpy as np
import torch
import pytest

from rl_server.transport.serialization import serialize, deserialize


class TestSerialization:

    def test_round_trip_dict(self):
        data = {'key': 'value', 'num': 42, 'nested': [1, 2, 3]}
        result = deserialize(serialize(data))
        assert result == data

    def test_round_trip_numpy(self):
        arr = np.random.randn(10, 5).astype(np.float32)
        result = deserialize(serialize(arr))
        np.testing.assert_array_equal(result, arr)

    def test_round_trip_tensor(self):
        t = torch.randn(4, 4)
        result = deserialize(serialize(t))
        assert torch.allclose(result, t)

    def test_serialize_produces_bytes(self):
        data = {'hello': 'world'}
        out = serialize(data)
        assert isinstance(out, bytes)

    def test_deserialize_corrupt_data_raises(self):
        with pytest.raises(Exception):
            deserialize(b'not valid compressed pickle data')

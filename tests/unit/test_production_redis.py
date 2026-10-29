"""Regression tests for Redis isolation and coherent model synchronization."""
import zlib

import pytest
import torch

fakeredis = pytest.importorskip("fakeredis")
from rl_server.transport.redis_cache import RedisCache
from rl_server.utils.logging import Log


def cache(client, namespace="run_a"):
    instance = RedisCache.__new__(RedisCache)
    instance.log = Log("test_prod_redis")
    instance.redis_config = {"namespace": namespace}
    instance.namespace = namespace
    instance.conn = client
    instance._pop_exps_timeouts = 0
    instance._pop_grads_timeouts = 0
    return instance


def test_clear_data_does_not_flush_other_namespace_or_unrelated_keys():
    client = fakeredis.FakeRedis()
    first, second = cache(client, "first"), cache(client, "second")
    client.set("important_unrelated_data", "keep")
    first.set_exit_flag(True)
    second.set_exit_flag(True)
    first.clear_data()
    assert first.get_exit_flag() is None
    assert second.get_exit_flag() == 1
    assert client.get("important_unrelated_data") == b"keep"


def test_model_snapshot_version_and_weights_are_consistent():
    client = fakeredis.FakeRedis()
    transport = cache(client)
    source = torch.nn.Linear(4, 2)
    with torch.no_grad():
        source.weight.fill_(5.0)
    assert transport.set_train_version_model(9, source)
    target = torch.nn.Linear(4, 2)
    assert transport.get_train_snapshot(target) == 9
    assert transport.get_train_version() == 9
    assert torch.equal(target.weight, source.weight)


def test_corrupted_model_payload_does_not_advance_version():
    client = fakeredis.FakeRedis()
    transport = cache(client)
    client.set("run_a:train_snapshot", zlib.compress(b"not a torch checkpoint"))
    assert transport.get_train_snapshot(torch.nn.Linear(4, 2)) is None


def test_old_pickle_only_model_key_never_loaded():
    client = fakeredis.FakeRedis()
    transport = cache(client)
    client.set("run_a:train_model", zlib.compress(b"untrusted legacy pickle"))
    assert transport.get_train_model(torch.nn.Linear(4, 2)) is False


def test_run_scoping_applies_to_exps_and_grads():
    import numpy as np
    client = fakeredis.FakeRedis()
    transport = cache(client)
    assert transport.push_exps([[1, 2, 3]], 1)
    assert transport.push_grads([np.zeros((2,), dtype=np.float32)], 1, 1)
    assert client.llen("run_a:exps") == 1
    assert client.llen("run_a:grads") == 1
    assert client.llen("exps") == 0
    assert client.llen("grads") == 0

import os
import sys
import torch
import pytest
import pickle
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from libs.log import Log


@pytest.fixture
def fake_redis_cache():
    if not HAS_FAKEREDIS:
        pytest.skip("fakeredis not installed")
    from libs.redis_cache import RedisCache
    log = Log("test_redis")
    config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
    cache = RedisCache.__new__(RedisCache)
    cache.log = log
    cache.redis_config = config
    cache.conn = fakeredis.FakeRedis()
    return cache


class TestRedisCache:
    def test_set_and_get_exit_flag(self, fake_redis_cache):
        cache = fake_redis_cache
        cache.set_exit_flag(True)
        flag = cache.get_exit_flag()
        assert flag == 1

    def test_set_exit_flag_false(self, fake_redis_cache):
        cache = fake_redis_cache
        cache.set_exit_flag(False)
        flag = cache.get_exit_flag()
        assert flag == 0

    def test_get_exit_flag_when_not_set(self, fake_redis_cache):
        cache = fake_redis_cache
        flag = cache.get_exit_flag()
        assert flag is None

    def test_set_and_get_train_version_model(self, fake_redis_cache):
        cache = fake_redis_cache
        model = torch.nn.Linear(4, 2)
        cache.set_train_version_model(5, model)
        version = cache.get_train_version()
        assert version == 5

    def test_get_train_model_loads_state(self, fake_redis_cache):
        cache = fake_redis_cache
        model = torch.nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        cache.set_train_version_model(1, model)

        fresh_model = torch.nn.Linear(4, 2)
        result = cache.get_train_model(fresh_model)
        assert result is True
        assert torch.allclose(original_weight, fresh_model.weight.data)

    def test_get_train_model_when_not_set(self, fake_redis_cache):
        cache = fake_redis_cache
        model = torch.nn.Linear(4, 2)
        result = cache.get_train_model(model)
        assert result is False

    def test_get_train_version_when_not_set(self, fake_redis_cache):
        cache = fake_redis_cache
        version = cache.get_train_version()
        assert version is None

    def test_push_and_pop_exps(self, fake_redis_cache):
        cache = fake_redis_cache
        exps = [[1, 2, 3], [4, 5, 6]]
        cache.push_exps(exps, sample_version=1)
        assert cache.conn.llen('exps') == 1

    def test_push_and_pop_grads(self, fake_redis_cache):
        cache = fake_redis_cache
        grads = [[0.1, 0.2], [0.3, 0.4]]
        cache.push_grads(grads, grads_version=2, sample_version=1)
        assert cache.conn.llen('grads') == 1

    def test_clear_data(self, fake_redis_cache):
        cache = fake_redis_cache
        cache.conn.set('test_key', 'test_value')
        cache.clear_data()
        assert cache.conn.get('test_key') is None

    def test_health_check(self, fake_redis_cache):
        cache = fake_redis_cache
        assert cache.health_check() is True

    def test_version_name_constants(self):
        from libs.redis_cache import RedisCache
        assert RedisCache.version_name == 'version'
        assert RedisCache.model_name == 'model'
        assert RedisCache.exps_name == 'exps'
        assert RedisCache.exit_flag_name == 'exit'
        assert RedisCache.train_version_name == 'TRAIN_VERSION'
        assert RedisCache.grads_name == 'grads'

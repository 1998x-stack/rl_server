import pytest
import torch

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False


@pytest.mark.integration
class TestRedisPipeline:
    @pytest.fixture
    def fake_cache(self):
        if not HAS_FAKEREDIS:
            pytest.skip("fakeredis not installed")
        from rl_server.transport.redis_cache import RedisCache
        from rl_server.utils.logging import Log
        log = Log("test")
        cache = RedisCache.__new__(RedisCache)
        cache.log = log
        cache.redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
        cache.conn = fakeredis.FakeRedis()
        return cache

    def test_model_sync_roundtrip(self, fake_cache):
        model = torch.nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        fake_cache.set_train_version_model(1, model)
        version = fake_cache.get_train_version()
        assert version == 1

        fresh = torch.nn.Linear(4, 2)
        assert fake_cache.get_train_model(fresh)
        assert torch.allclose(original_weight, fresh.weight.data)

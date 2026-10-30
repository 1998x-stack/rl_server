import numpy as np
import fakeredis

from rl_server.algorithms import create_net
from rl_server.transport.redis_cache import RedisCache
from rl_server.utils.logging import Log
from rl_server.entrypoints import trainer as trainer_entrypoint


def test_redis_trainer_consumes_experiences_and_publishes_versioned_gradients(monkeypatch):
    server = fakeredis.FakeServer()
    log = Log("test_redis_trainer")
    namespace = "trainer_test"

    def cache_for_db(db):
        cache = RedisCache.__new__(RedisCache)
        cache.log = log
        cache.namespace = namespace
        cache.redis_config = {"namespace": namespace}
        cache.conn = fakeredis.FakeRedis(server=server, db=db)
        cache._pop_exps_timeouts = 0
        cache._pop_grads_timeouts = 0
        return cache

    monkeypatch.setattr(
        trainer_entrypoint, "RedisCache",
        lambda _log, config: cache_for_db(int(config["db"])),
    )
    model_cache = cache_for_db(0)
    exps_cache = cache_for_db(1)
    grads_cache = cache_for_db(2)
    assert model_cache.set_train_version_model(3, create_net("DQNGymClassic"))
    samples = [
        [np.zeros(4, dtype=np.float32), 0, 1.0, i == 10, 3]
        for i in range(300)
    ]
    assert exps_cache.push_exps(samples, 3)

    trainer = trainer_entrypoint.RedisTrainer(
        "DQNGymClassic", {
            name: {"host": "localhost", "port": 6379, "db": db,
                   "password": "", "namespace": namespace}
            for db, name in enumerate(("model", "exps", "grads"))
        }, log,
    )
    trainer.calculator.num_repeat = 1
    assert trainer.run_once()

    gradients, grads_version, sample_version = grads_cache.pop_grads()
    assert grads_version == sample_version == 3
    assert len(gradients) == len(list(trainer.model.parameters()))
    assert bool(trainer.calculator.exps_buffer.buffer[10].done)

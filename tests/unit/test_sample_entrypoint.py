from argparse import Namespace
from unittest.mock import MagicMock

import torch

from rl_server.entrypoints import sample


def test_sampler_loads_initial_version_zero_snapshot(monkeypatch):
    model_cache = MagicMock()
    model_cache.get_exit_flag.return_value = 0
    model_cache.get_train_version.return_value = 0
    model_cache.get_train_snapshot.return_value = 0
    agent = MagicMock()
    agent.sample_multi_envs.return_value = None

    monkeypatch.setattr(sample, 'parse_args', lambda: Namespace(config=None, override=None, env_name=None))
    monkeypatch.setattr(sample, 'setup_seed', lambda: None)
    monkeypatch.setattr(sample, 'setup_signal_handlers', lambda: None)
    monkeypatch.setattr(sample, 'load_config', lambda *_: {
        'training': {'env_name': 'DQNGymClassic'},
        'redis': {
            'model': {'db': 0}, 'exps': {'db': 1},
        },
    })
    log_stub = MagicMock()
    monkeypatch.setattr(sample, 'Log', MagicMock(from_config=lambda *_: log_stub))
    monkeypatch.setattr(sample, 'create_net', lambda *_: torch.nn.Linear(4, 2))
    monkeypatch.setattr(sample, 'create_agent', lambda *_: agent)
    monkeypatch.setattr(sample, 'RedisCache', lambda _log, config: model_cache if config['db'] == '0' else MagicMock())
    exits = iter([False, True])
    monkeypatch.setattr(sample, 'should_exit', lambda: next(exits))
    monkeypatch.setattr(sample.time, 'sleep', lambda _: None)

    sample.main()

    assert model_cache.get_train_snapshot.call_count == 1

# -*- coding: utf-8 -*-
"""Tests for worker construction and safe teardown (no process spawning)."""
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from rl_server.utils.logging import Log


class TestSamplerWorker:

    def test_init_stores_attributes(self):
        from rl_server.workers.sampler import SamplerWorker
        log = Log("test")
        model_dict = {'is_exit': False, 'TRAIN_VERSION': 0}
        share_model = nn.Linear(4, 2)
        sample_queue = MagicMock()

        worker = SamplerWorker(
            idx=0, model_dict=model_dict, share_model=share_model,
            sample_queue=sample_queue, env_name='DQNGymClassic', log=log
        )
        assert worker.sampler_id == 0
        assert worker.env_name == 'DQNGymClassic'
        assert worker.process is None

    def test_stop_without_start_is_safe(self):
        from rl_server.workers.sampler import SamplerWorker
        log = Log("test")
        worker = SamplerWorker(
            idx=0, model_dict={}, share_model=nn.Linear(4, 2),
            sample_queue=MagicMock(), env_name='DQNGymClassic', log=log
        )
        # Should not raise
        worker.stop()


class TestTrainerWorker:

    def test_init_stores_attributes(self):
        from rl_server.workers.trainer import TrainerWorker
        log = Log("test")
        model_dict = {'is_exit': False, 'TRAIN_VERSION': 0}
        share_model = nn.Linear(4, 2)

        worker = TrainerWorker(
            idx=1, model_dict=model_dict, share_model=share_model,
            sample_queue=MagicMock(), grads_queue=MagicMock(),
            env_name='DQNGymClassic', log=log
        )
        assert worker.trainer_id == 1
        assert worker.env_name == 'DQNGymClassic'
        assert worker.process is None

    def test_stop_without_start_is_safe(self):
        from rl_server.workers.trainer import TrainerWorker
        log = Log("test")
        worker = TrainerWorker(
            idx=0, model_dict={}, share_model=nn.Linear(4, 2),
            sample_queue=MagicMock(), grads_queue=MagicMock(),
            env_name='DQNGymClassic', log=log
        )
        worker.stop()


class TestCheckerWorker:

    def test_init_stores_attributes(self):
        from rl_server.workers.checker import CheckerWorker
        log = Log("test")
        model_dict = {'is_exit': False, 'TRAIN_VERSION': 0}
        share_model = nn.Linear(4, 2)

        worker = CheckerWorker(
            model_dict=model_dict, share_model=share_model,
            env_name='DQNGymClassic', log=log
        )
        assert worker.env_name == 'DQNGymClassic'
        assert worker.process is None
        assert worker.self_version == 0

    def test_stop_without_start_is_safe(self):
        from rl_server.workers.checker import CheckerWorker
        log = Log("test")
        worker = CheckerWorker(
            model_dict={'is_exit': False, 'TRAIN_VERSION': 0},
            share_model=nn.Linear(4, 2),
            env_name='DQNGymClassic', log=log
        )
        worker.stop()


class TestGradsAggregatorInit:

    def test_init_with_mock_redis(self):
        """Test GradsAggregator construction with mocked Redis."""
        from unittest.mock import patch
        from rl_server.workers.grads_aggregator import GradsAggregator

        redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': '', 'pool_size': '2'}

        with patch('rl_server.workers.grads_aggregator.RedisCache') as MockRedis, \
             patch('rl_server.workers.grads_aggregator.load_model', return_value=None):
            mock_instance = MagicMock()
            MockRedis.return_value = mock_instance

            agg = GradsAggregator(
                env_name='DQNGymClassic',
                model_prefix='test',
                grads_redis_config=redis_config,
                model_redis_config=redis_config,
            )
            assert agg.current_train_version == 0
            assert agg.env_name == 'DQNGymClassic'

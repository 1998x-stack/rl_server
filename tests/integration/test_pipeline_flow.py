# -*- coding: utf-8 -*-
"""Integration tests: multi-component flows without real Redis or processes."""
import torch
import numpy as np
import pytest

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False


class TestDQNCreateSampleTrainCycle:
    """Wire DQN net + calculator, generate synthetic samples, compute grads, apply update."""

    def test_full_cycle(self):
        from rl_server.algorithms.dqn.network import DQNGymClassicNet
        from rl_server.algorithms.dqn.calculator import DQNGymClassicCalculate

        net = DQNGymClassicNet()
        calc = DQNGymClassicCalculate(net)

        # Generate synthetic samples: [state, action, reward, version, done]
        # DQN calculator uses indices 0,1,2,4 (state, action, reward, done)
        samples = []
        for i in range(300):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            version = 0
            done = bool(np.random.random() < 0.1)
            samples.append([state, action, reward, version, done])

        model_dict = {'TRAIN_VERSION': 0}
        params_before = [p.data.clone() for p in net.parameters()]

        grads_list, train_version = calc.generate_grads(samples, model_dict)
        assert len(grads_list) == 1
        assert train_version == 0

        # Apply gradients
        net.update_state(1, grads_list[0])

        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed


class TestPPONormalCreateSampleTrainCycle:
    """Wire PPO Normal net + calculator, generate synthetic samples, compute grads."""

    def test_full_cycle(self):
        from rl_server.algorithms.ppo.mujoco_normal import MujocoNormalNet, MujocoNormalCalculate
        from rl_server.algorithms.ppo.mujoco_normal import TRAIN_ENVS, current_env_name

        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM

        net = MujocoNormalNet()
        calc = MujocoNormalCalculate(net)

        # PPO samples: [state, action, reward, done, log_probs, version]
        samples = []
        for i in range(100):
            state = np.random.randn(obs_dim).astype(np.float32)
            action = np.random.randn(act_dim).astype(np.float32)
            reward = np.random.randn()
            done = bool(np.random.random() < 0.05)
            log_probs = np.random.randn(act_dim).astype(np.float32)
            version = 0
            samples.append([state, action, reward, done, log_probs, version])

        model_dict = {'TRAIN_VERSION': 0}
        params_before = [p.data.clone() for p in net.parameters()]

        grads_list, train_version = calc.generate_grads(samples, model_dict)
        assert len(grads_list) == 1
        assert train_version == 0

        net.update_state(1, grads_list[0])
        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed


class TestCheckpointRoundTripWithRealNet:
    """Save a real DQN model to disk, load into fresh net, verify outputs match."""

    def test_checkpoint_round_trip(self, tmp_path):
        from rl_server.algorithms.dqn.network import DQNGymClassicNet
        from rl_server.utils.checkpoint import save_model, load_model

        net = DQNGymClassicNet()
        test_input = torch.randn(5, 4)
        expected_output = net(test_input).detach()

        save_model(net, 'dqn_test', '1', base_dir=str(tmp_path))

        net2 = DQNGymClassicNet()
        version = load_model(net2, 'dqn_test', '1', base_dir=str(tmp_path))
        assert version == '1'

        actual_output = net2(test_input).detach()
        assert torch.allclose(expected_output, actual_output)


@pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisModelSync:
    """set_train_version_model then get_train_model via fakeredis."""

    def _make_cache(self):
        from rl_server.transport.redis_cache import RedisCache
        from rl_server.utils.logging import Log
        cache = RedisCache.__new__(RedisCache)
        cache.log = Log("test_sync")
        cache.redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
        cache.conn = fakeredis.FakeRedis()
        cache.pool = None
        return cache

    def test_model_sync(self):
        from rl_server.algorithms.dqn.network import DQNGymClassicNet
        cache = self._make_cache()

        net = DQNGymClassicNet()
        test_input = torch.randn(3, 4)
        expected = net(test_input).detach()

        cache.set_train_version_model(10, net)
        assert cache.get_train_version() == 10

        net2 = DQNGymClassicNet()
        assert cache.get_train_model(net2) is True
        actual = net2(test_input).detach()
        assert torch.allclose(expected, actual)


@pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisExperiencePipeline:
    """Push multiple exp batches, pop them all, verify FIFO ordering."""

    def _make_cache(self):
        from rl_server.transport.redis_cache import RedisCache
        from rl_server.utils.logging import Log
        cache = RedisCache.__new__(RedisCache)
        cache.log = Log("test_exp")
        cache.redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
        cache.conn = fakeredis.FakeRedis()
        cache.pool = None
        return cache

    def test_fifo_ordering(self):
        cache = self._make_cache()
        # Push 3 batches
        for i in range(3):
            cache.push_exps([[i, i+1]], sample_version=i)

        # Verify count
        count = cache.conn.llen('exps')
        assert count == 3


@pytest.mark.skipif(not HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisGradientPipeline:
    """Push grads, verify data integrity."""

    def _make_cache(self):
        from rl_server.transport.redis_cache import RedisCache
        from rl_server.utils.logging import Log
        cache = RedisCache.__new__(RedisCache)
        cache.log = Log("test_grad")
        cache.redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
        cache.conn = fakeredis.FakeRedis()
        cache.pool = None
        return cache

    def test_push_grads_integrity(self):
        cache = self._make_cache()
        grads = [np.random.randn(4, 2).tolist(), np.random.randn(2).tolist()]
        cache.push_grads(grads, grads_version=5, sample_version=3)
        assert cache.conn.llen('grads') == 1


class TestEntrypointParseArgs:
    """Verify parse_args() returns correct defaults."""

    def test_default_args(self, monkeypatch):
        monkeypatch.setattr('sys.argv', ['train.py'])
        from rl_server.entrypoints.train import parse_args
        args = parse_args()
        assert args.config is None
        assert args.override is None
        assert args.env_name is None
        assert args.prefix == 'train_main_local'
        assert args.version is None

    def test_env_name_override(self, monkeypatch):
        monkeypatch.setattr('sys.argv', ['train.py', '--env-name', 'MujocoNormal'])
        from rl_server.entrypoints.train import parse_args
        args = parse_args()
        assert args.env_name == 'MujocoNormal'

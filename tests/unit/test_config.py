import os
import sys
import types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock missing optional dependencies before importing config
# gym_microrts is not available in test environments
if 'gym_microrts' not in sys.modules:
    mock_microrts = types.ModuleType('gym_microrts')
    mock_microrts.microrts_ai = types.ModuleType('gym_microrts.microrts_ai')
    sys.modules['gym_microrts'] = mock_microrts
    sys.modules['gym_microrts.microrts_ai'] = mock_microrts.microrts_ai

# Check if ppo_microrts can be imported; if not, create a minimal mock
try:
    import algo_envs.ppo_microrts
except Exception:
    mock_module = types.ModuleType('algo_envs.ppo_microrts')
    mock_module.MicroRTSNet = type('MicroRTSNet', (), {})
    mock_module.MicroRTSAgent = type('MicroRTSAgent', (), {})
    mock_module.MicroRTSCalculate = type('MicroRTSCalculate', (), {})
    sys.modules['algo_envs.ppo_microrts'] = mock_module

import libs.config as config


class TestConfigFactories:
    def test_create_net_dqn(self):
        net = config.create_net("DQNGymClassic")
        assert net is not None
        params = list(net.parameters())
        assert len(params) > 0

    def test_create_net_mujoco_normal(self):
        net = config.create_net("MujocoNormal")
        assert net is not None
        params = list(net.parameters())
        assert len(params) > 0

    def test_create_net_invalid_raises(self):
        with pytest.raises(SystemExit):
            config.create_net("NonExistentAlgo")

    def test_get_current_env_name(self):
        name = config.get_current_env_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_queue_config_has_required_keys(self):
        qc = config.get_current_queue_config()
        required_keys = ['num_trainer', 'num_sampler', 'len_grads_queue',
                         'len_sample_queue', 'num_update_grads']
        for key in required_keys:
            assert key in qc, f"Missing key: {key}"

    def test_get_queue_config_values_are_positive(self):
        qc = config.get_current_queue_config()
        for key in ['num_trainer', 'num_sampler', 'len_grads_queue', 'len_sample_queue']:
            assert qc[key] > 0, f"{key} should be positive"

    def test_get_redis_config_has_required_keys(self):
        rc = config.get_current_redis_MODEL_CONFIG()
        required_keys = ['ip', 'port', 'db', 'pw']
        for key in required_keys:
            assert key in rc, f"Missing key: {key}"

    def test_get_redis_exps_config(self):
        rc = config.get_current_redis_exps_config()
        assert 'ip' in rc
        assert 'port' in rc

    def test_get_redis_grads_config(self):
        rc = config.get_current_redis_grads_config()
        assert 'ip' in rc
        assert 'port' in rc

    def test_create_agent_dqn(self):
        net = config.create_net("DQNGymClassic")
        agent = config.create_agent("DQNGymClassic", net, is_checker=True)
        assert agent is not None

    def test_create_calculate_dqn(self):
        net = config.create_net("DQNGymClassic")
        calc = config.create_calculate("DQNGymClassic", net)
        assert calc is not None

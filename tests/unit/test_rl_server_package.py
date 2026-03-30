"""``rl_server`` 包可导入性与关键符号存在性测试。"""
import pytest


class TestPackageImports:
    def test_core_imports(self):
        from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
        assert AlgoBaseNet is not None

    def test_core_actions(self):
        from rl_server.core.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector, ProbabilityActionSelector
        assert ArgmaxActionSelector is not None

    def test_core_buffers(self):
        from rl_server.core.buffers import Experience, ExperienceBuffer, TrajectoryBuffer
        assert Experience is not None

    def test_core_noisy(self):
        from rl_server.core.noisy import NoisyLinear, GradCoef
        assert NoisyLinear is not None

    def test_transport_serialization(self):
        from rl_server.transport.serialization import serialize, deserialize
        data = {'key': 'value', 'num': 42}
        serialized = serialize(data)
        result = deserialize(serialized)
        assert result == data

    def test_config_loader(self):
        from rl_server.config.loader import load_config, interpolate_env_vars
        assert load_config is not None

    def test_config_schema(self):
        from rl_server.config.schema import validate_config
        config = {
            'redis': {'model': {}, 'exps': {}},
            'training': {'env_name': 'DQN', 'num_samplers': 1, 'num_trainers': 1}
        }
        validate_config(config)  # Should not raise

    def test_config_schema_invalid(self):
        from rl_server.config.schema import validate_config
        with pytest.raises(ValueError):
            validate_config({})

    def test_utils_process(self):
        from rl_server.utils.process import setup_seed, should_exit, write_heartbeat, cleanup_heartbeat
        assert setup_seed is not None

    def test_utils_checkpoint(self):
        from rl_server.utils.checkpoint import save_model, load_model
        assert save_model is not None

    def test_utils_logging(self):
        from rl_server.utils.logging import Log, setup_logging
        assert Log is not None

    def test_algorithms_registry(self):
        from rl_server.algorithms import create_net, create_agent, create_calculate
        net = create_net('DQNGymClassic')
        assert net is not None
        params = list(net.parameters())
        assert len(params) > 0

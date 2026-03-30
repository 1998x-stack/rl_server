import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Patch numpy for gym 0.26 compatibility with numpy 2.x
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

from algo_envs.dqn_gym_classic import DQNGymClassicNet, DQNGymClassicAgent, DQNGymClassicCalculate
import libs.exps as Exps


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestLocalDQNPipeline:
    """Test full local pipeline: create net -> sample -> compute grads -> update."""

    def test_sample_train_cycle(self):
        net = DQNGymClassicNet()
        agent = DQNGymClassicAgent(net, is_checker=False)
        model_dict = {'TRAIN_VERSION': 0}

        # Sample experiences from multiple environments
        exps_list = agent.sample_multi_envs(model_dict)
        assert len(exps_list) > 0
        assert len(exps_list[0]) > 0

        # Each sample is [state, action, reward, done, train_version]
        first_sample = exps_list[0][0]
        assert len(first_sample) == 5

        # Create calculate module and fill its experience buffer
        calc = DQNGymClassicCalculate(net)
        all_samples = []
        for env_exps in exps_list:
            all_samples.extend(env_exps)

        # Need enough samples for the batch size
        assert len(all_samples) >= calc.batch_size, \
            f"Not enough samples ({len(all_samples)}) for batch_size ({calc.batch_size})"

        # generate_grads fills the buffer internally from samples
        grads_list, version = calc.generate_grads(all_samples, model_dict)
        assert len(grads_list) == 1
        assert version == 0

        # Apply gradients - parameters should change
        params_before = [p.data.clone() for p in net.parameters()]
        net.update_state(1, grads_list[0])
        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestCheckpointRoundtrip:
    def test_checkpoint_roundtrip(self, tmp_path):
        net = DQNGymClassicNet()
        states = torch.randn(2, 4)
        output_before = net(states).detach().clone()

        checkpoint = {'state_dict': net.state_dict(), 'version': 1}
        path = tmp_path / "checkpoint.td"
        torch.save(checkpoint, path)

        fresh_net = DQNGymClassicNet()
        loaded = torch.load(path)
        fresh_net.load_state_dict(loaded['state_dict'])
        output_after = fresh_net(states).detach().clone()

        assert torch.allclose(output_before, output_after)


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestDQNChecker:
    def test_check_single_env(self):
        """Test the checker agent runs a full episode."""
        net = DQNGymClassicNet()
        agent = DQNGymClassicAgent(net, is_checker=True)
        result = agent.check_single_env()
        assert 'sum_rewards' in result
        assert 'average_mus' in result
        assert isinstance(result['sum_rewards'], (int, float, np.floating))

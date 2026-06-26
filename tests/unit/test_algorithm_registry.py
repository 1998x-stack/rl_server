# -*- coding: utf-8 -*-
"""Tests for rl_server.algorithms registry and factory functions."""
import pytest
import torch.nn as nn

from rl_server.algorithms import create_net, create_agent, create_calculate, register, _REGISTRY
from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate


class TestAlgorithmRegistry:

    def test_create_net_dqn(self):
        net = create_net('DQNGymClassic')
        assert isinstance(net, AlgoBaseNet)
        assert isinstance(net, nn.Module)

    def test_create_agent_dqn(self):
        net = create_net('DQNGymClassic')
        agent = create_agent('DQNGymClassic', net)
        assert isinstance(agent, AlgoBaseAgent)

    def test_create_calculate_dqn(self):
        net = create_net('DQNGymClassic')
        calc = create_calculate('DQNGymClassic', net)
        assert isinstance(calc, AlgoBaseCalculate)

    def test_create_net_ppo_normal(self):
        net = create_net('MujocoNormal')
        assert isinstance(net, AlgoBaseNet)

    def test_create_net_ppo_beta(self):
        net = create_net('MujocoBeta')
        assert isinstance(net, AlgoBaseNet)

    def test_create_net_sac(self):
        net = create_net('SACMujocoNormal')
        assert isinstance(net, AlgoBaseNet)

    def test_create_net_td3(self):
        net = create_net('TD3MujocoNormal')
        assert isinstance(net, AlgoBaseNet)

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_net('NonExistentAlgorithm')

    def test_register_custom(self):
        # Register a dummy algorithm
        class DummyNet(AlgoBaseNet):
            def __init__(self):
                super().__init__()
            def forward(self, states):
                return states
            def update_state(self, version, grads_buffer):
                pass

        class DummyAgent(AlgoBaseAgent):
            def __init__(self, net, is_checker):
                super().__init__()

        class DummyCalc(AlgoBaseCalculate):
            def __init__(self, net):
                super().__init__()

        register('TestDummy', DummyNet, DummyAgent, DummyCalc)
        net = create_net('TestDummy')
        assert isinstance(net, DummyNet)
        # cleanup
        del _REGISTRY['TestDummy']

    def test_lazy_load_idempotent(self):
        net1 = create_net('DQNGymClassic')
        net2 = create_net('DQNGymClassic')
        assert isinstance(net1, type(net2))

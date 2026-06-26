"""DQN 网络、智能体与计算器单元测试。"""
import torch
import numpy as np

from rl_server.algorithms.dqn.network import DQNGymClassicNet


class TestDQNGymClassicNet:
    def test_forward_returns_actions(self):
        """forward() should return action indices (argmax of Q-values)."""
        net = DQNGymClassicNet()
        states = torch.randn(3, 4)  # batch of 3, obs_dim=4
        actions = net(states)
        assert actions.shape == (3,)
        # Actions should be valid indices (0 or 1 for CartPole)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 2)

    def test_get_q_values_shape(self):
        """get_q_values() should return Q-values for all actions."""
        net = DQNGymClassicNet()
        states = torch.randn(5, 4)  # batch of 5, obs_dim=4
        q_values = net.get_q_values(states)
        assert q_values.shape == (5, 2)  # act_dim=2 for CartPole

    def test_get_q_values_differentiable(self):
        """Q-values should support gradient computation."""
        net = DQNGymClassicNet()
        states = torch.randn(2, 4, requires_grad=True)
        q_values = net.get_q_values(states)
        loss = q_values.sum()
        loss.backward()
        assert states.grad is not None

    def test_forward_consistent_with_q_values(self):
        """forward() should return argmax of get_q_values()."""
        net = DQNGymClassicNet()
        states = torch.randn(4, 4)
        actions = net(states)
        q_values = net.get_q_values(states)
        expected_actions = torch.argmax(q_values, dim=-1)
        assert torch.allclose(actions, expected_actions)

    def test_update_state_modifies_parameters(self):
        """update_state() should change network parameters."""
        net = DQNGymClassicNet()
        params_before = [p.data.clone() for p in net.parameters()]

        # Create fake gradients matching parameter shapes
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)

        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed

    def test_single_input(self):
        """Should handle single observation (batch_size=1)."""
        net = DQNGymClassicNet()
        state = torch.randn(1, 4)
        action = net(state)
        assert action.shape == (1,)

    def test_network_has_parameters(self):
        """Network should have trainable parameters."""
        net = DQNGymClassicNet()
        params = list(net.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

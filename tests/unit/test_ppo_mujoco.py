"""MuJoCo PPO 网络与计算器单元测试。"""
import torch
import numpy as np

from rl_server.algorithms.ppo.mujoco_normal import MujocoNormalNet, TRAIN_ENVS, current_env_name


# Get dimensions from current config
OBS_DIM = TRAIN_ENVS[current_env_name].OBS_DIM
ACT_DIM = TRAIN_ENVS[current_env_name].ACT_DIM


class TestMujocoNormalNet:
    def test_forward_shape(self):
        """forward() returns mu values with shape (batch, act_dim)."""
        net = MujocoNormalNet()
        states = torch.randn(3, OBS_DIM)
        mus = net(states)
        assert mus.shape == (3, ACT_DIM)

    def test_forward_output_bounded(self):
        """Output of forward() should be bounded by Tanh in [-1, 1]."""
        net = MujocoNormalNet()
        states = torch.randn(10, OBS_DIM)
        mus = net(states)
        assert torch.all(mus >= -1.0)
        assert torch.all(mus <= 1.0)

    def test_get_distributions(self):
        """get_distributions() returns Normal distributions."""
        net = MujocoNormalNet()
        states = torch.randn(3, OBS_DIM)
        dists = net.get_distributions(states)
        # Should be able to sample from distributions
        samples = dists.sample()
        assert samples.shape == (3, ACT_DIM)

    def test_get_sample_data(self):
        """get_sample_data() returns (actions, log_probs)."""
        net = MujocoNormalNet()
        states = torch.randn(4, OBS_DIM)
        actions, log_probs = net.get_sample_data(states)
        assert actions.shape == (4, ACT_DIM)
        assert log_probs.shape == (4, ACT_DIM)

    def test_get_calculate_data(self):
        """get_calculate_data(states, actions) returns (values, log_probs, entropy)."""
        net = MujocoNormalNet()
        states = torch.randn(4, OBS_DIM)
        actions = torch.randn(4, ACT_DIM)
        values, log_probs, entropy = net.get_calculate_data(states, actions)
        assert values.shape == (4, 1)
        assert log_probs.shape == (4, ACT_DIM)
        assert entropy.shape == (4, ACT_DIM)

    def test_update_state_modifies_parameters(self):
        """update_state() should change network parameters."""
        net = MujocoNormalNet()
        params_before = [p.data.clone() for p in net.parameters()]

        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)

        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed

    def test_gradient_flow(self):
        """Gradients should flow through the network."""
        net = MujocoNormalNet()
        states = torch.randn(2, OBS_DIM, requires_grad=True)
        actions = torch.randn(2, ACT_DIM)
        values, log_probs, entropy = net.get_calculate_data(states, actions)
        loss = values.sum() + log_probs.sum() + entropy.sum()
        loss.backward()
        assert states.grad is not None

    def test_log_std_is_parameter(self):
        """log_std should be a learnable parameter."""
        net = MujocoNormalNet()
        assert hasattr(net, 'log_std')
        assert isinstance(net.log_std, torch.nn.Parameter)
        assert net.log_std.shape == (ACT_DIM,)

    def test_value_head_shape(self):
        """Value head should output scalar per observation."""
        net = MujocoNormalNet()
        states = torch.randn(5, OBS_DIM)
        values, _, _ = net.get_calculate_data(states, torch.randn(5, ACT_DIM))
        assert values.shape == (5, 1)

# -*- coding: utf-8 -*-
"""Smoke tests for all algorithm network forward passes and update_state."""
import torch
import numpy as np
import pytest


class TestDQNSmoke:
    """Smoke tests for DQN Gym Classic network."""

    def _make_net(self):
        from rl_server.algorithms.dqn.network import DQNGymClassicNet
        return DQNGymClassicNet()

    def test_instantiate(self):
        net = self._make_net()
        assert net is not None
        assert len(list(net.parameters())) > 0

    def test_forward_shape(self):
        net = self._make_net()
        states = torch.randn(4, 4)  # CartPole: obs_dim=4
        actions = net(states)
        assert actions.shape == (4,)

    def test_get_q_values_shape(self):
        net = self._make_net()
        states = torch.randn(4, 4)
        q_vals = net.get_q_values(states)
        assert q_vals.shape == (4, 2)  # CartPole: act_dim=2

    def test_update_state_changes_params(self):
        net = self._make_net()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, list(p.data for p in net.parameters())))
        assert changed


class TestPPONormalSmoke:
    """Smoke tests for PPO MuJoCo Normal network (Pusher: obs=23, act=7)."""

    def _make_net(self):
        from rl_server.algorithms.ppo.mujoco_normal import MujocoNormalNet
        return MujocoNormalNet()

    def _obs_dim(self):
        from rl_server.algorithms.ppo.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].OBS_DIM

    def _act_dim(self):
        from rl_server.algorithms.ppo.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].ACT_DIM

    def test_instantiate(self):
        net = self._make_net()
        assert len(list(net.parameters())) > 0

    def test_forward_shape(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        out = net(states)
        assert out.shape == (3, self._act_dim())

    def test_get_sample_data(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions, log_probs = net.get_sample_data(states)
        assert actions.shape == (3, self._act_dim())
        assert log_probs.shape == (3, self._act_dim())

    def test_get_check_data(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        mus, entropy = net.get_check_data(states)
        assert mus.shape == (3, self._act_dim())
        assert entropy.shape == (3, self._act_dim())

    def test_get_calculate_data(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions = torch.randn(3, self._act_dim())
        values, log_probs, entropy = net.get_calculate_data(states, actions)
        assert values.shape == (3, 1)
        assert log_probs.shape == (3, self._act_dim())

    def test_update_state(self):
        net = self._make_net()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, list(p.data for p in net.parameters())))
        assert changed


class TestPPOBetaSmoke:
    """Smoke tests for PPO MuJoCo Beta network (Swimmer: obs=8, act=2)."""

    def _make_net(self):
        from rl_server.algorithms.ppo.mujoco_beta import MujocoBetaNet
        return MujocoBetaNet()

    def _obs_dim(self):
        from rl_server.algorithms.ppo.mujoco_beta import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].OBS_DIM

    def _act_dim(self):
        from rl_server.algorithms.ppo.mujoco_beta import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].ACT_DIM

    def test_instantiate(self):
        net = self._make_net()
        assert len(list(net.parameters())) > 0

    def test_forward_shape(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        out = net(states)
        assert out.shape == (3, self._act_dim())

    def test_get_sample_data_returns_three(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        sample_actions, actions, log_probs = net.get_sample_data(states)
        assert sample_actions.shape == (3, self._act_dim())
        assert actions.shape == (3, self._act_dim())
        assert log_probs.shape == (3, self._act_dim())

    def test_update_state(self):
        net = self._make_net()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, list(p.data for p in net.parameters())))
        assert changed


class TestSACSmoke:
    """Smoke tests for SAC MuJoCo Normal network (Swimmer: obs=8, act=2)."""

    def _make_net(self):
        from rl_server.algorithms.sac.mujoco_normal import MujocoNormalQNet
        return MujocoNormalQNet()

    def _obs_dim(self):
        from rl_server.algorithms.sac.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].OBS_DIM

    def _act_dim(self):
        from rl_server.algorithms.sac.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].ACT_DIM

    def test_instantiate(self):
        net = self._make_net()
        assert len(list(net.parameters())) > 0

    def test_forward_shape(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        out = net(states)
        assert out.shape == (3, self._act_dim())
        # SAC forward applies tanh
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_get_sample_data(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions = net.get_sample_data(states)
        assert actions.shape == (3, self._act_dim())

    def test_get_q_values(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions = torch.randn(3, self._act_dim())
        q1, q2 = net.get_q_values(states, actions)
        assert q1.shape == (3, 1)
        assert q2.shape == (3, 1)

    def test_get_mu_q_values(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        mu_q = net.get_mu_q_values(states)
        assert mu_q.shape == (3, 1)

    def test_update_state(self):
        net = self._make_net()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, list(p.data for p in net.parameters())))
        assert changed


class TestTD3Smoke:
    """Smoke tests for TD3 MuJoCo Normal network (Swimmer: obs=8, act=2)."""

    def _make_net(self):
        from rl_server.algorithms.td3.mujoco_normal import MujocoNormalQNet
        return MujocoNormalQNet()

    def _obs_dim(self):
        from rl_server.algorithms.td3.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].OBS_DIM

    def _act_dim(self):
        from rl_server.algorithms.td3.mujoco_normal import TRAIN_ENVS, current_env_name
        return TRAIN_ENVS[current_env_name].ACT_DIM

    def test_instantiate(self):
        net = self._make_net()
        assert len(list(net.parameters())) > 0

    def test_forward_shape(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        out = net(states)
        assert out.shape == (3, self._act_dim())

    def test_get_sample_data(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions, log_probs = net.get_sample_data(states)
        assert actions.shape == (3, self._act_dim())
        assert log_probs.shape == (3, self._act_dim())

    def test_get_q_values(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        actions = torch.randn(3, self._act_dim())
        next_states = torch.randn(3, self._obs_dim())
        q_vals, ref_q_vals = net.get_q_values(states, actions, next_states)
        assert q_vals.shape == (3, 1)
        assert ref_q_vals.shape == (3, 1)

    def test_get_mu_q_values(self):
        net = self._make_net()
        states = torch.randn(3, self._obs_dim())
        mu_q = net.get_mu_q_values(states)
        assert mu_q.shape == (3, 1)

    def test_update_state(self):
        net = self._make_net()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, list(p.data for p in net.parameters())))
        assert changed

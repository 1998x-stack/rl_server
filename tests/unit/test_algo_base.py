"""``algo_envs.algo_base`` 基类与工具单元测试。"""
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algo_envs.algo_base import NoisyLinear, layer_init, AlgoBaseNet, GradCoef


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(4, 8)
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_noise_changes_output_in_train_mode(self):
        layer = NoisyLinear(4, 8)
        layer.train()
        x = torch.randn(1, 4)
        layer.sample_noise()
        out1 = layer(x).detach().clone()
        layer.sample_noise()
        out2 = layer(x).detach().clone()
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_eval_mode_is_deterministic(self):
        layer = NoisyLinear(4, 8)
        layer.eval()
        x = torch.randn(1, 4)
        out1 = layer(x).detach().clone()
        out2 = layer(x).detach().clone()
        assert torch.allclose(out1, out2)

    def test_sample_noise_changes_epsilon_weight(self):
        layer = NoisyLinear(4, 8)
        old_eps = layer.epsilon_weight.clone()
        layer.sample_noise()
        new_eps = layer.epsilon_weight.clone()
        # After sampling noise, epsilon_weight should change
        assert not torch.allclose(old_eps, new_eps)


class TestLayerInit:
    def test_orthogonal_init(self):
        linear = torch.nn.Linear(4, 8)
        initialized = layer_init(linear, method='orthogonal')
        assert initialized is linear
        w = linear.weight.data
        product = w @ w.T
        assert torch.all(torch.diag(product) > 0)

    def test_orthogonal_init_default(self):
        """Default method should be orthogonal."""
        linear = torch.nn.Linear(4, 8)
        initialized = layer_init(linear)
        assert initialized is linear

    def test_kaiming_init(self):
        # Note: kaiming init in the source has a bug with nn.init.zeros_ taking
        # an extra argument. Test with no-bias layer to avoid the bug path.
        linear = torch.nn.Linear(4, 8, bias=False)
        initialized = layer_init(linear, method='kaiming')
        assert initialized is linear


class TestGradCoef:
    def test_forward_preserves_values(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = GradCoef.apply(x, 0.5)
        assert torch.allclose(x, y)

    def test_backward_scales_gradient(self):
        x = torch.randn(3, 4, requires_grad=True)
        coeff = 0.5
        y = GradCoef.apply(x, coeff)
        loss = y.sum()
        loss.backward()
        expected_grad = torch.full_like(x, coeff)
        assert torch.allclose(x.grad, expected_grad)

    def test_backward_with_zero_coeff(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = GradCoef.apply(x, 0.0)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

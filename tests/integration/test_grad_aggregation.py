import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestGradAggregation:
    def test_accumulate_and_apply_grads(self):
        from rl_server.algorithms.dqn.network import DQNGymClassicNet
        net = DQNGymClassicNet()
        params_before = [p.data.clone() for p in net.parameters()]

        # Generate two sets of gradients
        grads1 = [np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in net.parameters()]
        grads2 = [np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in net.parameters()]

        # Accumulate
        accumulated = [g1 + g2 for g1, g2 in zip(grads1, grads2)]

        # Apply
        net.update_state(1, accumulated)
        params_after = [p.data.clone() for p in net.parameters()]

        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed

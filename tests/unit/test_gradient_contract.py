"""Pure gradient contract tests: stale versions, shape and finite values."""
import numpy as np
import torch
from rl_server.workers.grads_aggregator import GradsAggregator


def aggregator():
    value = GradsAggregator.__new__(GradsAggregator)
    value.grads_net = torch.nn.Linear(4, 2)
    value.current_train_version = 10
    value.max_gradient_lag = 0
    return value


def valid_grads(value):
    return [np.ones(tuple(p.shape), dtype=np.float32) for p in value.grads_net.parameters()]


def test_gradient_contract_accepts_current_valid_weights():
    instance = aggregator()
    assert instance._accept_gradients(valid_grads(instance), 10)


def test_gradient_contract_rejects_stale_and_future_versions():
    instance = aggregator()
    gradients = valid_grads(instance)
    assert not instance._accept_gradients(gradients, 9)
    assert not instance._accept_gradients(gradients, 11)


def test_gradient_contract_rejects_missing_wrong_shaped_or_nan():
    instance = aggregator()
    gradients = valid_grads(instance)
    assert not instance._accept_gradients(gradients[:-1], 10)
    invalid = valid_grads(instance)
    invalid[0] = np.ones((7,), dtype=np.float32)
    assert not instance._accept_gradients(invalid, 10)
    invalid = valid_grads(instance)
    invalid[0][0, 0] = float('nan')
    assert not instance._accept_gradients(invalid, 10)

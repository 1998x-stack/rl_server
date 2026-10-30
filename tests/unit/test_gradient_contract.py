"""Pure gradient contract tests: stale versions, shape and finite values."""
import collections

import numpy as np
import torch
from unittest.mock import MagicMock, patch
from rl_server.workers.grads_aggregator import GradsAggregator
from rl_server.core.gradient_contract import accepts_gradient_batch


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


def test_shared_gradient_contract_rejects_stale_and_nonfinite_batches():
    instance = aggregator()
    params = list(instance.grads_net.parameters())
    gradients = valid_grads(instance)

    assert accepts_gradient_batch(gradients, 10, params, 10)
    assert not accepts_gradient_batch(gradients, 9, params, 10)
    gradients[0][0, 0] = float('nan')
    assert not accepts_gradient_batch(gradients, 10, params, 10)


def test_aggregator_retries_snapshot_publish_without_reapplying_gradients():
    instance = GradsAggregator.__new__(GradsAggregator)
    instance.grads_net = torch.nn.Linear(4, 2)
    instance.grads_net.update_state = MagicMock()
    instance.current_train_version = 0
    instance.max_gradient_lag = 0
    instance.batch_update_count = 1
    instance.rejected_gradients = 0
    instance.model_prefix = "test"
    instance.env_name = "DQNGymClassic"
    instance.log = MagicMock()
    instance.grads_redis = MagicMock()
    instance.grads_redis.pop_grads.side_effect = [
        (valid_grads(instance), 0, 0),
        (None, None, None),
    ]
    model_cache = MagicMock()
    model_cache.set_train_version_model.side_effect = [False, True]
    instance.model_redis = model_cache
    checks = iter([False, False, False, False, True])

    with patch('rl_server.workers.grads_aggregator.should_exit', side_effect=checks), \
         patch('rl_server.workers.grads_aggregator.save_model'):
        instance.run()

    assert instance.grads_net.update_state.call_count == 1
    assert model_cache.set_train_version_model.call_count == 2
    assert instance.current_train_version == 1


def test_rejection_reasons_identify_the_actual_defect():
    """Rejections used to be logged as one undifferentiated "stale or invalid" counter."""
    from rl_server.core.gradient_contract import (
        REASON_FUTURE,
        REASON_LENGTH_MISMATCH,
        REASON_NON_FINITE,
        REASON_SHAPE_MISMATCH,
        REASON_STALE,
        REASON_VERSION_TYPE,
        gradient_batch_rejection,
    )

    instance = aggregator()
    params = list(instance.grads_net.parameters())
    gradients = valid_grads(instance)

    assert gradient_batch_rejection(gradients, 10, params, 10, 0) is None
    assert gradient_batch_rejection(gradients, 9, params, 10, 0) == REASON_STALE
    assert gradient_batch_rejection(gradients, 9, params, 10, 1) is None, "lag 1 tolerates one in-flight batch"
    assert gradient_batch_rejection(gradients, 11, params, 10, 0) == REASON_FUTURE
    assert gradient_batch_rejection(gradients, "10", params, 10, 0) == REASON_VERSION_TYPE
    assert gradient_batch_rejection(gradients[:-1], 10, params, 10, 0) == REASON_LENGTH_MISMATCH
    wrong_shape = valid_grads(instance)
    wrong_shape[0] = np.ones((7,), dtype=np.float32)
    assert gradient_batch_rejection(wrong_shape, 10, params, 10, 0) == REASON_SHAPE_MISMATCH
    non_finite = valid_grads(instance)
    non_finite[0][0][0] = np.nan
    assert gradient_batch_rejection(non_finite, 10, params, 10, 0) == REASON_NON_FINITE


def test_aggregator_counts_rejections_per_reason():
    instance = aggregator()
    instance.batch_update_count = 1
    instance.rejected_gradients = 0
    instance.rejected_reasons = collections.Counter()
    instance.model_prefix = "test"
    instance.env_name = "DQNGymClassic"
    instance.log = MagicMock()
    instance.grads_net.update_state = MagicMock()
    instance.grads_redis = MagicMock()
    instance.grads_redis.pop_grads.side_effect = [
        (valid_grads(instance), 9, 9),   # stale
        (valid_grads(instance), 11, 9),  # future
        (None, None, None),
    ]
    instance.model_redis = MagicMock()
    checks = iter([False, False, False, False, True])

    with patch('rl_server.workers.grads_aggregator.should_exit', side_effect=checks), \
         patch('rl_server.workers.grads_aggregator.save_model'):
        instance.run()

    assert instance.rejected_gradients == 2
    assert instance.rejected_reasons["stale_version"] == 1
    assert instance.rejected_reasons["future_version"] == 1


def test_resolve_max_gradient_lag_auto_uses_batch_width():
    from rl_server.core.gradient_contract import (
        DEFAULT_LAG_CAP,
        is_auto_lag,
        resolve_max_gradient_lag,
    )

    assert is_auto_lag("auto") and is_auto_lag(" AUTO ")
    assert not is_auto_lag(3) and not is_auto_lag(None)
    # auto 在不知道批次规模前保守取 1
    assert resolve_max_gradient_lag("auto") == 1
    # 一批产出 N 条梯度时，需要 N-1 的滞后跨度才能吃满
    assert resolve_max_gradient_lag("auto", window=10) == 9
    assert resolve_max_gradient_lag("auto", window=1) == 1
    assert resolve_max_gradient_lag("auto", window=10 ** 6) == DEFAULT_LAG_CAP
    assert resolve_max_gradient_lag("auto", window=10, cap=4) == 4
    # 显式整数原样返回
    assert resolve_max_gradient_lag(0) == 0
    assert resolve_max_gradient_lag(7) == 7
    import pytest as _pytest
    with _pytest.raises(TypeError):
        resolve_max_gradient_lag("many")


def test_aggregator_auto_lag_scales_with_aggregation_window():
    instance = GradsAggregator.__new__(GradsAggregator)
    instance.grads_net = torch.nn.Linear(4, 2)
    instance.batch_update_count = 10
    from rl_server.core.gradient_contract import resolve_max_gradient_lag
    instance.max_gradient_lag = resolve_max_gradient_lag("auto", window=10)
    instance.current_train_version = 20
    grads = [np.ones(tuple(p.shape), dtype=np.float32) for p in instance.grads_net.parameters()]

    assert instance.max_gradient_lag == 9
    assert instance._accept_gradients(grads, 20)
    assert instance._accept_gradients(grads, 11)   # 窗口内的旧批次仍然可用
    assert not instance._accept_gradients(grads, 10)

"""SAC calculator contracts: full parameter coverage, target critics, learnable temperature."""
import numpy as np
import torch

from rl_server.algorithms.sac.mujoco_normal import (
    TRAIN_CONFIG,
    TRAIN_ENVS,
    MujocoNormalQCalculate,
    MujocoNormalQNet,
    current_env_name,
)
from rl_server.core.optim import trainable_parameters


def _samples(count, obs_dim):
    rng = np.random.default_rng(0)
    return [
        [rng.normal(size=obs_dim).astype(np.float32), rng.uniform(-1, 1, size=2).astype(np.float32),
         0.1, False, np.zeros(2, dtype=np.float32), 0.0]
        for _ in range(count)
    ]


def _calculator():
    net = MujocoNormalQNet()
    return net, MujocoNormalQCalculate(net)


def _batch_count():
    return max(TRAIN_CONFIG['BATCH_SIZE'] + 5, 300)


def test_every_trainable_parameter_gets_a_finite_gradient():
    obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
    net, calc = _calculator()

    grads_list, version = calc.generate_grads(_samples(_batch_count(), obs_dim),
                                              {'TRAIN_VERSION': 7})

    assert version == 7
    grads = grads_list[0]
    # 目标 critic 是 requires_grad=False 的参数，不能进入聚合（否则长度校验会整批拒绝）。
    assert len(grads) == len(trainable_parameters(calc.calculate_net))
    assert all(g is not None for g in grads), "actor loss must cover every trainable parameter"
    assert all(np.isfinite(g).all() for g in grads)


def test_target_critics_are_not_aggregated_and_are_synced_with_online():
    net = MujocoNormalQNet()
    trainable = trainable_parameters(net)
    target_params = set(id(p) for p in net.target_q1_value.parameters())
    target_params |= set(id(p) for p in net.target_q2_value.parameters())

    assert target_params, "target critics must exist on the net"
    assert not target_params & set(id(p) for p in trainable)

    net.sync_targets()
    for target, online in ((net.target_q1_value, net.q1_value), (net.target_q2_value, net.q2_value)):
        for t_param, o_param in zip(target.parameters(), online.parameters()):
            assert torch.equal(t_param, o_param), "sync_targets must copy online weights"


def test_update_state_soft_updates_targets_and_keeps_optimizer_state():
    from rl_server.core.optim import get_cached_adam

    net = MujocoNormalQNet()
    net.sync_targets()
    params = trainable_parameters(net)
    grads = [np.asarray(np.random.randn(*p.shape), dtype=np.float32) * 0.01 for p in params]

    before = [p.detach().clone() for p in net.target_q1_value.parameters()]
    net.update_state(1, grads)
    after = [p.detach().clone() for p in net.target_q1_value.parameters()]

    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "polyak update must move targets"
    tau = TRAIN_CONFIG['TAU']
    online = [p.detach() for p in net.q1_value.parameters()]
    for b, a, o in zip(before, after, online):
        expected = (1 - tau) * b + tau * o
        assert torch.allclose(a, expected, atol=1e-5)

    # 优化器必须被复用，否则 Adam 的矩估计每次都被清零。
    assert get_cached_adam(net, 'policy', params[:1], 1e-3) is get_cached_adam(net, 'policy', params[:1], 1e-3)
    assert get_cached_adam(net, 'policy', params[:1], 1e-2) is not get_cached_adam(net, 'policy', params[:1], 1e-3)


def test_log_alpha_is_trainable_and_driven_by_the_entropy_target():
    obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
    net, calc = _calculator()
    assert any(p is net.log_alpha for p in trainable_parameters(net))
    assert torch.isclose(net.alpha, torch.tensor(TRAIN_CONFIG['ALPHA']))

    grads_list, _ = calc.generate_grads(_samples(_batch_count(), obs_dim), {'TRAIN_VERSION': 0})
    names = [name for name, _ in calc.calculate_net.named_parameters()]
    log_alpha_grad = grads_list[0][names.index('log_alpha')]
    assert log_alpha_grad.shape == tuple(calc.calculate_net.log_alpha.shape)
    assert np.isfinite(log_alpha_grad).all()
    assert abs(float(log_alpha_grad)) > 0

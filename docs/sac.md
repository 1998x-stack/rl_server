---
layout: default
title: SAC Algorithm
description: Soft Actor-Critic — dual Q-networks with entropy regularization
---

# SAC — Soft Actor-Critic

**Registry key:** `SACMujocoNormal`

SAC combines off-policy Q-learning with maximum entropy reinforcement learning, using dual Q-networks to reduce overestimation bias.

## Files

```
rl_server/algorithms/sac/
  mujoco_normal.py  — MujocoNormalQNet, MujocoNormalQAgent, MujocoNormalQCalculate
```

(All three classes in a single file following the pattern of PPO's single-file architecture.)

---

## Network — `MujocoNormalQNet`

### Architecture

```
Input (obs_dim)
    ↓
┌────────────────────┬────────────────────┬──────────────────────┐
│  μ network          │  log_std network    │  Q1 network          │
│  MLP(hidden)        │  MLP(hidden)        │  Q2 network          │
│  [NoisyLinear]      │  [NoisyLinear]      │                      │
│  Linear(h, act_dim) │  Linear(h, act_dim) │  Q1/Q2: Linear(     │
│                     │                     │   obs+act, hidden)   │
│                     │                     │  → ReLU → Linear(   │
│                     │                     │   hidden,1)          │
└──────────┬──────────┴──────────┬──────────┴──────────┬───────────┘
           │                     │                     │
           ▼                     ▼                     ▼
        μ(s)                 log_std(s)            Q1(s,a), Q2(s,a)
           │                     │
           └──────────┬──────────┘
                      ▼
          π(a|s) = tanh(Normal(μ, exp(log_std)))
```

### Key Methods

| Method | Returns | Used By |
|--------|---------|---------|
| `forward(states)` | `tanh(μ)` — deterministic policy | — |
| `get_sample_data(states)` | `tanh(sampled actions)` | Sampler |
| `get_check_data(states)` | `(tanh(μ), entropy)` | Checker |
| `get_calculate_data(states, actions)` | `(Q1(s,a), log_prob, entropy)` | Calculator |
| `get_q_values(states, actions)` | `(Q1(s,a), Q2(s,a))` — both Q networks | Calculator (Q loss) |
| `get_mu_q_values(states)` | μ-based Q value | — |
| `update_state(version, grads_buffer)` | None | Main process |

### Noise Layers

When `USE_NOISE=True`, separate `NoisyLinear` pairs for the μ network and log_std network:

```python
alpha_noisy_layers = [NoisyLinear(hidden, hidden), NoisyLinear(hidden, act_dim)]  # for μ
beta_noisy_layers  = [NoisyLinear(hidden, hidden), NoisyLinear(hidden, act_dim)]  # for log_std
```

---

## Agent — `MujocoNormalQAgent`

### Multi-Environment Sampling

```python
for step in range(NUM_STEPS):
    actions = self._get_sample_actions(states)   # tanh(sampled from Normal)
    for i in range(num_envs):
        exps[i].append([state, action, reward, done, version])
```

Experience format: `[state, action, reward, done, train_version]` — same structure as DQN.

### Action Selection

```python
@torch.no_grad()
def _get_sample_actions(self, states):
    actions = self.sample_net.get_sample_data(states_v)
    return actions.cpu().numpy()
    # get_sample_data returns tanh(sampled_action)
```

### Checker

Same pattern as MuJoCo PPO: `(tanh(μ), entropy)` per step, returns `sum_rewards`, `sum_entropys`, `average_mus`.

---

## Calculator — `MujocoNormalQCalculate`

### Training Components

| Component | Detail |
|-----------|--------|
| **Replay buffer** | `ExperienceBuffer(capacity=100000)` |
| **Batch size** | 256 |
| **Repeat time** | 1 (one minibatch per `generate_grads` call) |
| **γ** | 0.99 |
| **Learning rate** | 2.5e-4 |
| **τ (polyak)** | 0.005 — 目标 critic 软更新 |
| **初始 α** | 0.2（之后由 `log_alpha` 学习） |
| **目标熵** | `-act_dim` |

### Gradient Computation Flow

```
generate_grads(samples, model_dict):
  1. Unpack samples → state/action/reward/done arrays
  2. Create Experience tuples (consecutive pairs)
  3. Append to ExperienceBuffer
  4. If buffer < batch_size: raise ValueError
  5. Sync calculate_net ← share_model.state_dict()
  6. Move calculate_net to device (GPU if configured)
  7. Repeat REPEAT_TIME (default 1):
     a. Sample batch from buffer
     a2. critic：Q1(s,a), Q2(s,a) from calculate_net
        - next_actions = calculate_net.get_sample_data(next_states)
        - next_q1, next_q2 = *target* critics (polyak 副本)
        - next_q = min(next_q1, next_q2)                  ← clipped double Q
        - target = r + γ * (1 - done) * next_q            ← 终局不自举
        - Q_loss = MSE(Q1, target) + MSE(Q2, target) → backward
     b. actor（冻结 critic 叶子参数）：
        - dists = calculate_net.get_distributions(states)
        - raw = dists.rsample(); a = tanh(raw)
        - logπ = dists.log_prob(raw) - log(1 - a²)         ← tanh 雅可比修正
        - actor_loss = (α·logπ - min(Q1(s,a), Q2(s,a))).mean() → backward
     c. temperature：alpha_loss = -(log_alpha·(logπ + target_entropy)).mean() → backward
     d. 软更新目标 critic：θ_target ← (1-τ)·θ_target + τ·θ_online
  8. Collect parameter.grad as numpy arrays（含 log_alpha）
  9. 写 trainer_metrics：q_loss / actor_loss / alpha_loss / entropy / alpha
 10. Return [grads], train_version
```

### SAC-Specific Features

| Feature | Implementation |
|---------|----------------|
| **Dual Q-networks** | Two separate MLPs (`q1_value`, `q2_value`) taking `concat(state, action)` |
| **Clipped double Q** | Target uses `min(Q1, Q2)` — reduces overestimation |
| **Entropy regularization** | Policy is stochastic by default (Gaussian with learnable log_std) |
| **Auto-temperature α** | `log_alpha` 是可学习参数（初值 0.2），目标熵 `-act_dim`；随梯度一起聚合 |
| **Experience replay** | Off-policy training via ExperienceBuffer |
| **Noise resampling** | Separate noisy layers for μ and log_std networks |

### Soft Update

目标 critic 是 `MujocoNormalQCalculate` 内部的独立副本（首次调用时从在线 critic 初始化），
每次 `generate_grads` 结束按 `τ=0.005` 软更新：

```python
θ_target ← (1 - τ) · θ_target + τ · θ_online
```

它们只存在于 trainer 进程，不进 checkpoint、也不参与梯度聚合（聚合的是在线网络参数）。
因此重启训练会重新用当前权重初始化目标网络——这会带来一次轻微的 target 收缩，属可接受行为。

### Supported Environments

Same MuJoCo suite as PPO Normal:

Swimmer, HalfCheetah, Ant, Hopper, Pusher, Humanoid, Walker2d, HumanoidStandup, Reacher, and others.

| Env | obs_dim | act_dim | hidden_dim |
|-----|---------|---------|------------|
| Swimmer | 8 | 2 | 32 |
| HalfCheetah | 17 | 6 | 64 |
| Ant | 27 | 8 | 256 |
| Humanoid | 376 | 17 | 512 |

### 吞吐与梯度滞后（调参建议）

SAC 的 trainer 每批样本会产出多条梯度，而主循环每应用一条就把 `TRAIN_VERSION` 推高一格。
`max_gradient_lag: auto`（默认）会按实测的到达滞后自适应，把正常在途的批次也接受下来；
仍然出现高拒绝率时，说明是**吞吐配比**问题而不是容错问题：

```
grads queue backlog depth=400/1000 (40%): trainers produce faster than the main loop applies;
consider raising training.num_trainers (now 1) or lowering num_update_grads
```

| 现象 | 处理 |
|---|---|
| `queue_grads` 持续上涨 + 拒绝率 > 30% | 增加 `training.num_trainers`，或调大 `num_update_grads`（多条梯度累加为一次更新） |
| `arrival_lag_p95` 持续大于 `max_gradient_lag_cap` | 说明消费端跟不上，同上；单纯调大 cap 只会让梯度更陈旧 |
| 版本速率远低于采样批次速率 | 检查是否只配了 1 个 trainer（默认 1） |

### Current Limitations

- **Actor/critic 同一步更新**：两者梯度在同一次 `update_state` 里按同一学习率生效
  （多数实现用两个 optimizer 分别更新），对稳定性有轻微影响。
- **目标 critic 不跨进程持久化**：重启后从在线权重重新初始化（见 Soft Update）。
- **`log_alpha` 与策略共用学习率**（2.5e-4），收敛比专用 α 学习率慢。
- **Single gradient pass per call**: `REPEAT_TIME=1`

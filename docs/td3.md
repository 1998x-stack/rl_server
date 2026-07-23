---
layout: default
title: TD3 Algorithm
description: Twin Delayed DDPG — clipped double Q-learning with target policy smoothing
---

# TD3 — Twin Delayed DDPG

**Registry key:** `TD3MujocoNormal`

TD3 adapts the PPO Normal algorithm's structure with additional Q-learning and deterministic policy components, implementing the key TD3 features: clipped double Q, delayed policy updates, and target policy smoothing.

## Files

```
rl_server/algorithms/td3/
  mujoco_normal.py  — MujocoNormalQNet (TD3 variant), MujocoNormalQAgent, MujocoNormalQCalculate
```

---

## Network — `MujocoNormalQNet` (TD3)

### Architecture

```
Input (obs_dim)
    ↓
┌────────────────────┬────────────────────┬──────────────────────┐
│  μ network          │  Q network          │  Value network       │
│  (deterministic     │  Q1(s,a):           │  V(s):               │
│   policy)           │  Linear(obs+act,    │  Linear(obs, hidden) │
│  MLP(hidden)        │    hidden) + ReLU   │  → ReLU →            │
│  Tanh → [-1,1]      │  Linear(hidden, 1)  │  Linear(hidden, 1)   │
│                     │                      │                      │
│  [NoisyLinear]      │                      │                      │
└──────────┬──────────┴──────────┬───────────┴──────────┬──────────┘
           │                     │                      │
           ▼                     ▼                      ▼
        μ(s)                 Q(s,a)                   V(s)
```

### Key Difference from PPO

The TD3 network includes a **Q network** (`q_value`) that takes `concat(state, action)` as input, and a **separate value network** (`value`) that takes only the state. This is in addition to the deterministic policy (`mu`).

### Key Methods

| Method | Returns | Used By |
|--------|---------|---------|
| `forward(states)` | μ(s) — deterministic | — |
| `get_sample_data(states)` | `(sampled_action, log_prob)` | Sampler |
| `get_check_data(states)` | `(μ(s), entropy)` | Checker |
| `get_calculate_data(states, actions)` | `(V(s), log_prob, entropy)` | Calculator |
| `get_q_values(states, actions, next_states)` | `(Q(s,a), Q_next)` | Calculator (Q loss) |
| `get_mu_q_values(states)` | `Q(s, μ(s) + noise)` | Calculator (policy Q loss) |

---

## Agent — `MujocoNormalQAgent`

Identical structure to PPO Normal and SAC agents. Multi-environment rollout with exploration noise:

```python
for step in range(NUM_STEPS):
    actions, log_probs = self._get_sample_actions(states)
    for i in range(num_envs):
        exps[i].append([state, action, reward, done, log_prob[i], version])
```

Experience format: `[state, action, reward, done, log_prob, train_version]` — same as PPO (log_prob stored for importance sampling).

---

## Calculator — `MujocoNormalQCalculate`

### Training Components

| Component | Value |
|-----------|-------|
| γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Max clip | 2.0 |
| Entropy coefficient | 0.001 |
| Value coefficient | 1.0 |
| Q coefficient | 1.0 |
| Q-μ coefficient | 0.003 |
| Learning rate | 2.5e-4 |
| Q repeat | 1 |

### Gradient Computation Flow

```
generate_grads(samples, model_dict):
  1. Unpack samples → state/action/reward/done/log_prob arrays
  2. Sync calculate_net ← share_model.state_dict()
  3. Move to device
  4. === PART 1: PPO-style policy + value gradients ===
     a. Get V(s), log_prob, entropy from get_calculate_data()
     b. Compute GAE advantages and returns
     c. Compute PPO clipped policy loss (all 4 variants via PPO_LOSS_TYPE)
     d. Compute MSE value loss
     e. Compute entropy loss
     f. pg_loss + v_loss + e_loss → backward()

  5. === PART 2: Policy Q gradient ===
     a. get_mu_q_values(states): Q(s, μ(s) + noise)
     b. q_mu_loss = -mean(Q_mu) * q_mu_coef / (exp(entropy) + 1)
     c. Temporarily disable Q network gradients
     d. q_mu_loss.backward()
     e. Re-enable Q network gradients

  6. === PART 3: Q learning ===
     a. get_q_values(states, actions, next_states) → Q(s,a), Q_next
     b. Q-return: r + γ * Q_next (or just r if done)
     c. MSE(Q(s,a), Q-return) * q_coef → backward()

  7. Collect all parameter.grad as numpy arrays
  8. Return [grads], train_version
```

### TD3-Specific Features

| Feature | Implementation |
|---------|----------------|
| **Clipped double Q** | Single Q network (`q_value`), not dual. Uses `get_q_values` → `(Q_current, Q_next)` |
| **Delayed policy updates** | Not explicitly implemented — policy and Q gradients computed every call |
| **Target policy smoothing** | Target action noise injected via `get_mu_q_values`: `μ(s) + exp(log_std) * noise` |
| **Deterministic policy** | `forward()` returns μ(s) without noise |
| **Q loss** | Separate Q-learning loss on top of PPO-style losses |

### Loss Composition

```python
# Total loss = PPO-style + Q-μ + Q-learning
loss = (pg_loss + v_loss + e_loss)     # PPO actor-critic
     + q_mu_loss                        # Policy Q guidance
     + q_loss                           # Q-learning
```

The Q-μ loss (`q_mu_coef=0.003`) guides the deterministic policy toward actions with high Q-values. The Q-learning loss (`q_coef=1.0`) trains the Q-network.

### Supported Environments

Same MuJoCo suite as PPO Normal and SAC.

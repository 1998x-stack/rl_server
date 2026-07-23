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
     b. Q1(s,a), Q2(s,a) from calculate_net
     c. Target Q (clipped double Q):
        - next_actions = calculate_net.get_sample_data(next_states)
        - next_q1, next_q2 = calculate_net.get_q_values(next_states, next_actions)
        - next_q = min(next_q1, next_q2)     ← clipped double Q
        - target = r + γ * next_q
     d. Q_loss = MSE(Q1, target) + MSE(Q2, target)
     e. Q_loss.backward()
  8. Collect parameter.grad as numpy arrays
  9. Return [grads], train_version
```

### SAC-Specific Features

| Feature | Implementation |
|---------|----------------|
| **Dual Q-networks** | Two separate MLPs (`q1_value`, `q2_value`) taking `concat(state, action)` |
| **Clipped double Q** | Target uses `min(Q1, Q2)` — reduces overestimation |
| **Entropy regularization** | Policy is stochastic by default (Gaussian with learnable log_std) |
| **Auto-temperature α** | Not yet implemented — uses fixed entropy coefficient |
| **Experience replay** | Off-policy training via ExperienceBuffer |
| **Noise resampling** | Separate noisy layers for μ and log_std networks |

### Soft Update

Unlike DQN which uses hard target network updates every N steps, SAC uses the same online network for both Q estimation and target — the target is produced by the same net (`calculate_net`) on `next_states`. There is no separate target network.

### Supported Environments

Same MuJoCo suite as PPO Normal:

Swimmer, HalfCheetah, Ant, Hopper, Pusher, Humanoid, Walker2d, HumanoidStandup, Reacher, and others.

| Env | obs_dim | act_dim | hidden_dim |
|-----|---------|---------|------------|
| Swimmer | 8 | 2 | 32 |
| HalfCheetah | 17 | 6 | 64 |
| Ant | 27 | 8 | 256 |
| Humanoid | 376 | 17 | 512 |

### Current Limitations

- **No separate target networks**: `update_state` does not soft-update target parameters
- **No learned temperature**: α is fixed, not tuned automatically
- **Single gradient pass per call**: `REPEAT_TIME=1`

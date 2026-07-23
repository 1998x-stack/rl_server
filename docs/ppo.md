---
layout: default
title: PPO Algorithm
description: Proximal Policy Optimization — three variants: Normal, Beta, MicroRTS
---

# PPO — Proximal Policy Optimization

**Registry keys:** `MujocoNormal`, `MujocoBeta`, `MicroRTS`

PPO in RL Server has three variants sharing the same GAE+clipped-surrogate core, differing in action distribution and network architecture.

## Files

```
rl_server/algorithms/ppo/
  mujoco_normal.py  — Gaussian policy, Tanh output, standard PPO
  mujoco_beta.py    — Beta distribution policy (bounded actions)
  microrts.py       — Masked categorical policy (discrete, action masking)
```

---

## Common PPO Core

All three variants share these training components:

| Component | Value |
|-----------|-------|
| GAE λ | 0.95 |
| γ | 0.99 |
| Clip ε | 0.2 |
| Max clip | 2.0 (4.0 for MicroRTS) |
| Entropy coefficient | 0.2 (0.01 for MicroRTS) |
| Value coefficient | 4.0 (1.0 for MicroRTS) |
| Learning rate | 2.5e-4 |
| NUM_ENVS | 32 (8 for MicroRTS) |
| NUM_STEPS | 1000 (512 for MicroRTS) |

### GAE Computation

```python
deltas = rewards + γ * next_values - values
deltas[dones] = rewards[dones] - values[dones]  # bootstrap from value on done

advantages = zeros(len(deltas) + 1)
gae = 0.0
for t in reversed(range(len(deltas))):
    gae = deltas[t] + γ * λ * (~dones[t]) * gae
    advantages[t] = gae

returns = advantages + values_full
```

### Clipped Surrogate Loss

```python
# PPO loss with asymmetric clipping (positive vs negative advantages)
def get_pg_loss(ratio, advantage):
    base = ratio * advantage
    clipped = clamp(ratio, 1-ε, 1+ε) * advantage
    min_loss = min(base, clipped)
    max_loss = max(min_loss, max_clip * advantage)
    return where(advantage >= 0, min_loss, max_loss)
```

The `max_clip` parameter (default 2.0) provides asymmetric clipping for negative advantages — preventing the policy from collapsing when an action was much worse than expected.

### PPO Loss Type Ablation

Controlled by `TRAIN_CONFIG['PPO_LOSS_TYPE']`:

| Mode | Ratio | Formula | Description |
|------|-------|---------|-------------|
| `discrete` | `ratio1` | `exp(new_log_prob - old_log_prob)` | Per-action-dimension ratio (default for discrete spaces) |
| `prod` | `ratio2` | `exp(sum(new) - sum(old))` | Joint product ratio across action dims |
| `mixed` | `ratio3` | `(ratio1 + ratio2) / 2` | Default — average of discrete and prod |
| `avg_discrete_prod` | `ratio4` | `(pg_loss1 + pg_loss2) / 2` | Average of the two loss values |

### Update State (Noise Resampling)

After each gradient update, the main process resamples noise for `NoisyLinear` layers (when `USE_NOISE=True`):

```python
def update_state(self, version, grads_buffer):
    train_optim.zero_grad()
    for param, grad in zip(self.parameters(), grads_buffer):
        param.grad = torch.FloatTensor(grad)
    train_optim.step()

    if USE_NOISE:
        self.noise_layer_out.sample_noise()
        self.noise_layer_hide.sample_noise()
```

---

## PPO Normal (`MujocoNormal`)

Gaussian policy with learnable log_std parameter.

### Network Architecture

```
Input (obs_dim)
    ↓
Linear(obs, hidden) + ReLU                 ← layer_init (orthogonal)
Linear(hidden, hidden) + ReLU
[NoisyLinear(hidden, hidden) + ReLU]        ← if USE_NOISE
[NoisyLinear(hidden, act_dim)]              ← if USE_NOISE, else Linear(hidden, act_dim)
Tanh                                        ← output squash to [-1, 1]
    ↓
┌────────────────────────────────────┐
│  μ (deterministic mean)            │
│  log_std (learnable parameter)     │
│  π(a|s) = Normal(μ, exp(log_std))  │
└────────────────────────────────────┘

Value head (separate):
    Linear(obs, hidden) + ReLU → Linear(hidden, hidden) + ReLU → Linear(hidden, 1)
```

### Key Methods

| Method | Returns | Used By |
|--------|---------|---------|
| `forward(states)` | μ (deterministic) | — |
| `get_sample_data(states)` | (actions, log_probs) | Sampler |
| `get_check_data(states)` | (μ, entropy) | Checker |
| `get_calculate_data(states, actions)` | (values, log_probs, entropy) | Calculator |

### Agent

```python
# Sample: stochastic with exploration noise (Gaussian)
actions, log_probs = net.get_sample_data(states)

# Check: deterministic mean
mu, entropy = net.get_check_data(state)
```

### Calculator

Full PPO loss:
```python
loss = pg_loss + vf_coef * MSE(returns, values) + ent_coef * mean(entropy)
loss.backward()
```

Writes training metrics to `model_dict['trainer_metrics']` for TensorBoard:
- `policy_loss`, `value_loss`, `entropy`, `gradient_norm`

### Supported Environments

| Env | obs_dim | act_dim | hidden_dim | USE_NOISE |
|-----|---------|---------|------------|-----------|
| Swimmer | 8 | 2 | 32 | ✅ |
| HalfCheetah | 17 | 6 | 64 | ✅ |
| Ant | 27 | 8 | 256 | ✅ |
| Hopper | 11 | 3 | 64 | ✅ |
| Pusher | 23 | 7 | 128 | ✅ |
| Humanoid | 376 | 17 | 512 | ✅ |
| Walker2d | 17 | 6 | 64 | ✅ |
| HumanoidStandup | 376 | 17 | 512 | ✅ |

---

## PPO Beta (`MujocoBeta`)

Beta distribution policy for naturally bounded actions [0, 1].

### Key Difference from Normal

- Output is **Beta distribution** (α, β parameters) instead of Gaussian (μ, σ)
- Actions are naturally bounded to (0, 1) — no Tanh squash needed
- Uses `GradCoef` for gradient scaling in the Beta path

### Network Architecture

```
Input → Shared MLP backbone
    ↓
┌────────────────────┬────────────────────┬──────────────┐
│  α head            │  β head            │  Value head   │
│  Linear(h, act_dim)│  Linear(h, act_dim)│  Linear(h, 1) │
│  Softplus + 1      │  Softplus + 1      │               │
└────────┬───────────┴────────┬───────────┴──────────────┘
         └──────────┬─────────┘
                    ▼
       π(a|s) = Beta(α, β)
       a ∈ (0, 1)
```

---

## PPO MicroRTS (`MicroRTS`)

Discrete action PPO with action masking for real-time strategy games via `gym_microrts 0.3.2`.

### Network Architecture

```
Input: 10×10×27 (H×W×C grid observation)
    ↓
Conv2D(27, 16, 3×3, stride=1, pad=1) + ReLU
Conv2D(16, 32, 3×3, stride=1, pad=1) + ReLU
Flatten (10×10×32 = 3200)
Linear(3200, 256) + ReLU
Linear(256, 256) + ReLU
    ↓
MaskedCategorical × 8 action heads

Value head:
Linear(3200, 256) + ReLU → Linear(256, 1)
```

### Action Space (Hierarchical)

MicroRTS uses a multi-discrete action space with 8 action types:

| Head | Size | Description |
|------|------|-------------|
| 0 | 100 | Unit type selection |
| 1 | 6 | Action type (attack, move, harvest, etc.) |
| 2 | 4 | Move/attack parameter |
| 3 | 4 | Parameter |
| 4 | 4 | Parameter |
| 5 | 4 | Parameter |
| 6 | 7 | Parameter |
| 7 | 49 | Target position (7×7 grid) |

### MaskedCategorical

Custom distribution that applies action masks by setting invalid action logits to `-inf`:

```python
class MaskedCategorical(Categorical):
    def _apply_action_masks(self, logits, probs):
        # Mask invalid actions
        clamped_logits = logits.masked_fill(~self.action_masks, float('-inf'))

        # NaN guard: if ALL actions masked, fall back to original logits
        clamped_logits = torch.nan_to_num(clamped_logits, nan=-1e9)
        all_masked = ~self.action_masks.any(dim=-1)
        if all_masked.any():
            masked[all_masked] = clamped_logits[all_masked]
```

### GPU Notes (from deployment guide)

| Component | Device | Rationale |
|-----------|--------|-----------|
| MicroRTSNet (main) | CPU | No share_memory(); synced via state_dict |
| MicroRTSNet (trainer) | GPU | `.to(device)` in calculator |
| update_state | CPU | `torch.FloatTensor(grad)` stays on CPU |
| generate_grads tensors | GPU | Explicit `.to(self.device)` required |

### Gym_microrts 0.3.2 Compatibility

This algorithm uses the **PyPI 0.3.2** API (`MicroRTSVecEnv`), NOT the git-HEAD `MicroRTSGridModeVecEnv`. Key API differences:

| API | 0.3.2 | git-HEAD |
|-----|-------|----------|
| Step | 4 values (no `truncated`) | 5 values |
| Reset | Returns numpy array directly | Returns tuple |
| Action | Hierarchical (unit → command) | Grid-based per cell |
| Compatibility | ✅ Supported | ❌ Incompatible |

### numpy 2.0 Shim

```python
import numpy
numpy.int = int
numpy.float = float
```

Applied at the module top level. A full source-code patch must also be applied to the installed `gym_microrts` package for spawned subprocesses (see [deployment docs](deployment.md)). 

---
layout: default
title: Core Module
description: Base classes, buffers, action selectors, and noisy layers
---

# Core Module

The `rl_server/core/` package provides foundational abstractions shared across all algorithms and deployment modes. Every algorithm's code imports from here.

## Package Layout

```
rl_server/core/
  base.py       — AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init
  buffers.py    — ExperienceBuffer, TrajectoryBuffer, Experience
  actions.py    — ActionSelector, ArgmaxActionSelector, EpsilonGreedyActionSelector,
                  ProbabilityActionSelector, EpsilonTracker
  noisy.py      — NoisyLinear (factorized Gaussian noise), GradCoef (gradient scaling)
```

---

## base.py — Algorithm Base Classes

### `layer_init(linear_layer, std=np.sqrt(2), bias_const=0.0, method='orthogonal')`

Weight initialization helper used by every network in the project. Two modes:

- **`orthogonal`** (default): `nn.init.orthogonal_(weight, std)`, bias set to `bias_const`. Matched to the `sqrt(2)` gain common in RL.
- **`kaiming`**: `nn.init.kaiming_normal_(weight, fan_in, relu)`, bias zeroed. Used by some custom networks.

Returns the same `nn.Linear` object (for chaining in `nn.Sequential`).

### `AlgoBaseNet(nn.Module)`

Abstract base for the policy/value network. Two abstract methods:

```python
def forward(self, states)          # Inference: states → actions
def update_state(self, version, grads_buffer)  # Apply aggregated gradients
```

Concrete subclasses implement:
- `forward()` — called by sampler and checker workers for action selection
- `update_state()` — called by main process (or grads aggregator) to apply averaged gradients using `torch.optim`

Additional methods per algorithm (non-standardized):
- `get_q_values()` — DQN, SAC, TD3
- `get_distributions()` / `get_sample_data()` / `get_calculate_data()` — PPO, SAC, TD3
- `get_check_data()` — for evaluation metrics

### `AlgoBaseAgent`

Abstract base for the agent that interacts with environments. Two abstract methods:

```python
def sample_multi_envs(self, model_dict)  # Rollout N parallel envs, return experiences
def check_single_env(self)               # Single-episode evaluation, return scalar metrics
```

- `sample_multi_envs()` runs in Sampler workers. Output format is algorithm-specific (list of trajs or transitions).
- `check_single_env()` runs in Checker worker. Returns dict of scalar metrics (sum_rewards, avg_action, entropy, etc.) for TensorBoard.

### `AlgoBaseCalculate`

Abstract base for gradient computation. One abstract method:

```python
def generate_grads(self, samples, model_dict)  # samples → gradient list + version
```

- Runs in Trainer workers.
- The output `grads_list` is a list of `List[np.ndarray]` — each element of the outer list corresponds to one gradient batch, and each inner list contains the per-parameter gradients as numpy arrays.
- Usually returns `[grads]` (single-element list) for one gradient batch per sample.

---

## buffers.py — Experience Replay Buffers

### `Experience` (namedtuple)

```python
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'next_state'])
```

Immutable single-step transition. Used by DQN and SAC (both off-policy with replay buffers). Note: PPO and TD3 use inline list/array structures instead.

### `ExperienceBuffer`

FIFO buffer of **lists of Experiences** (each `append` adds one rollout segment as a sampling unit).

| Method | Description |
|--------|-------------|
| `__init__(capacity)` | Max elements (deque with maxlen) |
| `append(exps_list)` | Append a list of `Experience` tuples |
| `sample(batch_size)` | Random without-replacement sample → `(states, actions, rewards, dones, next_states)` numpy arrays |

Used by: DQN, SAC

### `TrajectoryBuffer`

FIFO buffer of **whole trajectories** (sequences of timesteps).

| Method | Description |
|--------|-------------|
| `__init__(capacity)` | Max trajectories |
| `append(trajectory)` | Append one trajectory (list of timesteps) |
| `sample(batch_size)` | Random without-replacement → list of trajectories |

Not currently used in any algorithm — exists for future on-policy trajectory replay.

---

## actions.py — Action Selectors

Chainable action selection pipeline. All operate on numpy arrays (CPU-bound sampler/checker workers).

### `ActionSelector` (abstract)

```python
def __call__(scores)  # scores → actions
```

### `ArgmaxActionSelector`

Greedy: `np.argmax(scores, axis=1)`. Used for evaluation and as default fallback.

### `EpsilonGreedyActionSelector`

With probability `ε`, pick uniformly random action; otherwise delegate to inner selector (default argmax).

```python
selector = EpsilonGreedyActionSelector(epsilon=0.05)
actions = selector(q_values)  # batch_size × n_actions → batch_size actions
```

### `ProbabilityActionSelector`

Sample per-row from a probability distribution. Used when the policy outputs probabilities directly.

```python
actions = selector(probs)  # each row: np.random.choice(len(prob), p=prob)
```

### `EpsilonTracker`

Linear ε-decay scheduler. Updates `selector.epsilon` based on global frame count.

```python
tracker = EpsilonTracker(selector, eps_start=1.0, eps_final=0.01, eps_frames=100000)
tracker.frame(50000)  # selector.epsilon ≈ 0.5
```

---

## noisy.py — Noisy Layers and Gradient Tools

### `NoisyLinear(nn.Linear)`

Factorized Gaussian noise linear layer from the NoisyNet paper (Fortunato et al., 2017). Each forward pass during training injects learnable noise into weights and bias:

```python
# Training mode (self.training == True):
output = (W + σ_W ⊙ ε_W) @ input + (b + σ_b ⊙ ε_b)

# Evaluation mode (self.training == False):
output = W @ input + b     # Deterministic, same as nn.Linear
```

- `σ_weight`, `σ_bias`: learnable noise standard deviation parameters
- `ε_weight`, `ε_bias`: random normal buffers (resampled via `sample_noise()`)
- Weight initialization: `U(-√(3/in_features), √(3/in_features))`

`sample_noise()` must be called after each `update_state()` — the main process calls it (see PPO/TD3 `update_state` implementations).

### `GradCoef(autograd.Function)`

Custom autograd function that scales gradients by a coefficient during backward pass:

```python
# Forward: y = x    (identity)
# Backward: ∂L/∂x = coeff * ∂L/∂y
```

Used by PPO Beta variant to implement the reparameterized Beta distribution gradient scaling (`GradCoef.apply(x, 0.1)`).

---

## Dependencies

| Module | Used By |
|--------|---------|
| `base.py` | All algorithms, registry, workers |
| `buffers.py` | DQN calculator, SAC calculator |
| `actions.py` | Not currently used by any algorithm (DQN implements ε-greedy inline) |
| `noisy.py` | PPO mujoco_normal, PPO mujoco_beta, SAC, TD3 (when `USE_NOISE=True`) |

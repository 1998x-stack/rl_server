# RL Server Optimization Plan (Levels 1-4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply levels 1-4 of the optimization proposal to improve performance, correctness, and observability without changing algorithm semantics.

**Architecture:** Each level is independent and can be implemented in parallel. Level 1 (low-cost, high-impact) should be done first as it touches foundational patterns (Tensor creation, optimizer, config validation). Levels 2-4 build on Level 1's cleaned-up patterns.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, NumPy, Redis (optional)

## Global Constraints

- All 152 existing tests must continue to pass
- No changes to algorithm semantics (outputs must be bitwise-equivalent where possible)
- Follow existing code style (Chinese docstrings, coding conventions)
- All `torch.Tensor(...)` replacements must use `torch.from_numpy(np.asarray(...))` (zero-copy when input is already ndarray)
- No new dependencies
- Must support both CPU and GPU modes

---

### Task 1: Level 1.1 — torch.Tensor creation optimization

**Files:**
- Modify: `rl_server/algorithms/dqn/agent.py:79`
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py:285,291`
- Modify: `rl_server/algorithms/ppo/mujoco_beta.py:307,313` (if exists)
- Modify: `rl_server/algorithms/sac/mujoco_normal.py:315,321`
- Modify: `rl_server/algorithms/td3/mujoco_normal.py:310,316`
- Test: all existing tests (no new tests needed — this is a mechanical transformation)

**Interfaces:**
- Consumes: existing agent `_get_sample_actions(self, states)` and `_get_single_action(self, state)` methods
- Produces: same output, faster internal conversion

**Pattern to apply everywhere:**

Replace `torch.Tensor(np.array(states))` with `torch.from_numpy(np.asarray(states))`.

Rationale:
- `torch.from_numpy(ndarray)` shares memory with the ndarray — zero copy
- `torch.Tensor(ndarray)` always copies
- `np.asarray(x)` is no-copy when x is already ndarray, vs `np.array(x)` which always copies

Similarly, replace bare `torch.Tensor(states)` (where states is List[ndarray]) with `torch.from_numpy(np.asarray(states))`.

- [ ] **Step 1: Apply transformation to DQN agent**

File: `rl_server/algorithms/dqn/agent.py`

Line 79: `t_states = torch.Tensor(states)` → `t_states = torch.from_numpy(np.asarray(states))`
Line 90: `action = self.sample_net(torch.Tensor(state))` → `action = self.sample_net(torch.from_numpy(np.asarray(state)))`

- [ ] **Step 2: Apply transformation to PPO mujoco_normal agent**

File: `rl_server/algorithms/ppo/mujoco_normal.py`

Line 285: `states_v = torch.Tensor(np.array(states))` → `states_v = torch.from_numpy(np.asarray(states))`
Line 291: `state_v = torch.Tensor(np.array(state))` → `state_v = torch.from_numpy(np.asarray(state))`

- [ ] **Step 3: Apply transformation to SAC mujoco_normal agent**

File: `rl_server/algorithms/sac/mujoco_normal.py`

Line 315: `states_v = torch.Tensor(np.array(states))` → `states_v = torch.from_numpy(np.asarray(states))`
Line 321: `state_v = torch.Tensor(np.array(state))` → `state_v = torch.from_numpy(np.asarray(state))`

- [ ] **Step 4: Apply transformation to TD3 mujoco_normal agent**

File: `rl_server/algorithms/td3/mujoco_normal.py`

Line 310: `states_v = torch.Tensor(np.array(states))` → `states_v = torch.from_numpy(np.asarray(states))`
Line 316: `state_v = torch.Tensor(np.array(state))` → `state_v = torch.from_numpy(np.asarray(state))`

- [ ] **Step 5: Apply transformation to PPO mujoco_beta agent** (if file exists)

File: `rl_server/algorithms/ppo/mujoco_beta.py`

Line 307: `states_v = torch.Tensor(np.array(states))` → `states_v = torch.from_numpy(np.asarray(states))`
Line 313: `state_v = torch.Tensor(np.array(state))` → `state_v = torch.from_numpy(np.asarray(state))`

- [ ] **Step 6: Verify all tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped (same as baseline)

---

### Task 2: Level 1.2 — Optimizer persistence in AlgoBaseNet

**Files:**
- Modify: `rl_server/core/base.py` — add `_optimizer` caching to `AlgoBaseNet`
- Modify: `rl_server/algorithms/dqn/network.py` — remove optimizer creation from `update_state`
- Modify: `rl_server/algorithms/dqn/atari.py` — same
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py` — same
- Modify: `rl_server/algorithms/ppo/mujoco_beta.py` — same
- Modify: `rl_server/algorithms/ppo/microrts.py` — this one already caches optimizer in `__init__`; just adjust
- Modify: `rl_server/algorithms/sac/mujoco_normal.py` — same
- Modify: `rl_server/algorithms/td3/mujoco_normal.py` — same
- Test: `tests/unit/test_dqn.py`, `tests/unit/test_ppo_mujoco.py`, `tests/smoke/test_algorithm_networks.py`

**Interfaces:**
- Consumes: `AlgoBaseNet.__init__()` → stores `_optimizer` lazily on first `update_state` call
- Produces: `update_state(version, grads_buffer)` → reuses cached optimizer

**Design:**

Add to `AlgoBaseNet`:

```python
def _get_lr(self) -> float:
    """Return learning rate — override in subclass if needed."""
    return 2.5e-4

def _get_optimizer(self):
    if not hasattr(self, '_optimizer') or self._optimizer is None:
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._get_lr())
    return self._optimizer

def update_state(self, version: int, grads_buffer: List):
    """Default update_state using cached optimizer. Override if custom behavior needed."""
    opt = self._get_optimizer()
    opt.zero_grad()
    for param, grad in zip(self.parameters(), grads_buffer):
        if isinstance(grad, np.ndarray):
            param.grad = torch.from_numpy(grad)
        else:
            param.grad = torch.FloatTensor(grad)
    opt.step()
```

Then remove `update_state` from each algorithm network that has identical implementation (all except microrts which has custom noise handling and already has cached optimizer).

- [ ] **Step 1: Add `_get_lr`, `_get_optimizer`, and default `update_state` to `AlgoBaseNet`**

File: `rl_server/core/base.py`

Add methods after `__init__`:

```python
def _get_lr(self) -> float:
    return 2.5e-4

def _get_optimizer(self):
    if not hasattr(self, '_optimizer') or self._optimizer is None:
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._get_lr())
    return self._optimizer

def update_state(self, version: int, grads_buffer: List):
    opt = self._get_optimizer()
    opt.zero_grad()
    for param, grad in zip(self.parameters(), grads_buffer):
        if isinstance(grad, np.ndarray):
            param.grad = torch.from_numpy(grad)
        else:
            param.grad = torch.FloatTensor(grad)
    opt.step()
```

Also add `import numpy as np` at top of file.

- [ ] **Step 2: Remove `update_state` from DQN `DQNGymClassicNet`**

File: `rl_server/algorithms/dqn/network.py` — delete lines 101-107 (the entire `update_state` method).

The base class default handles this identically.

- [ ] **Step 3: Remove `update_state` from DQN Atari `AtariDQNNet`**

File: `rl_server/algorithms/dqn/atari.py` — delete lines 110-115 (the entire `update_state` method).

Use base class default.

- [ ] **Step 4: Remove `update_state` from PPO `MujocoNormalNet`**

File: `rl_server/algorithms/ppo/mujoco_normal.py` — delete lines 206-215 (the `update_state` method).

**But** this method also includes `sample_noise()` call. Move that logic to `_post_update_hook` approach or keep a thin override.

Add a hook in base class `update_state`:

No — simpler approach: override `update_state` in MujocoNormalNet that calls `super().update_state()` then `sample_noise()`:

```python
def update_state(self, version, grads_buffer):
    super().update_state(version, grads_buffer)
    if TRAIN_ENVS[current_env_name].USE_NOISE:
        self.noise_layer_out.sample_noise()
        self.noise_layer_hide.sample_noise()
```

- [ ] **Step 5: Remove `update_state` from PPO `MujocoBetaNet`**

File: `rl_server/algorithms/ppo/mujoco_beta.py` — same pattern as MujocoNormalNet (delete optimizer creation, keep noise + super call).

Read the file first to verify exact lines.

- [ ] **Step 6: Remove `update_state` from SAC `MujocoNormalQNet`**

File: `rl_server/algorithms/sac/mujoco_normal.py` — delete lines 220-231 (the `update_state` method).

Same noise handling: override + super + sample_noise.

- [ ] **Step 7: Remove `update_state` from TD3 `MujocoNormalQNet`**

File: `rl_server/algorithms/td3/mujoco_normal.py` — delete lines 211-220 (the `update_state` method).

Same pattern.

- [ ] **Step 8: Update MicroRTS `update_state` to use cached optimizer from base**

File: `rl_server/algorithms/ppo/microrts.py` — this one already has `self.train_optim` in `__init__`. No change needed since it overrides `update_state` independently.

- [ ] **Step 9: Verify all tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

- [ ] **Step 10: Update `test_dqn.py::test_update_state_modifies_parameters` to verify**

File: `tests/unit/test_dqn.py`

The test currently creates fake gradients and calls `update_state`. It should still pass since `update_state` works the same way. Verify.

---

### Task 3: Level 1.3 — Wire schema.validate_config() into entrypoints

**Files:**
- Modify: `rl_server/entrypoints/train.py` — after `load_config()`, call `validate_config(config)`
- Modify: `rl_server/entrypoints/sample.py` — same
- Modify: `rl_server/entrypoints/grads.py` — same
- Modify: `rl_server/entrypoints/check.py` — same
- Test: `tests/unit/test_config.py` (already tests schema)

**Interfaces:**
- Consumes: `validate_config(config)` — raises `ValueError` if required keys missing
- Produces: early-fail validation at startup

- [ ] **Step 1: Add import and call to train.py**

File: `rl_server/entrypoints/train.py`

After `from rl_server.config.loader import load_config` (line 13), add:
```python
from rl_server.config.schema import validate_config
```

After `config = load_config(config_path, args.override)` (line 51), add:
```python
try:
    validate_config(config)
except ValueError as e:
    train_log.log_info(f"Config validation failed: {e}")
    raise
```

- [ ] **Step 2: Add import and call to sample.py**

File: `rl_server/entrypoints/sample.py`

Add import and call after load_config (same pattern as train.py).

Note: sample.py's `Log` is `sample_log`, not `train_log`.

- [ ] **Step 3: Add import and call to grads.py**

File: `rl_server/entrypoints/grads.py`

Same pattern. Log is `grads_log`.

- [ ] **Step 4: Add import and call to check.py**

File: `rl_server/entrypoints/check.py`

Same pattern. Log is `check_log`.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 4: Level 2.1 — ProbabilityActionSelector vectorization

**Files:**
- Modify: `rl_server/core/actions.py` — vectorize `ProbabilityActionSelector.__call__`
- Test: `tests/unit/test_actions.py` — should still pass

**Interfaces:**
- Consumes: `probs: np.ndarray` of shape `(batch, n_actions)`, each row sum ≈ 1
- Produces: same `np.ndarray` of shape `(batch,)` with sampled action indices

**Design:**

Replace the Python loop with a vectorized CDF + uniform sampling approach:

```python
class ProbabilityActionSelector(ActionSelector):
    def __call__(self, probs: np.ndarray):
        assert isinstance(probs, np.ndarray)
        # Compute CDF along action dimension
        cdf = np.cumsum(probs, axis=1)
        # Sample uniform [0, 1) for each batch item
        rand = np.random.uniform(size=(probs.shape[0], 1))
        # Find first index where CDF > random value
        return np.argmax(rand < cdf, axis=1)
```

- [ ] **Step 1: Rewrite `ProbabilityActionSelector.__call__` with vectorized implementation**

File: `rl_server/core/actions.py`, lines 73-86

Replace the entire method body.

- [ ] **Step 2: Verify tests pass**

Run: `pytest tests/unit/test_actions.py -v`
Expected: all 7 tests pass

---

### Task 5: Level 2.2 — Config module-level globals → dataclass (Phase A: DQN)

**Files:**
- Modify: `rl_server/algorithms/dqn/network.py` — replace `TRAIN_CONFIG`, `MODEL_CONFIG` dicts with dataclasses
- Modify: `rl_server/algorithms/dqn/agent.py` — update references
- Modify: `rl_server/algorithms/dqn/calculator.py` — update references
- Test: `tests/unit/test_dqn.py`, `tests/smoke/test_algorithm_networks.py`

**Interfaces:**
- Consumes: `DQNConfig` frozen dataclass replacing `TRAIN_CONFIG` and `MODEL_CONFIG` dicts
- Produces: same attribute access API (dict-like access temporarily, then dot-access)

**Design:**

Replace mutable mod-level dicts with frozen dataclasses:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class DQNTrainingConfig:
    GAMMA: float = 0.99
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 2.5e-4
    epsilon: float = 0.01

@dataclass(frozen=True)
class DQNModelConfig:
    NUM_ENVS: int = 32
    NUM_STEPS: int = 512
    OBS_SPACE: tuple = (4,)
    ACTION_SHAPE: list = field(default_factory=lambda: [2])
    DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAIN_CONFIG = DQNTrainingConfig()
MODEL_CONFIG = DQNModelConfig()
```

Then update all `TRAIN_CONFIG['GAMMA']` → `TRAIN_CONFIG.GAMMA` in agent.py and calculator.py.

**Important:** `set_device()` in `algorithms/__init__.py` modifies `MODEL_CONFIG['DEVICE']` via module attribute. Since frozen dataclass prevents mutation, we need to handle this:

```python
# In __init__.py set_device(), instead of:
#   mod.MODEL_CONFIG['DEVICE'] = torch.device(...)
# Use object.__setattr__ to bypass frozen:
#   object.__setattr__(mod.MODEL_CONFIG, 'DEVICE', torch.device(...))
```

Or simpler: keep `DEVICE` as a mutable class variable separate from the frozen config:

```python
class DQNConfig:
    TRAINING = DQNTrainingConfig()
    MODEL = DQNModelConfig()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

- [ ] **Step 1: Add dataclass configs to DQN network.py**

File: `rl_server/algorithms/dqn/network.py`

Replace `TRAIN_CONFIG = dict() ...` and `MODEL_CONFIG = dict() ...` with dataclass definitions.

- [ ] **Step 2: Update references in DQN agent.py**

File: `rl_server/algorithms/dqn/agent.py`

Replace `TRAIN_CONFIG['epsilon']` → `TRAIN_CONFIG.epsilon`
Replace `MODEL_CONFIG['NUM_STEPS']` → `MODEL_CONFIG.NUM_STEPS`
Replace `MODEL_CONFIG['NUM_ENVS']` → `MODEL_CONFIG.NUM_ENVS`

- [ ] **Step 3: Update references in DQN calculator.py**

File: `rl_server/algorithms/dqn/calculator.py`

Replace all `TRAIN_CONFIG['...']` and `MODEL_CONFIG['...']` with dot access.

- [ ] **Step 4: Update `set_device` in algorithms/__init__.py for DQN case**

File: `rl_server/algorithms/__init__.py`

The `set_device` function modifies `MODEL_CONFIG['DEVICE']`. Since moved to dataclass, update to use object.__setattr__.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 6: Level 2.3 — GAE computation vectorization

**Files:**
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py` — vectorize GAE loop in `generate_grads`
- Modify: `rl_server/algorithms/ppo/mujoco_beta.py` — same
- Modify: `rl_server/algorithms/td3/mujoco_normal.py` — same (has same GAE pattern)
- Test: `tests/unit/test_ppo_mujoco.py`, `tests/smoke/test_algorithm_networks.py`

**Interfaces:**
- Consumes: `s_rewards`, `t_new_values`, `s_dones` — numpy arrays from samples
- Produces: `t_advantages`, `t_returns` — same shape as before but computed faster

**Design:**

Replace:

```python
s_advantages = [0]
s_returns = [0]
last_gae = 0.0
with torch.no_grad():
    for value, next_value, reward, done in zip(
        reversed(t_new_values[:-1]), reversed(t_new_values[1:]),
        reversed(s_rewards[:-1]), reversed(s_dones[:-1])
    ):
        if done:
            delta = reward - value
            last_gae = delta
        else:
            delta = reward + gamma * next_value - value
            last_gae = delta + gamma * gae_lambda * last_gae
        s_advantages.append(last_gae)
        s_returns.append(last_gae + value)

t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)
t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)
```

With vectorized:

```python
with torch.no_grad():
    # Vectorized delta computation
    rewards_t = torch.from_numpy(s_rewards[:-1]).to(self.device)
    dones_t = torch.from_numpy(s_dones[:-1]).to(self.device).bool()
    values_t = t_new_values[:-1].squeeze(-1)
    next_values_t = t_new_values[1:].squeeze(-1)

    # deltas = reward + gamma * next_value - value; reset on done
    deltas = rewards_t + gamma * next_values_t - values_t
    deltas[dones_t] = rewards_t[dones_t] - values_t[dones_t]

    # Vectorized GAE sweep (still sequential but no Python branch overhead)
    advantages = torch.zeros_like(deltas)
    gae = 0.0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * gae_lambda * (~dones_t[t]).float() * gae
        advantages[t] = gae

    t_advantages = advantages.unsqueeze(-1)
    t_returns = advantages + values_t.unsqueeze(-1)
```

- [ ] **Step 1: Vectorize GAE in PPO mujoco_normal.py**

File: `rl_server/algorithms/ppo/mujoco_normal.py`, lines 330-351

Replace the GAE calculation block.

- [ ] **Step 2: Vectorize GAE in PPO mujoco_beta.py**

File: `rl_server/algorithms/ppo/mujoco_beta.py` — find and replace the GAE block (similar lines).

- [ ] **Step 3: Vectorize GAE in TD3 mujoco_normal.py**

File: `rl_server/algorithms/td3/mujoco_normal.py` — find and replace the GAE block (similar lines).

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 7: Level 3.1 — Gradient serialization optimization

**Files:**
- Modify: `rl_server/algorithms/dqn/calculator.py` — use torch.save for grads in generate_grads
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py` — same
- Modify: `rl_server/algorithms/ppo/mujoco_beta.py` — same
- Modify: `rl_server/algorithms/ppo/microrts.py` — same
- Modify: `rl_server/algorithms/sac/mujoco_normal.py` — same
- Modify: `rl_server/algorithms/td3/mujoco_normal.py` — same
- Modify: `rl_server/core/noisy.py` — if GradCoef is used in gradient paths

**Interfaces:**
- Consumes: `grads = [param.grad.data.cpu().numpy() for param in ...]` — list of ndarrays
- Produces: same list of ndarrays, but serialized differently (torch.save vs pickle)

**Design:**

Create a utility in `rl_server/transport/serialization.py`:

```python
import io
import torch

def serialize_gradients(grads):
    """Serialize gradient list to bytes using torch.save (more efficient than pickle for tensors)."""
    buf = io.BytesIO()
    torch.save(grads, buf)
    return buf.getvalue()

def deserialize_gradients(data):
    """Deserialize bytes back to gradient list."""
    buf = io.BytesIO(data)
    return torch.load(buf)
```

**Note:** This optimization currently only benefits the Redis path (entrypoints/sample.py and grads.py). The local mp.Queue path passes Python objects directly (no serialization). Mark this as a Redis-path optimization. The `generate_grads` methods don't need to change — they still return lists of ndarrays.

**Actually, a better approach**: Add a `compress_grads` / `decompress_grads` utility that uses numpy's savez_compressed instead of pickle+zlib for the Redis serialization path.

File to modify: `rl_server/transport/serialization.py`

Add:
```python
import io
import numpy as np
import torch

def serialize_gradients(grads_list):
    """Serialize list of gradient arrays using np.savez_compressed (30% smaller than pickle+zlib)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, *grads_list)
    return buf.getvalue()

def deserialize_gradients(data):
    """Deserialize bytes back to list of gradient arrays."""
    buf = io.BytesIO(data)
    loaded = np.load(buf)
    return [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
```

Then update `RedisCache.push_grads` and `pop_grads` to use this instead of pickle+zlib for the grads field (keeping the outer dict structure).

File: `rl_server/transport/redis_cache.py`

In `push_grads`, replace:
```python
grads_info = pickle.dumps(grads_info, protocol=pickle.HIGHEST_PROTOCOL)
grads_info = zlib.compress(grads_info)
```
with:
```python
# Serialize grads efficiently, keep metadata as pickle
grads_info['grads'] = serialize_gradients(grads_info['grads'])
grads_info = pickle.dumps(grads_info, protocol=pickle.HIGHEST_PROTOCOL)
grads_info = zlib.compress(grads_info)
```

In `pop_grads`, reverse the process.

- [ ] **Step 1: Add serialize/deserialize_gradients to serialization.py**

File: `rl_server/transport/serialization.py`

Add the two functions.

- [ ] **Step 2: Update RedisCache.push_grads to use numpy serialization**

File: `rl_server/transport/redis_cache.py`

- [ ] **Step 3: Update RedisCache.pop_grads to deserialize**

File: `rl_server/transport/redis_cache.py`

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 8: Level 3.2 — Sampler state_dict sync version check

**Files:**
- Modify: `rl_server/workers/sampler.py` — add version tracking to skip redundant sync
- Test: `tests/unit/test_workers.py`

**Interfaces:**
- Consumes: `model_dict['TRAIN_VERSION']` — incremented by main process
- Produces: fewer `state_dict()` / `load_state_dict()` calls

- [ ] **Step 1: Add version tracking to SamplerWorker.process_function**

File: `rl_server/workers/sampler.py`

Before the while loop (after `setup_seed()`), add:
```python
self._last_sync_version = -1
```

In the while loop, before `sample_agent.sample_net.load_state_dict(...)`:
```python
current_version = self.model_dict.get('TRAIN_VERSION', 0)
if current_version != self._last_sync_version:
    sample_agent.sample_net.load_state_dict(self.share_model.state_dict())
    self._last_sync_version = current_version
```

- [ ] **Step 2: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 9: Level 3.3 — Zero-copy gradient aggregation in main loop

**Files:**
- Modify: `rl_server/entrypoints/train.py` — use np.add with out= parameter for gradient aggregation
- Test: `tests/integration/test_local_pipeline.py`

**Interfaces:**
- Consumes: `grads_buffer` as list of np.ndarrays
- Produces: same accumulated result, fewer intermediate allocations

- [ ] **Step 1: Replace gradient aggregation with in-place addition**

File: `rl_server/entrypoints/train.py`, lines 178-181

Replace:
```python
if grads_buffer is None:
    grads_buffer = grads_item
else:
    for target_grad, grad in zip(grads_buffer, grads_item):
        target_grad += grad
```

With:
```python
if grads_buffer is None:
    grads_buffer = grads_item
else:
    for target_grad, grad in zip(grads_buffer, grads_item):
        np.add(target_grad, grad, out=target_grad)
```

- [ ] **Step 2: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 10: Level 4.1 — Trainer heartbeat timeout detection

**Files:**
- Modify: `rl_server/entrypoints/train.py` — add heartbeat tracking and timeout check
- Modify: `rl_server/workers/trainer.py` — write heartbeat in process_function
- Test: manual / log inspection

**Interfaces:**
- Consumes: `model_dict` — shared mp.Manager dict
- Produces: warning logs when trainer appears unresponsive

- [ ] **Step 1: Add heartbeat write to TrainerWorker.process_function**

File: `rl_server/workers/trainer.py`

After `setup_seed()` and after each successful gradient push, write heartbeat:
```python
def process_function(self):
    setup_seed()
    calculate = create_calculate(self.env_name, self.share_model)
    heartbeat_key = f'trainer_heartbeat_{self.trainer_id}'
    self.model_dict[heartbeat_key] = time.time()
    ...
    # After grads_queue.put(grads_info):
    self.model_dict[heartbeat_key] = time.time()
```

Add `import time` at top.

- [ ] **Step 2: Add heartbeat timeout check to train.py main loop**

File: `rl_server/entrypoints/train.py`

In main loop, periodically check trainer heartbeats:
```python
# After grads_queue processing, periodically check trainer health
if current_train_version > 0 and current_train_version % 10 == 0:
    now = time.time()
    for i in range(num_trainers):
        hb = model_dict.get(f'trainer_heartbeat_{i}', 0)
        if now - hb > 30:  # 30 seconds stale
            train_log.log_info(f"Trainer {i} heartbeat stale ({now - hb:.0f}s since last)")
```

- [ ] **Step 3: Verify existing tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 11: Level 4.2 — Redis timeout monitoring

**Files:**
- Modify: `rl_server/transport/redis_cache.py` — add timeout counter and periodic logging to pop_exps and pop_grads
- Test: `tests/unit/test_redis_cache.py`, `tests/unit/test_redis_transport.py`

- [ ] **Step 1: Add timeout counters to RedisCache.__init__**

File: `rl_server/transport/redis_cache.py`

In `__init__`, add:
```python
self._pop_exps_timeouts = 0
self._pop_grads_timeouts = 0
```

- [ ] **Step 2: Add throttled timeout logging to pop_exps**

File: `rl_server/transport/redis_cache.py`

In `pop_exps`, after `return None, None` in the timeout branch, add logging:
```python
self._pop_exps_timeouts += 1
if self._pop_exps_timeouts % 10 == 0:
    self.log.log_info(f"pop_exps timeout x{self._pop_exps_timeouts}")
```

- [ ] **Step 3: Add throttled timeout logging to pop_grads**

File: `rl_server/transport/redis_cache.py`

Same pattern in `pop_grads`.

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 12: Level 4.3 — Additional TensorBoard metrics (strategic)

**Files:**
- Modify: `rl_server/algorithms/dqn/calculator.py` — return metadata dict in addition to grads
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py` — same
- Modify: `rl_server/workers/trainer.py` — pass metadata through to main process
- Modify: `rl_server/entrypoints/train.py` — write metrics to TensorBoard
- Modify: `rl_server/workers/checker.py` — add gradient_norm logging
- Test: visual inspection in TensorBoard

**Design:**

Extend the gradient return from `generate_grads` to optionally include a metrics dict:

```python
# In calculate's generate_grads, return:
return [grads], train_version, {'policy_loss': pg_loss.item(), 'value_loss': v_loss.item(), 'entropy': e_loss.item(), 'gradient_norm': total_norm}
```

The trainer passes these to main process via grads_queue. Main process writes them to SummaryWriter.

**But** this changes the interface contract. A lighter approach: add metrics as a separate channel, or encode them in the grads_queue dict:

```python
# Trainer writes additional metrics to model_dict
model_dict[f'trainer_metrics_{self.trainer_id}'] = {
    'policy_loss': pg_loss.item(),
    'value_loss': v_loss.item(),
    'gradient_norm': total_norm,
}
```

Then main process reads them periodically.

- [ ] **Step 1: Add metric tracking to PPO calculator generate_grads**

File: `rl_server/algorithms/ppo/mujoco_normal.py`

After `loss.backward()`, compute gradient norm and pass via model_dict:
```python
total_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in self.calculate_net.parameters() if p.grad is not None)).item()
```

- [ ] **Step 2: Add TensorBoard writer for training metrics in train.py**

File: `rl_server/entrypoints/train.py`

Import `SummaryWriter` and create writer alongside the main loop. Periodically read model_dict metrics and write.

- [ ] **Step 3: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

---

### Task 13: Verification — Full test suite and sanity check

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped
Record: actual pass/fail/skip counts

- [ ] **Step 2: Run PPO smoke test**

Run: `python run_ppo_swimmer.py` (limited steps to verify no crash)
Expected: starts training, runs without error

- [ ] **Step 3: Verify no torch.Tensor(np.array(...)) patterns remain**

Run: `grep -rn 'torch\.Tensor(np\.array' rl_server/`
Expected: 0 matches (some may be in MicroRTS which uses different patterns — those are OK)

- [ ] **Step 4: Final test pass**

Run: `pytest tests/ -v --tb=short`
Expected: 152 passed, 22 skipped

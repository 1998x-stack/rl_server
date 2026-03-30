# RL Server Tutorial

A step-by-step guide to get started with RL Server — from installation to training your first agent to adding custom algorithms.

## Table of Contents

1. [Setup](#1-setup)
2. [Your First Training Run: DQN on CartPole](#2-your-first-training-run-dqn-on-cartpole)
3. [Understanding the Pipeline](#3-understanding-the-pipeline)
4. [Training PPO on MuJoCo](#4-training-ppo-on-mujoco)
5. [Configuration Deep Dive](#5-configuration-deep-dive)
6. [Distributed Training with Redis](#6-distributed-training-with-redis)
7. [Adding a New Algorithm](#7-adding-a-new-algorithm)
8. [Testing](#8-testing)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Setup

### Prerequisites

- Python 3.10 or later
- pip (or conda)
- Git

### Install

```bash
# Clone the repo
git clone <repo-url>
cd rl_server

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
pytest tests/ -v
```

You should see **174 tests passed**. If some MuJoCo tests are skipped, that's expected if you haven't installed MuJoCo yet (see [Section 4](#4-training-ppo-on-mujoco)).

---

## 2. Your First Training Run: DQN on CartPole

CartPole is the simplest environment — a pole balanced on a cart. The agent chooses left or right at each step.

### Start Training

```bash
python -m rl_server.entrypoints.train --env-name DQNGymClassic
```

This launches:
- **2 Sampler workers** — collect experience by running CartPole environments
- **1 Trainer worker** — compute gradients from experiences
- **1 Checker worker** — periodically evaluate the agent and log to TensorBoard
- **Main process** — aggregate gradients and update the shared network

### What Happens Under the Hood

```
1. Samplers run CartPole with epsilon-greedy exploration
2. Experiences (state, action, reward, done) are pushed to a shared queue
3. Trainer pulls batches, computes DQN loss and gradients
4. Main process averages gradients and calls net.update_state()
5. Checker evaluates the greedy policy and logs rewards
```

### Monitor Training

Open a new terminal:

```bash
tensorboard --logdir logs/
```

Navigate to `http://localhost:6006` to see reward curves.

### Stop Training

Press `Ctrl+C`. The system handles `SIGINT` gracefully:
1. Workers finish their current iteration
2. A checkpoint is saved automatically
3. All processes exit cleanly

### Resume Training

```bash
python -m rl_server.entrypoints.train --env-name DQNGymClassic --version 500
```

This loads the checkpoint at version 500 and continues training.

---

## 3. Understanding the Pipeline

RL Server uses a **producer-consumer** architecture with three roles:

### The Three Components

| Component | Class | What It Does |
|-----------|-------|-------------|
| **Network** (`AlgoBaseNet`) | `DQNGymClassicNet` | Defines the neural network. Takes states, returns actions or Q-values. |
| **Agent** (`AlgoBaseAgent`) | `DQNGymClassicAgent` | Runs environments, collects experience using the network. |
| **Calculator** (`AlgoBaseCalculate`) | `DQNGymClassicCalculate` | Takes experience batches, computes loss and gradients. |

### Data Flow

```
              Sampler Workers                    Trainer Workers
              ┌────────────┐                     ┌────────────┐
 Env ──obs──> │   Agent    │ ──experiences──>    │ Calculator │ ──gradients──>
              │ (rollout)  │    sample_queue      │ (loss+grad)│    grads_queue
              └────────────┘                     └────────────┘
                    ^                                                  |
                    |                                                  v
                    |              Main Process                        |
                    └──── shared ── [  Network  ] <── update_state ────┘
                           model    (PyTorch)       (avg gradients)
```

### Key Concepts

**Shared Memory Model**: The network is created with `net.share_memory()`, allowing all worker processes to read its parameters without copying.

**Version Tracking**: Every gradient batch is tagged with the model version it was computed from. Stale gradients (computed from an old model) can be detected and filtered.

**Checkpoint Versioning**: Models are saved with version numbers. Atomic file writes (save to `.tmp`, then rename) prevent corruption if the process is killed mid-save.

---

## 4. Training PPO on MuJoCo

### Install MuJoCo

```bash
pip install gymnasium[mujoco]
```

Verify:

```python
import gymnasium as gym
env = gym.make("Swimmer-v4")
obs, _ = env.reset()
print(f"obs shape: {obs.shape}")  # (8,)
env.close()
```

### Quick PPO Run (No Multiprocessing)

The included `run_ppo_swimmer.py` script runs a self-contained PPO training loop without Redis or multiprocessing — useful for debugging and quick verification:

```bash
python run_ppo_swimmer.py
```

Expected output:

```
PPO Training: Swimmer-v4
  obs_dim=8, act_dim=2
  num_envs=4, num_steps=200, iterations=10

Pre-training eval reward: 22.15

  Iter 1/10: version=1, grads_from=4 envs, time=1.2s
  Iter 2/10: version=2, grads_from=4 envs, time=2.3s
  ...

Training complete in 12.5s
Post-training eval reward: 25.40
Improvement: +3.25
```

### Full PPO Training (Multiprocess)

```bash
python -m rl_server.entrypoints.train --env-name MujocoNormal
```

By default this uses `Swimmer-v4`. To change the environment, modify the `current_env_name` variable in `rl_server/algorithms/ppo/mujoco_normal.py`:

```python
current_env_name = 'HalfCheetah'  # or 'Ant', 'Hopper', 'Walker2d', etc.
```

### Available MuJoCo Environments

| Name | Env ID | Obs Dim | Act Dim |
|------|--------|---------|---------|
| Swimmer | Swimmer-v4 | 8 | 2 |
| HalfCheetah | HalfCheetah-v4 | 17 | 6 |
| Ant | Ant-v4 | 27 | 8 |
| Hopper | Hopper-v4 | 11 | 3 |
| Walker2d | Walker2d-v4 | 17 | 6 |
| Humanoid | Humanoid-v4 | 376 | 17 |
| HumanoidStandup | HumanoidStandup-v4 | 376 | 17 |
| Pusher | Pusher-v5 | 23 | 7 |
| Reacher3D | Reacher-v4 | 11 | 2 |

---

## 5. Configuration Deep Dive

### Config Files

```
config/
  default.yaml    # Production defaults
  dev.yaml        # Local development overrides
```

### Config Structure

```yaml
redis:
  model:                    # Redis for model broadcast
    host: "${REDIS_MODEL_HOST:localhost}"
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD:}"
  exps:                     # Redis for experience queue
    host: "${REDIS_EXPS_HOST:localhost}"
    port: 6379
    db: 1
  grads:                    # Redis for gradient queue
    host: "${REDIS_GRADS_HOST:localhost}"
    port: 6379
    db: 2

training:
  env_name: "DQNGymClassic" # Algorithm registry key
  num_samplers: 2           # Number of sampler workers
  num_trainers: 1           # Number of trainer workers
  num_update_grads: 1       # Gradients to accumulate before update
  enable_checker: true      # Enable evaluation worker
  checkpoint_interval: 100  # Save every N versions
  max_versions: 10000       # Max checkpoints on disk

queues:
  len_grads_queue: 1000     # Max pending gradient batches
  len_sample_queue: 1000    # Max pending experience batches

logging:
  level: "INFO"             # INFO, DEBUG, WARNING, ERROR
  dir: "logs"               # Log and TensorBoard output directory
  tensorboard: true         # Enable TensorBoard logging
```

### Environment Variables

Config values like `"${REDIS_PASSWORD:}"` are interpolated at load time:
- `${VAR}` — replaced with the environment variable `VAR` (error if unset)
- `${VAR:default}` — replaced with `VAR` if set, otherwise `default`

### Config Overrides

The override file is deep-merged on top of the base config. Only specify the keys you want to change:

```yaml
# config/dev.yaml
training:
  num_samplers: 1
  num_trainers: 1
logging:
  level: "DEBUG"
```

```bash
python -m rl_server.entrypoints.train \
  --config config/default.yaml \
  --override config/dev.yaml
```

---

## 6. Distributed Training with Redis

For scaling beyond a single machine, RL Server uses Redis as the communication backbone.

### Setup

1. **Start Redis** (3 logical databases for model/exps/grads):

```bash
redis-server --port 6379
```

2. **Set environment variables** (optional, defaults to localhost):

```bash
export REDIS_MODEL_HOST=redis-primary.internal
export REDIS_EXPS_HOST=redis-primary.internal
export REDIS_GRADS_HOST=redis-primary.internal
export REDIS_PASSWORD=mysecret
```

### Launch Workers

Each worker type runs as an independent process. Scale samplers horizontally across machines.

```bash
# Machine 1: Gradient aggregation server (1 instance)
python -m rl_server.entrypoints.grads --env-name MujocoNormal

# Machine 2-N: Samplers (scale to N)
python -m rl_server.entrypoints.sample --env-name MujocoNormal

# Machine N+1: Evaluation (1 instance, optional)
python -m rl_server.entrypoints.check --env-name MujocoNormal
```

### How It Works

```
  Machine 2         Machine 3         Machine 1
┌──────────┐     ┌──────────┐     ┌───────────────┐
│ Sampler  │     │ Sampler  │     │  Grads Server  │
│          │     │          │     │                │
│ env.step │     │ env.step │     │ pop grads      │
│    |     │     │    |     │     │ accumulate     │
│ push exp │     │ push exp │     │ update_state() │
│  to Redis│     │  to Redis│     │ push model     │
│          │     │          │     │ save checkpoint│
│ poll     │     │ poll     │     └───────────────┘
│  model   │     │  model   │
└──────────┘     └──────────┘
       \              |              /
        \             |             /
         v            v            v
     ┌──────────────────────────────┐
     │         Redis Server         │
     │  db0: model   db1: exps      │
     │  db2: grads   exit flag      │
     └──────────────────────────────┘
```

### Scaling Guidelines

| Parameter | Low (Dev) | Medium | High |
|-----------|-----------|--------|------|
| `num_samplers` | 1 | 4-8 | 16-64 |
| `batch_update_grads_server` | 1 | 4-8 | 16-32 |
| Redis instances | 1 shared | 1 shared | 3 separate |

---

## 7. Adding a New Algorithm

This section walks through adding a custom algorithm to RL Server.

### Step 1: Create the Algorithm Module

Create a new directory under `rl_server/algorithms/`:

```
rl_server/algorithms/my_algo/
  __init__.py
  network.py
  agent.py
  calculator.py
```

### Step 2: Implement the Network

```python
# rl_server/algorithms/my_algo/network.py
import torch
import torch.nn as nn
import numpy as np
from rl_server.core.base import AlgoBaseNet, layer_init

class MyAlgoNet(AlgoBaseNet):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(4, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 2)),
        )

    def forward(self, states):
        """Return actions given states."""
        return torch.argmax(self.net(states), dim=-1)

    def get_q_values(self, states):
        """Return Q-values for all actions."""
        return self.net(states)

    def update_state(self, version, grads_buffer):
        """Apply gradients to update network parameters."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer.zero_grad()
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        optimizer.step()
```

### Step 3: Implement the Agent

```python
# rl_server/algorithms/my_algo/agent.py
import torch
import numpy as np
import gymnasium as gym
from rl_server.core.base import AlgoBaseAgent

class MyAlgoAgent(AlgoBaseAgent):
    def __init__(self, sample_net, is_checker):
        super().__init__()
        self.sample_net = sample_net
        self.num_envs = 4
        self.num_steps = 128

        if not is_checker:
            self.envs = [gym.make("CartPole-v1") for _ in range(self.num_envs)]
            self.states = [env.reset()[0] for env in self.envs]
        else:
            self.envs = gym.make("CartPole-v1")
            self.states = self.envs.reset()[0]

    def sample_multi_envs(self, model_dict):
        """Collect experiences from parallel environments."""
        exps = [[] for _ in range(self.num_envs)]
        for _ in range(self.num_steps):
            with torch.no_grad():
                states_t = torch.FloatTensor(np.array(self.states))
                actions = self.sample_net(states_t).numpy()
            for i in range(self.num_envs):
                next_state, reward, done, truncated, _ = self.envs[i].step(actions[i])
                if done or truncated:
                    next_state, _ = self.envs[i].reset()
                exps[i].append([self.states[i], actions[i], reward, done,
                                model_dict['TRAIN_VERSION']])
                self.states[i] = next_state
        return exps

    def check_single_env(self):
        """Run one evaluation episode."""
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(self.states).unsqueeze(0)
                action = self.sample_net(state_t).item()
            self.states, reward, done, truncated, _ = self.envs.step(action)
            if done or truncated:
                self.states = self.envs.reset()[0]
                done = True
            total_reward += reward
        return {'sum_rewards': total_reward}
```

### Step 4: Implement the Calculator

```python
# rl_server/algorithms/my_algo/calculator.py
import torch
import numpy as np
from torch.nn import functional as F
from rl_server.core.base import AlgoBaseCalculate

class MyAlgoCalculate(AlgoBaseCalculate):
    def __init__(self, share_model):
        super().__init__()
        from rl_server.algorithms.my_algo.network import MyAlgoNet
        self.share_model = share_model
        self.calculate_net = MyAlgoNet()

    def generate_grads(self, samples, model_dict):
        """Compute gradients from a batch of experiences."""
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        train_version = model_dict['TRAIN_VERSION']

        states = torch.FloatTensor(np.array([s[0] for s in samples]))
        actions = torch.LongTensor(np.array([s[1] for s in samples]))
        rewards = torch.FloatTensor(np.array([s[2] for s in samples]))

        q_values = self.calculate_net.get_q_values(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Simple MSE loss against rewards (simplified for tutorial)
        loss = F.mse_loss(q_selected, rewards)

        self.calculate_net.zero_grad()
        loss.backward()

        grads = [p.grad.data.cpu().numpy() for p in self.calculate_net.parameters()]
        return [grads], train_version
```

### Step 5: Register the Algorithm

Add a new branch in `rl_server/algorithms/__init__.py`:

```python
def _lazy_load(env_name: str):
    try:
        # ... existing branches ...
        elif env_name == 'MyAlgo':
            from rl_server.algorithms.my_algo.network import MyAlgoNet
            from rl_server.algorithms.my_algo.agent import MyAlgoAgent
            from rl_server.algorithms.my_algo.calculator import MyAlgoCalculate
            register(env_name, MyAlgoNet, MyAlgoAgent, MyAlgoCalculate)
        else:
            raise ValueError(f"Unknown algorithm: {env_name}")
    except ImportError as e:
        raise ValueError(f"Failed to load algorithm {env_name}: {e}")
```

### Step 6: Train

```bash
python -m rl_server.entrypoints.train --env-name MyAlgo
```

---

## 8. Testing

### Running Tests

```bash
# Full suite (174 tests)
pytest tests/ -v

# Specific category
pytest tests/unit/ -v
pytest tests/smoke/ -v
pytest tests/integration/ -v

# Single file
pytest tests/unit/test_dqn.py -v

# With pattern matching
pytest tests/ -v -k "test_forward"
```

### Test Structure

```
tests/
  conftest.py              # Shared fixtures (temp_model_dir, small_dqn_net)
  unit/                    # Fast, isolated tests
    test_dqn.py            #   DQN network forward/backward
    test_ppo_mujoco.py     #   PPO network tests
    test_redis_cache.py    #   Redis with fakeredis
    test_checkpoint.py     #   Save/load/retention
    test_config.py         #   YAML loading
    test_workers.py        #   Worker construction (no processes spawned)
    ...
  smoke/                   # Algorithm validation
    test_algorithm_networks.py  # Forward pass for all algorithms
  integration/             # Multi-component flows
    test_pipeline_flow.py  #   Sampler -> Trainer -> Update cycle
    test_local_pipeline.py #   Local training pipeline
    ...
```

### Writing Tests for Your Algorithm

```python
# tests/unit/test_my_algo.py
import torch
import pytest

class TestMyAlgoNet:
    def test_forward_returns_actions(self):
        from rl_server.algorithms.my_algo.network import MyAlgoNet
        net = MyAlgoNet()
        states = torch.randn(3, 4)
        actions = net(states)
        assert actions.shape == (3,)

    def test_update_state_changes_params(self):
        from rl_server.algorithms.my_algo.network import MyAlgoNet
        import numpy as np
        net = MyAlgoNet()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32)
                 for p in net.parameters()]
        net.update_state(1, grads)
        changed = any(not torch.allclose(b, a)
                       for b, a in zip(params_before,
                                       [p.data for p in net.parameters()]))
        assert changed
```

---

## 9. Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'rl_server'`**

Run from the project root, or install in editable mode:

```bash
pip install -e .
```

Or add the project root to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/rl_server:$PYTHONPATH
```

**`gymnasium.error.NameNotFound: Environment Swimmer-v3 doesn't exist`**

Gymnasium 1.0+ removed v2/v3 MuJoCo environments. This project uses v4/v5. If you see this error, you may be importing from the legacy `algo_envs/` directory instead of `rl_server/algorithms/`.

**`redis.exceptions.ConnectionError: Error connecting to localhost:6379`**

Redis is only needed for distributed mode. For local training, use:

```bash
python -m rl_server.entrypoints.train --env-name DQNGymClassic
```

For distributed mode, ensure Redis is running:

```bash
redis-server --port 6379
redis-cli ping  # Should return PONG
```

**`RuntimeError: Cannot re-initialize CUDA in forked subprocess`**

Set the multiprocessing start method to `spawn` (already done in `setup_mp()`). If you're calling training code directly, add:

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

**Tests fail with `pytest.PytestUnraisableExceptionWarning`**

Some environment cleanup warnings are harmless. Run with:

```bash
pytest tests/ -v -W ignore::pytest.PytestUnraisableExceptionWarning
```

**Checkpoint not found when resuming**

Checkpoints are saved as `models/{prefix}_{env_name}_{version}.pt`. Check:

```bash
ls models/train_main_local_DQNGymClassic_*.pt
```

If no files exist, the training hadn't reached a checkpoint interval yet.

---

## Next Steps

- Read the [README](README.md) for full API reference and production deployment details
- Explore `rl_server/algorithms/ppo/mujoco_normal.py` for a complete PPO implementation with GAE, clipped surrogate loss, and value function clipping
- Check `rl_server/core/noisy.py` for NoisyNet exploration (used in DQN and TD3)
- Review `rl_server/transport/redis_cache.py` for the resilience patterns (retry, pool, reconnect)
- Run `tensorboard --logdir logs/` to visualize training curves

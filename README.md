<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/tests-174%20passed-success?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Redis-5+-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">
</p>

<p align="center">
  <h1 align="center">⚡ RL Server</h1>
  <p align="center"><strong>Production-ready distributed reinforcement learning — train DQN, PPO, SAC, and TD3 at scale.</strong></p>
  <p align="center">Local multiprocess → Redis-cluster → Gradient aggregation server. <br>One config file, zero code changes between modes.</p>
</p>

---

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-supported-algorithms">Algorithms</a> •
  <a href="#-deployment-modes">Deployment</a> •
  <a href="#-configuration">Config</a> •
  <a href="TUTORIAL.md">Tutorial</a> •
  <a href="#-testing">Testing</a>
</p>

---

## 🎯 Why RL Server?

| | RL Server | Stable-Baselines3 | CleanRL | RLlib |
|---|---|---|---|---|
| **Distributed training** | ✅ Redis + local MP | ❌ | ❌ | ✅ Ray (heavy) |
| **Zero-code mode switch** | ✅ Same config | — | — | ❌ |
| **Production hardened** | ✅ Graceful shutdown, atomic ckpt | ❌ | ❌ | ⚠️ Partial |
| **Lazy-loading algos** | ✅ Import on demand | ❌ | ❌ | ❌ |
| **Config env interpolation** | ✅ `${VAR:default}` | ❌ | ❌ | ❌ |
| **Redis resilience** | ✅ Retry + backoff + pool | — | — | ❌ |
| **Lines to first train** | 2 | 4 | 15+ | 20+ |
| **Dependencies** | 6 | 12+ | 8+ | 50+ |

**RL Server** is for practitioners who need to go from laptop → cluster without rewriting their training loop. Built ground-up for production with atomic checkpointing, signal-based graceful shutdown, and Redis resilience patterns that survive network partitions.

---

## 📦 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train DQN on CartPole (no Redis, no config — just works)
python -m rl_server.entrypoints.train --env-name DQNGymClassic

# 3. Monitor
tensorboard --logdir logs/
```

**That's it.** Two shell commands from zero to a running RL agent with 2 sampler workers, 1 trainer, and evaluation — all locally.

Want PPO on MuJoCo?

```bash
pip install gymnasium[mujoco]
python -m rl_server.entrypoints.train --env-name MujocoNormal
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        MAIN PROCESS                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │ load_config │───▶│ create_net  │───▶│ while not exit:       │ │
│  │ + override  │    │ share_mem() │    │   grads ← grads_queue │ │
│  └─────────────┘    └─────────────┘    │   accumulate          │ │
│                                        │   update_state()      │ │
│                                        │   TRAIN_VERSION += 1  │ │
│                                        └──────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │ mp.Queue
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   SAMPLERS   │ │   TRAINERS   │ │   CHECKER    │
│   (N proc)   │ │   (M proc)   │ │   (1 proc)   │
│              │ │              │ │              │
│ sample_multi │ │ generate     │ │ check_single │
│ _envs() ─────┼▶│ _grads() ────┼▶│ _env() ──────┼──▶ TensorBoard
│              │ │              │ │              │
│ [Gym envs]   │ │ [Calculate]  │ │ [Eval env]   │
└──────────────┘ └──────────────┘ └──────────────┘
       △              △
       │ model_dict (shared mp.Manager dict)
       │  • is_exit: bool      — shutdown signal
       │  • TRAIN_VERSION: int — global training step
```

Every algorithm is **three classes** implementing a shared contract:

| Base Class | Key Method | Where It Runs |
|---|---|---|
| `AlgoBaseNet(nn.Module)` | `forward(states)` → actions | Main process + all workers |
| | `update_state(version, grads)` | Main process (gradient aggregation) |
| `AlgoBaseAgent` | `sample_multi_envs(model_dict)` → experiences | Sampler workers (CPU) |
| | `check_single_env()` → metrics | Checker worker |
| `AlgoBaseCalculate` | `generate_grads(samples, model_dict)` → gradients | Trainer workers |

---

## ⚙️ Supported Algorithms

| Algorithm | Registry Key | Action Space | Environments |
|---|---|---|---|
| **DQN** (Dueling + NoisyNet) | `DQNGymClassic` | Discrete | CartPole, MountainCar, Acrobot, LunarLander |
| **PPO** (Gaussian policy) | `MujocoNormal` | Continuous | Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid, Pusher, Reacher |
| **PPO** (Beta policy) | `MujocoBeta` | Continuous (bounded) | Same MuJoCo suite |
| **PPO** (Masked actions) | `MicroRTS` | Discrete (masked) | gym-microrTS |
| **SAC** (dual Q, entropy) | `SACMujocoNormal` | Continuous | Same MuJoCo suite |
| **TD3** (clipped double Q) | `TD3MujocoNormal` | Continuous (deterministic) | Same MuJoCo suite |

<details>
<summary><b>📊 MuJoCo environment reference (12+ envs)</b></summary>

| Env | obs_dim | act_dim | Env | obs_dim | act_dim |
|---|---|---|---|---|---|
| Swimmer-v4 | 8 | 2 | Walker2d-v4 | 17 | 6 |
| HalfCheetah-v4 | 17 | 6 | Humanoid-v4 | 376 | 17 |
| Ant-v4 | 27 | 8 | HumanoidStandup-v4 | 376 | 17 |
| Hopper-v4 | 11 | 3 | Pusher-v5 | 23 | 7 |
| Reacher-v4 | 11 | 2 | InvertedPendulum-v4 | 4 | 1 |
| InvertedDoublePendulum-v4 | 11 | 1 | | | |

</details>

### Adding a custom algorithm

```python
# 1. Implement 3 classes in rl_server/algorithms/my_algo/
# 2. Add one elif branch in rl_server/algorithms/__init__.py
# 3. Train it:
python -m rl_server.entrypoints.train --env-name MyAlgo
```

See the full walkthrough in [TUTORIAL.md §7](TUTORIAL.md#7-adding-a-new-algorithm).

---

## 🚀 Deployment Modes

### Mode 1: Local Multiprocess (laptop / single GPU node)

```bash
python -m rl_server.entrypoints.train --config config/default.yaml
```

- Spawns sampler/trainer/checker as subprocesses
- Uses `torch.multiprocessing.Queue` for IPC
- Zero external dependencies

### Mode 2: Redis Distributed (cluster, any number of machines)

```bash
# Machine 1: Gradient aggregation server
python -m rl_server.entrypoints.grads --env-name MujocoNormal

# Machines 2..N: Samplers (scale horizontally)
python -m rl_server.entrypoints.sample --env-name MujocoNormal

# Optional: Evaluation
python -m rl_server.entrypoints.check --env-name MujocoNormal
```

- Redis-backed queues replace `mp.Queue`
- Each worker is an independent process — no shared memory
- Model weights, experiences, and gradients flow through Redis lists

### Mode 3: Gradient Aggregation Server (standalone daemon)

The grads server (`entrypoints/grads.py`) can run as a persistent daemon, consuming from Redis gradient queues and pushing updated models back. Samplers and trainers connect to it as clients.

| Scale | Samplers | Redis | Grads batch |
|---|---|---|---|
| **Dev** | 1–2 | 1 shared instance | 1 |
| **Medium** | 4–8 | 1 shared instance | 4–8 |
| **High** | 16–64 | 3 separate instances | 16–32 |

---

## 🧠 Algorithm Registry

Algorithms are **lazy-loaded** — zero startup overhead, imports happen on first use:

```python
from rl_server.algorithms import create_net, create_agent, create_calculate

net    = create_net('DQNGymClassic')       # Imports dqn/ only now
agent  = create_agent('DQNGymClassic', net)
calc   = create_calculate('DQNGymClassic', net)
```

The registry (`_REGISTRY`) maps `env_name → (NetClass, AgentClass, CalculateClass)`. All 6 built-in algorithms are registered in `rl_server/algorithms/__init__.py:_lazy_load()`.

---

## 🔧 Configuration

```yaml
# config/default.yaml
redis:
  model:
    host: "${REDIS_MODEL_HOST:localhost}"   # ← env var with fallback
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD:}"

training:
  env_name: "DQNGymClassic"
  num_samplers: 2
  num_trainers: 1
  checkpoint_interval: 100
  max_versions: 10000

queues:
  len_grads_queue: 1000
  len_sample_queue: 1000

logging:
  level: "INFO"
  dir: "logs"
  tensorboard: true
```

**Override for dev:** `python -m rl_server.entrypoints.train --override config/dev.yaml`
Merges `dev.yaml` on top of defaults — only specify what you're changing.

**Inject secrets via env:**
```bash
export REDIS_PASSWORD=mysecret
export REDIS_MODEL_HOST=redis-primary.internal
```

---

## 🛡 Production Features

### Graceful Shutdown
`SIGTERM` / `SIGINT` → workers finish current iteration → checkpoint saved → clean exit. No corrupted `.pt` files, no orphaned processes.

### Atomic Checkpoints
Write to `.tmp` → `os.rename()` to final path. If the process is killed mid-save, the `.tmp` is discarded and the last good checkpoint remains intact. Version retention (`max_versions`) keeps disk usage bounded.

```
models/train_main_local_DQNGymClassic/
  train_main_local_DQNGymClassic_100_20260101120000.td
  train_main_local_DQNGymClassic_200_20260101120100.td
```

### Redis Resilience
- Connection pooling (configurable `pool_size`)
- Exponential backoff retry: 1s → 2s → 4s
- `brpop` with 5s timeout (no infinite blocking)
- Automatic reconnect on `ConnectionError`
- `health_check()` for monitoring

### Monitoring
- **TensorBoard**: Checker worker writes evaluation metrics every episode
- **Heartbeat files**: `/tmp/rl_server/{worker_type}_{id}_{pid}` — external monitoring can watch these
- **Structured logging**: Timestamped, leveled, per-process log files

---

## 🧪 Testing

```bash
# Full suite — 174 tests
pytest tests/ -v

# By layer
pytest tests/unit/ -v           # Fast, isolated (no I/O)
pytest tests/smoke/ -v          # Algorithm forward/backward pass
pytest tests/integration/ -v    # Multi-component pipelines (fakeredis)

# Pattern matching
pytest tests/ -v -k "ppo"
```

```
tests/
  conftest.py                  # Shared fixtures
  unit/        (15 files)      # Registry, checkpoint, config, DQN, PPO, Redis transport...
  smoke/       (1 file)        # All 6 algorithms: forward pass + update_state
  integration/ (4 files)       # End-to-end pipeline, Redis pipeline, grad aggregation
```

All Redis tests use `fakeredis` — no Redis server needed. 30s timeout enforced per test.

---

## 📁 Project Structure

```
rl_server/                     # ← Active package (work here)
  algorithms/   dqn/ ppo/ sac/ td3/
  config/       loader.py, default.yaml
  core/         base.py, buffers.py, noisy.py, actions.py
  entrypoints/  train.py, sample.py, grads.py, check.py
  transport/    redis_cache.py, serializer
  utils/        checkpoint.py, logging.py, process.py
  workers/      sampler.py, trainer.py, checker.py, grads_aggregator.py
tests/          unit/  smoke/  integration/
config/         default.yaml, dev.yaml
run_ppo_swimmer.py             # Single-process PPO quick test
```

> **Legacy directories** (`algo_envs/`, `libs/`, `*_main/`) are pre-refactor snapshots — read-only reference, do not modify.

---

<p align="center">
  <sub>Built with PyTorch • Gymnasium • Redis • 174 tests • 3 deployment modes</sub>
</p>

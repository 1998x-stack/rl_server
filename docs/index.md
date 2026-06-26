---
layout: default
title: Home
description: Production-ready distributed reinforcement learning — train DQN, PPO, SAC, and TD3 at scale
---

# ⚡ RL Server

**Production-ready distributed reinforcement learning framework built on PyTorch and Redis.**

Local multiprocess → Redis cluster → gradient aggregation server. One config file, zero code changes between deployment modes.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train DQN on CartPole (no Redis, no config — just works)
python -m rl_server.entrypoints.train --env-name DQNGymClassic

# Monitor
tensorboard --logdir logs/
```

## Features

- **4 Algorithm Families** — DQN (Dueling + NoisyNet), PPO (Normal / Beta / MicroRTS), SAC, TD3
- **12+ MuJoCo Environments** — Swimmer, HalfCheetah, Ant, Hopper, Humanoid, Walker2d, and more
- **3 Deployment Modes** — Local multiprocess, Redis-distributed workers, gradient aggregation server
- **Production Hardened** — Atomic checkpoints, signal-based graceful shutdown, Redis connection pooling with retry, heartbeat monitoring
- **YAML Configuration** — Environment variable interpolation (`${VAR:default}`), config override merging
- **Lazy Algorithm Registry** — Zero startup overhead; algorithms loaded on first use
- **174 Tests** — Unit, integration, and smoke tests with `fakeredis` for Redis mocking

## Deployment Modes

### Local Multiprocess

```bash
python -m rl_server.entrypoints.train --config config/default.yaml
```

Spawns sampler/trainer/checker subprocesses with `mp.Queue` for IPC. Zero external dependencies.

### Redis Distributed

```bash
# Gradient aggregation server
python -m rl_server.entrypoints.grads --env-name MujocoNormal

# Samplers (scale horizontally across machines)
python -m rl_server.entrypoints.sample --env-name MujocoNormal

# Evaluation (optional)
python -m rl_server.entrypoints.check --env-name MujocoNormal
```

Redis-backed queues; each worker is an independent process with no shared memory.

## Supported Algorithms

| Algorithm | Registry Key | Action Space | Key Features |
|-----------|-------------|-------------|--------------|
| **DQN** | `DQNGymClassic` | Discrete | Dueling architecture, NoisyNet, ε-greedy |
| **PPO Normal** | `MujocoNormal` | Continuous (Gaussian) | GAE, clipped surrogate, value loss clipping |
| **PPO Beta** | `MujocoBeta` | Continuous (Beta) | Beta distribution for bounded actions |
| **PPO MicroRTS** | `MicroRTS` | Discrete (Masked) | Invalid action masking |
| **SAC** | `SACMujocoNormal` | Continuous | Dual Q-networks, entropy regularization |
| **TD3** | `TD3MujocoNormal` | Continuous (deterministic) | Clipped double Q, delayed updates |

## Testing

```bash
# Full suite — 174 tests
pytest tests/ -v

# By layer
pytest tests/unit/ -v
pytest tests/smoke/ -v
pytest tests/integration/ -v
```

## License

MIT — see [LICENSE](LICENSE) for details.

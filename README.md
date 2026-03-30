# RL Server

A production-ready distributed reinforcement learning training framework built on PyTorch and Redis. Supports DQN, PPO, SAC, and TD3 algorithms across Gymnasium environments with flexible deployment modes: local multiprocess, Redis-distributed, and gradient aggregation server.

```
Samplers (N)          Trainers (M)          Main Process
  |                      |                      |
  | sample_multi_envs()  | generate_grads()     | update_state()
  v                      v                      v
[Environments] --exp--> [Queue/Redis] --grad--> [Shared Net] --model--> [Checkpoint]
                                                     |
                                                CheckerWorker --> TensorBoard
```

## Features

- **4 Algorithm Families** — DQN (Dueling + NoisyNet), PPO (Normal / Beta / MicroRTS), SAC, TD3
- **12+ MuJoCo Environments** — Swimmer, HalfCheetah, Ant, Hopper, Humanoid, Walker2d, and more
- **3 Deployment Modes** — Local multiprocess, Redis-distributed workers, gradient aggregation server
- **Production Hardened** — Atomic checkpoints, signal-based graceful shutdown, Redis connection pooling with retry, heartbeat monitoring
- **YAML Configuration** — Environment variable interpolation (`${VAR:default}`), config override merging
- **Lazy Algorithm Registry** — Zero startup overhead; algorithms loaded on first use
- **174 Tests** — Unit, integration, and smoke tests with `fakeredis` for Redis mocking

## Architecture

```
rl_server/
  algorithms/          # Algorithm implementations (lazy-loaded registry)
    __init__.py        #   register(), create_net(), create_agent(), create_calculate()
    dqn/               #   Dueling DQN for classic control (CartPole, etc.)
    ppo/               #   PPO Normal (MuJoCo), Beta, MicroRTS
    sac/               #   Soft Actor-Critic with dual Q-networks
    td3/               #   Twin Delayed DDPG
  config/              # YAML loader with env var interpolation + schema validation
    default.yaml       #   Default Redis, training, queue, logging settings
  core/                # Shared abstractions
    base.py            #   AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate
    actions.py         #   Argmax, EpsilonGreedy, Probability action selectors
    buffers.py         #   ExperienceBuffer, TrajectoryBuffer
    noisy.py           #   NoisyLinear (factorized Gaussian noise)
  entrypoints/         # CLI entry points
    train.py           #   Local multiprocess training
    sample.py          #   Redis-based distributed sampler
    check.py           #   Model evaluation worker
    grads.py           #   Gradient aggregation server
  transport/           # Redis communication layer
    redis_cache.py     #   ConnectionPool, retry, reconnect, health_check
    serialization.py   #   pickle + zlib compression
  utils/               # Operational utilities
    checkpoint.py      #   Atomic save/load with version retention
    logging.py         #   Structured logging (file + console)
    process.py         #   Signal handlers, seed, heartbeat files
  workers/             # Subprocess workers
    sampler.py         #   SamplerWorker (env rollout -> queue)
    trainer.py         #   TrainerWorker (queue -> gradients)
    checker.py         #   CheckerWorker (periodic evaluation + TensorBoard)
    grads_aggregator.py#   GradsAggregator (Redis gradient collection)
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.6+
- Redis 5+ (for distributed mode only)

### Installation

```bash
git clone <repo-url> && cd rl_server

# Install runtime dependencies
pip install -r requirements.txt

# Install dev/test dependencies
pip install -r requirements-dev.txt

# (Optional) MuJoCo environments
pip install gymnasium[mujoco]
```

### Run Local Training

```bash
# DQN on CartPole (default)
python -m rl_server.entrypoints.train

# PPO on Swimmer-v4
python -m rl_server.entrypoints.train --env-name MujocoNormal

# With custom config
python -m rl_server.entrypoints.train --config config/default.yaml --override config/dev.yaml

# Resume from checkpoint
python -m rl_server.entrypoints.train --env-name DQNGymClassic --version 500
```

### Run Distributed (Redis)

Start Redis, then launch workers independently:

```bash
# Terminal 1: Gradient aggregation server
python -m rl_server.entrypoints.grads --env-name MujocoNormal

# Terminal 2-N: Samplers (scale horizontally)
python -m rl_server.entrypoints.sample --env-name MujocoNormal

# Terminal N+1: Evaluation (optional)
python -m rl_server.entrypoints.check --env-name MujocoNormal
```

### Run Tests

```bash
# All 174 tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Smoke tests (algorithm forward passes)
pytest tests/smoke/ -v

# Integration tests
pytest tests/integration/ -v
```

## Supported Algorithms

| Algorithm | Registry Key | Policy | Environments | Key Features |
|-----------|-------------|--------|-------------|--------------|
| **DQN** | `DQNGymClassic` | Discrete | CartPole, MountainCar, Acrobot, LunarLander | Dueling architecture, NoisyNet, epsilon-greedy |
| **PPO Normal** | `MujocoNormal` | Continuous (Gaussian) | Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid, Pusher | GAE, clipped surrogate, value loss clipping |
| **PPO Beta** | `MujocoBeta` | Continuous (Beta) | Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid | Beta distribution for bounded actions |
| **PPO MicroRTS** | `MicroRTS` | Discrete (Masked) | MicroRTS | Hierarchical actions, invalid action masking |
| **SAC** | `SACMujocoNormal` | Continuous | Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid | Dual Q-networks, entropy regularization |
| **TD3** | `TD3MujocoNormal` | Continuous (Deterministic) | Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid | Clipped double Q, delayed policy updates, target smoothing |

## Configuration

Configuration uses YAML with environment variable interpolation:

```yaml
# config/default.yaml
redis:
  model:
    host: "${REDIS_MODEL_HOST:localhost}"
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD:}"
  exps:
    host: "${REDIS_EXPS_HOST:localhost}"
    port: 6379
    db: 1
  grads:
    host: "${REDIS_GRADS_HOST:localhost}"
    port: 6379
    db: 2

training:
  env_name: "DQNGymClassic"
  num_samplers: 2
  num_trainers: 1
  num_update_grads: 1
  enable_checker: true
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

Override for specific deployments:

```bash
python -m rl_server.entrypoints.train \
  --config config/default.yaml \
  --override config/dev.yaml
```

Or inject secrets via environment variables:

```bash
export REDIS_PASSWORD=mysecret
export REDIS_MODEL_HOST=redis-primary.internal
python -m rl_server.entrypoints.train
```

## Algorithm Registry

Algorithms are registered lazily and instantiated via factory functions:

```python
from rl_server.algorithms import create_net, create_agent, create_calculate

# Create components
net = create_net('DQNGymClassic')
agent = create_agent('DQNGymClassic', net)
calculator = create_calculate('DQNGymClassic', net)

# Or register a custom algorithm
from rl_server.algorithms import register
register('MyCustomAlgo', MyNet, MyAgent, MyCalculate)
```

All algorithms implement three base classes:

| Class | Method | Purpose |
|-------|--------|---------|
| `AlgoBaseNet` | `forward(states)` | Action inference |
| | `update_state(version, grads)` | Apply gradient update |
| `AlgoBaseAgent` | `sample_multi_envs(model_dict)` | Parallel environment rollout |
| | `check_single_env()` | Single-episode evaluation |
| `AlgoBaseCalculate` | `generate_grads(samples, model_dict)` | Compute parameter gradients |

## CLI Reference

### `train.py` — Local Multiprocess Training

```
python -m rl_server.entrypoints.train [OPTIONS]

  --config PATH       Config YAML path (default: rl_server/config/default.yaml)
  --override PATH     Override config YAML path
  --env-name NAME     Algorithm registry key (e.g., DQNGymClassic, MujocoNormal)
  --prefix PREFIX     Checkpoint file prefix (default: train_main_local)
  --version VERSION   Resume from specific checkpoint version
```

### `sample.py` — Redis Distributed Sampler

```
python -m rl_server.entrypoints.sample [OPTIONS]

  --config PATH       Config YAML path
  --env-name NAME     Algorithm registry key
```

### `grads.py` — Gradient Aggregation Server

```
python -m rl_server.entrypoints.grads [OPTIONS]

  --config PATH       Config YAML path
  --env-name NAME     Algorithm registry key
```

### `check.py` — Evaluation Worker

```
python -m rl_server.entrypoints.check [OPTIONS]

  --config PATH       Config YAML path
  --env-name NAME     Algorithm registry key
```

## Production Features

### Graceful Shutdown

All workers respond to `SIGTERM` / `SIGINT`:

1. Main process sets `model_dict['is_exit'] = True`
2. Workers complete current iteration and exit
3. Checkpoint is saved before process termination

```bash
kill -TERM <pid>  # Triggers graceful shutdown
```

### Checkpoint Management

- **Atomic writes**: Save to `.tmp` then `os.rename()` — no corruption on crash
- **Version retention**: Configurable `max_versions` keeps disk usage bounded
- **Resume training**: `--version` flag loads specific checkpoint

### Redis Resilience

- Connection pooling (configurable pool size)
- Exponential backoff retry (3 attempts: 1s, 2s, 4s)
- `brpop` with 5-second timeout (no infinite blocking)
- `health_check()` for monitoring
- Automatic reconnection on connection failure

### Monitoring

- **TensorBoard**: CheckerWorker writes evaluation metrics
- **Heartbeat files**: Workers write periodic heartbeats for external monitoring
- **Structured logging**: Timestamped, leveled logs with process ID

```bash
tensorboard --logdir logs/
```

## Dependencies

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.6.0 | Neural networks, multiprocessing |
| `gymnasium` | >= 0.29.0 | Environment interface (Gym v26+ API) |
| `numpy` | >= 1.26.4 | Array operations |
| `redis` | >= 5.0.0 | Distributed communication |
| `pyyaml` | >= 6.0 | Configuration loading |
| `tensorboardX` | >= 2.6 | Training metrics visualization |

### Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 8.0 | Test framework |
| `pytest-timeout` | >= 2.2 | Test timeout enforcement (30s) |
| `fakeredis` | >= 2.0 | Redis mocking for tests |

## License

See [LICENSE](LICENSE) for details.

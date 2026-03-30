# RL Server Production Rebuild — Design Spec

**Date:** 2026-03-30
**Goal:** Transform the distributed RL training system into a production-ready, well-tested, cleanly structured Python package.
**Approach:** Hybrid — fix bugs and add tests first, then restructure with the safety net of test coverage.

---

## 1. Current State

A distributed reinforcement learning training system (~35 Python files) supporting DQN, PPO (multiple variants), SAC, and TD3 algorithms. Three execution modes: local multiprocessing, Redis-distributed, and gradient aggregation server. Redis is used for inter-process communication (experiences, models, gradients).

### Critical Issues Found

| # | File | Issue |
|---|------|-------|
| 1 | `grads_main/grads_main.py:91` | `current_grads_version` is undefined — should be `current_train_version` |
| 2 | `train_main_redis/trainer_redis.py` | Missing `import libs.redis_cache as redis_cache` |
| 3 | `libs/redis_cache.py` | `brpop()` blocks indefinitely with no timeout |
| 4 | `libs/config.py` | Hardcoded Redis IPs (`192.168.1.69`) and weak password (`12345678`) |
| 5 | `algo_envs/ppo_microrts.py:251` | Commented-out `del self.env` — environments leak memory |
| 6 | `test.py` | References nonexistent test.ipynb imports; not a real test file |
| 7 | `proto/experience_pb2.py` | Generated protobuf code that is never used anywhere |

---

## 2. Target Architecture

### Package Structure

```
rl_server/
  __init__.py
  core/
    __init__.py
    base.py              # AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate
    actions.py           # ArgmaxActionSelector, EpsilonGreedyActionSelector, ProbabilityActionSelector
    buffers.py           # ExperienceBuffer, TrajectoryBuffer
    noisy.py             # NoisyLinear, GradCoef
  algorithms/
    __init__.py
    dqn/
      __init__.py
      network.py         # DQNGymClassicNet (Dueling DQN)
      agent.py           # DQNGymClassicAgent
      calculator.py      # DQNGymClassicCalculate
    ppo/
      __init__.py
      mujoco_normal.py   # PPO continuous control (Normal distribution)
      mujoco_beta.py     # PPO beta distribution variants (merge beta, beta_alpha, beta_relative into one module with mode parameter)
      microrts.py        # PPO for MicroRTS (hierarchical actions, masked categorical)
      mobile.py          # PPO mobile environment
    sac/
      __init__.py
      mujoco_normal.py   # Soft Actor-Critic
    td3/
      __init__.py
      mujoco_normal.py   # Twin Delayed DDPG
  workers/
    __init__.py
    sampler.py           # Unified SamplerWorker (local queue + Redis modes via Transport)
    trainer.py           # Unified TrainerWorker
    checker.py           # CheckerWorker (evaluation)
    grads_aggregator.py  # GradsAggregatorWorker
  transport/
    __init__.py
    redis_cache.py       # RedisCache with connection pool, retry, timeout, health check
    serialization.py     # pickle/zlib serialization (protobuf interface reserved)
  config/
    __init__.py
    loader.py            # YAML loading with env var interpolation (${VAR:default})
    schema.py            # Config validation
    default.yaml         # Default configuration
  entrypoints/
    __init__.py
    train.py             # Unified training entry point (--mode local|redis)
    sample.py            # Standalone sampling worker
    check.py             # Model evaluation worker
    grads.py             # Gradient aggregation server
  utils/
    __init__.py
    checkpoint.py        # Atomic save/load, version tracking, retention policy
    logging.py           # Structured logging with worker identity
    process.py           # Signal handling, seed setup, process management
tests/
  unit/
    test_config.py
    test_redis_cache.py
    test_actions.py
    test_exps.py
    test_utils.py
    test_algo_base.py
    test_dqn.py
    test_ppo_mujoco.py
    test_ppo_microrts.py
    test_sac.py
    test_td3.py
  integration/
    test_local_pipeline.py
    test_redis_pipeline.py
    test_checkpoint.py
    test_grad_aggregation.py
  conftest.py
config/
  default.yaml
  dev.yaml
  prod.yaml
```

### Configuration System

External YAML files with environment variable interpolation for secrets:

```yaml
redis:
  model:
    host: ${REDIS_MODEL_HOST:localhost}
    port: ${REDIS_MODEL_PORT:6379}
    password: ${REDIS_PASSWORD:}
    db: 0
  exps:
    host: ${REDIS_EXPS_HOST:localhost}
    port: ${REDIS_EXPS_PORT:6379}
    password: ${REDIS_PASSWORD:}
    db: 1

training:
  env_name: "DQNGymClassic"
  num_samplers: 4
  num_trainers: 2
  batch_update_grads: 2
  checkpoint_interval: 100
  max_versions: 10000

logging:
  level: INFO
  dir: logs
  tensorboard: true
```

---

## 3. Production Hardening

### Graceful Shutdown
- Signal handler for SIGTERM/SIGINT on main process
- Sets shared `multiprocessing.Event` checked by all workers
- Workers finish current batch, flush buffers, then exit
- 30-second hard timeout before SIGKILL

### Redis Resilience
- Connection pool (configurable size, default 10)
- All operations: 3 retries with exponential backoff (1s, 2s, 4s)
- `brpop` timeout: 5 seconds (no longer blocks forever)
- Health check: `ping()` before first use + periodic reconnect
- Automatic reconnection on connection loss

### Checkpoint Management
- Save every N versions (configurable, default 100)
- Keep last K checkpoints (configurable, default 5)
- Checkpoint contents: model state_dict, optimizer state, version number, config hash
- Atomic writes: write to `.tmp`, then `os.rename()`

### Monitoring
- Worker heartbeat files in `/tmp/rl_server/` (per worker PID)
- TensorboardX metrics: reward, loss, gradient norms, throughput (samples/sec), Redis queue depth
- Structured logging: `[timestamp] [level] [worker_type:pid] message`

### Error Handling
- All Redis operations wrapped with retry + exponential backoff
- Process-level exception handlers with structured error logging
- Replace file-based exit signaling with signal handling

---

## 4. gym to gymnasium Migration

Migrate from deprecated `gym==0.26.2` to `gymnasium>=0.29.0`:

- `import gym` → `import gymnasium as gym`
- `env.step()` returns `(obs, reward, terminated, truncated, info)` instead of `(obs, reward, done, info)`
- `done = terminated or truncated` for backward compatibility in experience tuples
- `env.reset()` returns `(obs, info)` instead of just `obs`
- Update all environment creation calls to use `gymnasium.make()`

---

## 5. Testing Strategy

### Unit Tests (target: <30s)
- All core components (networks, agents, calculators, Redis cache, config)
- No external dependencies — Redis mocked via `fakeredis`
- Each algorithm tested for: forward pass shape correctness, non-zero gradient computation, valid action sampling
- Small network variants (hidden_size=16) for fast execution

### Integration Tests (target: <2min, marked `@pytest.mark.integration`)
- Full local training pipeline: sample → train → checkpoint (1-2 iterations)
- Redis pipeline: push exps → pop → train → push model (requires running Redis)
- Checkpoint: save → reload → verify identical forward pass
- Gradient aggregation: multiple grads → correct model update

### Test Dependencies
- `pytest>=8.0`
- `pytest-timeout>=2.2`
- `fakeredis>=2.0`

---

## 6. Dependencies

```
# requirements.txt
numpy>=1.26.4
gymnasium>=0.29.0
torch>=2.6.0
tensorboardX>=2.6
redis>=5.0.0
pyyaml>=6.0

# requirements-dev.txt
fakeredis>=2.0
pytest>=8.0
pytest-timeout>=2.2
```

---

## 7. Implementation Stages

Each stage is independently committable and pushable.

| Stage | Description | Git Message |
|-------|-------------|-------------|
| 1 | Fix critical bugs (undefined vars, missing imports, blocking brpop, env leaks) | `fix: resolve critical bugs in grads_main, trainer_redis, redis_cache` |
| 2 | Security: extract credentials to YAML config system with env var interpolation | `feat: add YAML config system, remove hardcoded credentials` |
| 3 | Error handling: retry logic, signal-based shutdown, structured logging | `feat: add production error handling, graceful shutdown, structured logging` |
| 4 | Add comprehensive test suite (unit + integration) against current structure | `test: add unit and integration tests for all core components` |
| 5 | Package restructure into `rl_server/` with clean module boundaries | `refactor: restructure into rl_server package` |
| 6 | Migrate gym → gymnasium, update all env creation code | `feat: migrate from gym to gymnasium` |
| 7 | Production hardening: checkpoints, monitoring, heartbeats, dependency update | `feat: add checkpoint management, monitoring, production dependencies` |
| 8 | Update tests for new structure, final integration tests | `test: update tests for new package structure, add pipeline integration tests` |

---

## 8. Out of Scope

- Ray/RLlib integration (future consideration)
- Kubernetes deployment manifests
- Web UI / API endpoints
- Algorithm hyperparameter tuning
- Multi-GPU training support
- CI/CD pipeline setup

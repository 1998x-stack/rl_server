# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# All tests (30s timeout per test via pytest-timeout)
pytest tests/ -v

# Subset by layer
pytest tests/unit/ -v          # Fast, isolated tests
pytest tests/smoke/ -v         # Algorithm forward-pass validation
pytest tests/integration/ -v   # Multi-component flows (needs fakeredis)

# Single file / pattern match
pytest tests/unit/test_dqn.py -v
pytest tests/ -v -k "test_forward"

# Run local training (no Redis needed)
python -m rl_server.entrypoints.train --env-name DQNGymClassic
python -m rl_server.entrypoints.train --env-name MujocoNormal

# Quick PPO smoke test (single-process, no multiprocessing)
python run_ppo_swimmer.py
```

No linter or type-checker is configured in this repo. `pytest.ini` sets `timeout = 30` and marks `integration` tests.

## Architecture

### Three-component abstraction

Every algorithm exposes three classes, all constructed via the lazy registry in `rl_server/algorithms/__init__.py`:

| Class | Method | Runs in | Purpose |
|---|---|---|---|
| `AlgoBaseNet` | `forward(states)` | Main + all workers | Inference |
| | `update_state(version, grads_buffer)` | Main process | Apply aggregated gradients |
| `AlgoBaseAgent` | `sample_multi_envs(model_dict)` | Sampler worker | Rollout N environments |
| | `check_single_env()` | Checker worker | Single-episode eval |
| `AlgoBaseCalculate` | `generate_grads(samples, model_dict)` | Trainer worker | Compute gradients from batch |

`model_dict` is a `mp.Manager().dict()` shared across processes with keys `is_exit` (bool), `TRAIN_VERSION` (int). It is the only cross-process communication channel besides `mp.Queue`.

### Producer-consumer pipeline (local mode)

```
SamplerWorkers ──experiences──> sample_queue (mp.Queue)
TrainerWorkers <──samples────── sample_queue
TrainerWorkers ──gradients────> grads_queue (mp.Queue)
Main process   <──grads──────── grads_queue
Main process aggregates grads, calls net.update_state(), increments TRAIN_VERSION
```

### Three deployment modes

1. **Local multiprocess** (`rl_server/entrypoints/train.py`) — spawns sampler/trainer/checker subprocesses, uses `mp.Queue` for communication. The default and most common mode.
2. **Redis-distributed** (`entrypoints/sample.py`, `entrypoints/grads.py`) — workers run as independent processes on different machines, communicate via Redis lists. Model syncs through `RedisCache.set_train_version_model` / `get_train_model`.
3. **Gradient aggregation server** (`entrypoints/grads.py`) — a standalone process pulls gradients from Redis, aggregates, and pushes updated model back.

### Algorithm registry (lazy loading)

`rl_server/algorithms/__init__.py` holds a `_REGISTRY` dict and three factory functions (`create_net`, `create_agent`, `create_calculate`). On first use of an `env_name`, `_lazy_load()` imports the specific algorithm subpackage and calls `register()`. This avoids importing all algorithms at startup.

To add a new algorithm: add an `elif` branch in `_lazy_load()`, then implement Net/Agent/Calculate classes in a new `rl_server/algorithms/<name>/` package.

### Config system

`rl_server/config/loader.py` loads YAML, deep-merges an optional override file, then interpolates `${VAR:default}` patterns from environment variables. There is a `schema.py` for validation but it is not yet wired into the entry points.

### Checkpoint format

Saved as `.td` files under `models/{prefix}/` with naming `{prefix}_{version}_{timestamp}.td`. Each file is a `torch.save` dict with keys `state_dict`, `version`, `timestamp`. Atomic writes: save to `.tmp` then `os.rename()`.

### Redis key space (distributed mode)

When using `RedisCache` (`rl_server/transport/redis_cache.py`):
- `train_model` / `TRAIN_VERSION` — model weights broadcast by grads server
- `exps` (list) — experience batches, `lpush` by samplers, `brpop` by trainers
- `grads` (list) — gradient batches, `lpush` by trainers, `brpop` by grads server
- `exit` — global shutdown flag

All values are `pickle + zlib.compress`. `brpop` uses 5s timeout. Writes have 3-retry exponential backoff (1s, 2s, 4s) on `ConnectionError`.

### Process conventions

- `setup_mp()` sets start method to `spawn` (required for CUDA compatibility) and limits BLAS threads to 1
- `setup_seed()` fixes all random seeds to `1970010101`
- Signal handlers for SIGTERM/SIGINT set a `threading.Event` checked by `should_exit()` in the main loop
- Workers detect shutdown via `model_dict['is_exit'] == True` and exit after current iteration

## Directory map

```
rl_server/
  algorithms/       # Algorithm impls: dqn/, ppo/ (3 variants), sac/, td3/
  config/           # YAML loader + default.yaml
  core/             # AlgoBaseNet/Agent/Calculate, buffers, noisy layers, action selectors
  entrypoints/      # CLI scripts: train.py, sample.py, grads.py, check.py
  transport/        # RedisCache (Redis ops), serialization (pickle+zlib)
  utils/            # checkpoint, logging, process (signals/seeds/heartbeats)
  workers/          # SamplerWorker, TrainerWorker, CheckerWorker, GradsAggregator

algo_envs/          # LEGACY — original monolithic algo files, kept for reference
libs/               # LEGACY — original lib modules before rl_server/ package split
check_main/         # LEGACY — original checker entrypoint
grads_main/         # LEGACY — original grads server entrypoint
sample_main/        # LEGACY — original Redis sampler entrypoint
train_main_local/   # LEGACY — original local training entrypoint
train_main_grads/   # LEGACY — original grads training entrypoint
train_main_redis/   # LEGACY — original Redis training entrypoint
```

The `rl_server/` package is the active code. The `algo_envs/`, `libs/`, and `*_main*/` directories are pre-refactor snapshots. Always work in `rl_server/` — do not modify the legacy directories.

---
layout: default
title: Entrypoints
description: CLI entrypoints for all deployment modes
---

# Entrypoints

The `rl_server/entrypoints/` package provides the CLI entrypoints for all three deployment modes. Each file defines `parse_args()` and `main()` and is run via `python -m rl_server.entrypoints.<name>`.

## Package Layout

```
rl_server/entrypoints/
  train.py   — Local multiprocess: main + sampler + trainer + checker
  sample.py  — Redis-distributed sampler: standalone sample process
  trainer.py — Redis-distributed trainer: consumes experiences and publishes gradients
  grads.py   — Redis gradient aggregation server: standalone grads daemon
  check.py   — Redis-distributed checker: standalone evaluation
```

---

## train.py — Local Multiprocess Training

**The primary entrypoint.** Spawns all worker types as subprocesses and coordinates gradient aggregation in the main process.

### CLI

```bash
python -m rl_server.entrypoints.train \
  [--config <path>] \
  [--override <path>] \
  [--env-name <name>] \
  [--prefix <str>] \
  [--version <str>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/default.yaml` (auto-detected) | Config YAML path |
| `--override` | None | Override YAML (deep-merged) |
| `--env-name` | From config (`DQNGymClassic`) | Algorithm registry key |
| `--prefix` | `train_main_local` | Model file prefix |
| `--version` | None (latest) | Resume from specific version |

### Implementation Flow

```
main():
  1. parse_args()
  2. setup_mp()              → spawn start method, BLAS thread limit
  3. setup_seed()            → fixed seed 1970010101
  4. setup_signal_handlers() → SIGTERM/SIGINT handler
  5. load_config()           → read YAML, merge override, interpolate env vars
  6. set_device()            → set MODEL_CONFIG['DEVICE']
  7. create_net()            → build network
  8. share_memory() or skip  → CPU: share, GPU: skip (sync via state_dict)
  9. load_model()            → resume from checkpoint if available
  10. mp.Manager().dict()    → model_dict: shared state
  11. Start TrainerWorkers   → N processes
  12. Start SamplerWorkers   → M processes
  13. Start CheckerWorker    → 1 process (if enabled)
  14. Main loop:
        - Pull gradients from grads_queue
        - Accumulate N (num_update_grads) batches
        - update_state()
        - Increment TRAIN_VERSION
        - Write trainer metrics to TensorBoard
        - Periodic checkpoint every checkpoint_interval
        - Heartbeat monitor for trainers (30s timeout)
  15. On should_exit():
        - save_model()
        - set model_dict['is_exit'] = True
        - stop all workers
        - close TensorBoard writer
```

### Main Loop Detail

```python
while True:
    if should_exit():
        # Graceful shutdown sequence
        save_model(train_net, ...)
        model_dict['is_exit'] = True
        for s in samplers: s.stop()
        for t in trainers: t.stop()
        if train_checker: train_checker.stop()
        break

    if grads_count >= num_update_grads:
        # Apply accumulated gradients
        current_train_version += 1
        train_net.update_state(current_train_version, grads_buffer)
        model_dict['TRAIN_VERSION'] = current_train_version

        # TensorBoard metrics
        metrics = model_dict.get('trainer_metrics')
        if metrics:
            for key in ['policy_loss', 'value_loss', 'entropy', 'gradient_norm']:
                train_writer.add_scalar(f"train/{key}", ...)

        # Periodic checkpoint
        if current_train_version % checkpoint_interval == 0:
            save_model(...)

        grads_count = 0
        grads_buffer = None
    else:
        # Accumulate more gradients
        grads_info = grads_queue.get(block=False)
        # Element-wise addition
        for target_grad, grad in zip(grads_buffer, grads_item):
            np.add(target_grad, grad, out=target_grad)
```

### Gradient Accumulation

```python
grads_buffer = None
grads_count = 0

# Each grads_queue entry:
grads_info = {'grads': [param1_grad_np, param2_grad_np, ...]}

# Accumulation:
if grads_buffer is None:
    grads_buffer = grads_item
else:
    for target_grad, grad in zip(grads_buffer, grads_item):
        np.add(target_grad, grad, out=target_grad)
grads_count += 1

# When grads_count >= num_update_grads → apply and reset
```

### Heartbeat Monitor

Every 10 versions, checks if any trainer's heartbeat timestamp is >30 seconds stale:

```python
for i in range(num_trainers):
    hb = model_dict.get(f'trainer_heartbeat_{i}', 0)
    if time.time() - hb > 30:
        log.log_info(f"Trainer {i} heartbeat stale ({diff}s since last)")
```

### Configuration Mapping

| Config Key | Variable | Default |
|------------|----------|---------|
| `training.num_trainers` | `num_trainers` | 1 |
| `training.num_samplers` | `num_samplers` | 2 |
| `training.num_update_grads` | `num_update_grads` | 1 |
| `training.checkpoint_interval` | `checkpoint_interval` | 100 |
| `training.max_versions` | `max_versions` | 10000 |
| `training.enable_checker` | `enable_checker` | true |
| `training.device` | `device_str` | 'cpu' |
| `queues.len_grads_queue` | grads_queue maxsize | 1000 |
| `queues.len_sample_queue` | sample_queue maxsize | 1000 |

---

## sample.py — Redis-Distributed Sampler

**Purpose**: Standalone sampling process that syncs model weights from Redis and pushes experiences to Redis. No shared memory with other processes.

### CLI

```bash
python -m rl_server.entrypoints.sample \
  [--config <path>] \
  [--override <path>] \
  [--env-name <name>]
```

### Implementation Flow

```
main():
  1. Setup seed + signal handlers
  2. Load config
  3. Create two Redis connections:
     - model_redis: sync model weights (db 0)
     - exps_redis: push experiences (db 1)
  4. create_net() + create_agent()
  5. Loop:
        - Check exit flag from Redis
        - Poll TRAIN_VERSION
        - If newer: get_train_model() to sync weights
        - sample_multi_envs()
        - push_exps() for each trajectory
```

### Redis Interaction

```python
# Model sync (model_redis, db 0)
version = model_redis.get_train_version()
if version != local_version:
    model_redis.get_train_model(sample_net)
    local_version = version

# Experience push (exps_redis, db 1)
for exps in exps_list:
    exps_redis.push_exps(exps, sample_version=local_version)
```

---

## grads.py — Gradient Aggregation Server

**Purpose**: Standalone daemon that consumes gradients from Redis, aggregates them, and broadcasts the updated model.

### CLI

```bash
python -m rl_server.entrypoints.grads \
  [--config <path>] \
  [--override <path>] \
  [--env-name <name>] \
  [--prefix <str>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prefix` | `grads_main` | Model file prefix |

### Implementation Flow

```python
main():
  1. Setup seed + signal handlers
  2. Load config
  3. Create GradsAggregator with two Redis configs:
     - grads_redis: consume gradients (db 2)
     - model_redis: broadcast model (db 0)
  4. aggregator.run()
```

See [Workers → GradsAggregator](workers.md#gradsaggregator) for the main loop logic.

### Redis Configuration

| Service | Config Key | Default DB | Purpose |
|---------|------------|------------|---------|
| Model | `redis.model` | 0 | Broadcast model weights |
| Grads | `redis.grads` | 2 | Consume gradient batches |

---

## check.py — Standalone Checker/Evaluation

**Purpose**: Load a checkpoint (optional) and run evaluation episodes, writing results to TensorBoard. Works standalone or alongside running training.

### CLI

```bash
python -m rl_server.entrypoints.check \
  [--config <path>] \
  [--override <path>] \
  [--env-name <name>] \
  [--prefix <str>] \
  [--version <str>] \
  [--episodes <int>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prefix` | `checker` | Model file prefix for `load_model` |
| `--version` | None (latest) | Specific checkpoint version |
| `--episodes` | 0 (continuous) | Number of episodes to evaluate |

### Implementation Flow

```
main():
  1. Setup seed + signal handlers
  2. Load config
  3. create_net() + create_agent(is_checker=True)
  4. If --version: load_model()
  5. check_net.train(False)  # eval mode
  6. Loop:
        - check_single_env()
        - Write metrics to TensorBoard
        - If episodes > 0 and reached: break
        - Check should_exit()
  7. writer.close()
```

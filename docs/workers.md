---
layout: default
title: Workers
description: Sampler, Trainer, Checker, and Gradient Aggregation worker processes
---

# Workers

The `rl_server/workers/` package contains the four worker types that execute in subprocesses during training. Each worker is a class that wraps an `mp.Process` with `run()` / `stop()` lifecycle.

## Package Layout

```
rl_server/workers/
  sampler.py          — SamplerWorker: rollout environments, push experiences
  trainer.py          — TrainerWorker: consume experiences, compute gradients
  checker.py          — CheckerWorker: evaluate policy, write TensorBoard metrics
  grads_aggregator.py — GradsAggregator: consume gradients from Redis, update model
```

---

## SamplerWorker

**Purpose**: Runs N parallel environments, collects experience trajectories, and pushes them to the sample queue (local) or Redis (distributed).

### Class

```python
SamplerWorker(idx, model_dict, share_model, sample_queue, env_name, log)
```

### Lifecycle

```
run() → mp.Process(target=process_function) → ...
  loop:
    check model_dict['is_exit']
    sync: load_state_dict(share_model.state_dict()) if version changed
    sample: agent.sample_multi_envs(model_dict)
    push: for each trajectory, sample_queue.put({'exps': ..., 'sample_version': ...})
  cleanup: del sample_agent
```

### Key Details

- **Version-triggered sync**: Only calls `load_state_dict` when `TRAIN_VERSION` changes (checked each iteration). This avoids the overhead of copying weights every loop iteration.
- **Blocking put**: Uses `queue.put()` (blocks if full). On `queue.Full` exception, sleeps 5 seconds and retries.
- **Empty rollout handling**: If `sample_multi_envs` returns `None`, logs a message and continues.
- **Seed**: Each sampler calls `setup_seed()` in its subprocess (deterministic per-worker seed from fixed global seed).
- **Exit**: Checks `model_dict['is_exit']` at the top of each loop iteration.

### Error Handling

```python
except queue.Full:
    time.sleep(5)
    continue
except Exception:
    self.log.log_exception()
    continue
```

All non-queue exceptions are logged (with dedup) and the worker continues — transient failures don't crash the sampler.

---

## TrainerWorker

**Purpose**: Consumes experience batches from the sample queue, computes gradients via `AlgoBaseCalculate.generate_grads()`, and pushes gradients to the grads queue.

### Class

```python
TrainerWorker(idx, model_dict, share_model, sample_queue, grads_queue, env_name, log)
```

### Lifecycle

```
run() → mp.Process(target=process_function) → ...
  init: heartbeat_key = f'trainer_heartbeat_{id}'
  loop:
    check model_dict['is_exit']
    samples = sample_queue.get()          # blocking
    grads_list, version = calculate.generate_grads(samples['exps'], model_dict)
    for grads in grads_list:
        grads_queue.put({'grads': ..., 'grads_version': ..., 'sample_version': ...})
    update heartbeat timestamp
```

### Key Details

- **Blocking get**: `sample_queue.get()` blocks indefinitely until a sample is available. No busy-waiting.
- **Heartbeat**: Writes `time.time()` to `model_dict[f'trainer_heartbeat_{id}']` after each successful gradient computation. Main process monitors this for stale workers (30s threshold).
- **Gradient push**: Iterates over `grads_list` (algorithms usually return `[grads]` — a single-element list). Each gradient batch gets its own queue entry.
- **Seed**: Same `setup_seed()` pattern as sampler.
- **Error Handling**: Catches `queue.Full`, `queue.Empty`, and generic exceptions — never crashes.

---

## CheckerWorker

**Purpose**: Periodically evaluates the current policy on a single environment and writes metrics to TensorBoard. Runs in its own subprocess.

### Class

```python
CheckerWorker(model_dict, share_model, env_name, log)
```

### Lifecycle

```
run(comment='_algo_env_mixed') → mp.Process(target=process_function) → ...
  init: create_net(env_name), create_agent(env_name, net, is_checker=True)
  init: SummaryWriter(comment=...)
  loop:
    check model_dict['is_exit']
    sync: check_net.load_state_dict(share_model.state_dict())
    check_net.train(False)     # eval mode
    infos = agent.check_single_env()
    for key, value in infos:
        writer.add_scalar(key, value, self_version)
    writer.flush()              # critical — see gotcha below
```

### Key Details

- **`is_checker=True`**: Agent creates a single env (not N parallel), runs one episode until `done`, returns dict of scalar metrics.
- **Version tracking**: Uses `model_dict['TRAIN_VERSION']` as the x-axis for TensorBoard scalars.
- **`writer.flush()`**: Explicit flush after each write. The default `SummaryWriter` buffers writes for ~120 seconds, which means the last evaluation episode might never flush. This was a fixed bug.
- **Evaluation metrics**: Typically `sum_rewards`, `average_mus`, `sum_entropys`. Algorithm-specific.
- **Seeds**: Same `setup_seed()` for deterministic evaluation.

### TensorBoard Scalars

| Metric | Source | Description |
|--------|--------|-------------|
| `sum_rewards` | All algorithms | Total episode return |
| `average_mus` | All algorithms | Mean action value |
| `sum_entropys` | PPO, SAC, TD3 | Total policy entropy |

---

## GradsAggregator

**Purpose**: Standalone daemon running in its own process (or as the main process in grads-server mode). Consumes gradients from Redis, aggregates them, updates the model, and broadcasts the new weights back to Redis.

### Class

```python
GradsAggregator(env_name, model_prefix, grads_redis_config, model_redis_config,
                batch_update_count=10, log=None)
```

### Lifecycle

```
__init__:
    create_net(env_name)
    load_model(grads_net, f"{prefix}_{env_name}")   # Resume from checkpoint
    grads_redis = RedisCache(...)
    model_redis = RedisCache(...)
    grads_redis.clear_data()           # Start fresh
    model_redis.clear_data()
    model_redis.set_train_version_model(version, grads_net)

run():
    loop:
        check should_exit()
        if grads_count >= batch_update_count:
            version += 1
            grads_net.update_state(version, grads_buffer)
            model_redis.set_train_version_model(version, grads_net)
            reset buffer
        else:
            grads_item, gv, sv = grads_redis.pop_grads()
            accumulate into grads_buffer
```

### Key Details

- **Dual Redis**: One connection for consuming gradients (`grads_redis`), another for broadcasting the model (`model_redis`). These can be different Redis instances.
- **`clear_data()`**: Clears all gradient and model keys at startup. **This means restarting the grads server discards any in-flight gradients.**
- **`batch_update_count`**: From config `training.batch_update_grads_server` (default 10). Controls how many gradient batches to accumulate before each update step.
- **Checkpoint on shutdown**: Saves current model before setting the exit flag in Redis, which signals remote workers to stop.

### Gradient Accumulation

Same pattern as local mode: element-wise summation of gradient arrays from multiple trainer workers:

```python
if grads_buffer is None:
    grads_buffer = grads_item
else:
    for target_grad, grad in zip(grads_buffer, grads_item):
        target_grad += grad
```

---

## Worker Comparison

| Aspect | Sampler | Trainer | Checker | GradsAggregator |
|--------|---------|---------|---------|-----------------|
| **Input** | env actions, model weights | sample_queue / Redis exps | model weights | grads Redis |
| **Output** | sample_queue / Redis exps | grads_queue / Redis grads | TensorBoard scalars | model Redis |
| **Blocking** | queue.put (non-blocking try) | queue.get (blocking) | N/A | Redis BRPOP (5s timeout) |
| **GPU** | CPU only | Optionally GPU | CPU only | CPU |
| **Heartbeat** | ❌ | ✅ `trainer_heartbeat_{id}` | ❌ | ❌ |
| **Exit signal** | `model_dict['is_exit']` | `model_dict['is_exit']` | `model_dict['is_exit']` | `should_exit()` + Redis exit flag |

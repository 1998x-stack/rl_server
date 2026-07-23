---
layout: default
title: Transport & Redis
description: Redis-backed communication layer for distributed training
---

# Transport & Redis

The `rl_server/transport/` package implements the Redis-backed communication layer for distributed training. In local mode, `mp.Queue` handles IPC directly; in distributed mode, all inter-process communication flows through Redis.

## Package Layout

```
rl_server/transport/
  redis_cache.py     — RedisCache class: model sync, experience/gradient queues, exit flag
  serialization.py   — pickle+zlib for objects, np.savez_compressed for gradients
```

---

## serialization.py — Efficient Serialization

Two serialization paths, chosen based on data type:

### `serialize(obj)` / `deserialize(data)`

General-purpose `pickle + zlib` roundtrip. Used for model state_dicts and experience dictionaries.

```
bytes = pickle.dumps(obj, protocol=HIGHEST)  →  zlib.compress(bytes)
obj   = pickle.loads(zlib.decompress(data))
```

### `serialize_gradients(grads_list)` / `deserialize_gradients(data)`

Specialized for gradient arrays (list of numpy ndarrays). Uses `np.savez_compressed` which is ~30% smaller than pickle+zlib for homogeneous float arrays.

```
bytes = np.savez_compressed(BytesIO, *grads_list).getvalue()
grads = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
```

---

## redis_cache.py — RedisCache

### Redis Key Space

| Redis Key | Type | Purpose |
|-----------|------|---------|
| `TRAIN_VERSION` | `string` (int) | Current global training step |
| `train_model` | `string` (pickle+zlib) | Compressed model state_dict |
| `sample_version` | `string` (int) | Sampler's last synced version |
| `sample_model` | `string` (pickle+zlib) | Sampler-side model copy |
| `exps` | `list` | Experience queue (LPUSH/BRPOP) |
| `grads` | `list` | Gradient queue (LPUSH/BRPOP) |
| `exit` | `string` (int 0/1) | Global shutdown flag |

### RedisCache Class

```python
cache = RedisCache(log, redis_config)
```

**Constructor** (`__init__`):

| Parameter | Description |
|-----------|-------------|
| `log` | `Log` instance for structured logging |
| `redis_config` | Dict with `ip`, `port`, `db`, `pw`, optional `pool_size` |

Establishes a `redis.ConnectionPool` (default max_connections=10) and performs a `PING` health check. Raises `ConnectionError` if Redis is unreachable.

**Destructor** (`__del__`): closes the connection, swallowing exceptions.

### Model Sync Methods

| Method | Description | Retry |
|--------|-------------|-------|
| `set_train_version_model(version, model)` | Serialize `model.state_dict()`, compress, write `train_model` + `TRAIN_VERSION` | ✅ |
| `get_train_version()` | Read and parse `TRAIN_VERSION` as int | ❌ |
| `get_train_model(model)` | Read `train_model`, decompress, `model.load_state_dict()` | ❌ |

`set_train_version_model` is called by the gradient aggregation server or local main process after each update. Remote workers call `get_train_model` to sync.

### Experience Queue Methods

| Method | Description |
|--------|-------------|
| `push_exps(exps, sample_version)` | Serialize `{exps, sample_version}`, `LPUSH` to `exps` list | ✅ |
| `pop_exps()` | `BRPOP exps timeout=5` → `(exps, sample_version)` or `(None, None)` | ❌ (read) |

Timeouts are tracked: every 10th consecutive timeout is logged. This prevents log spam during idle periods.

### Gradient Queue Methods

| Method | Description |
|--------|-------------|
| `push_grads(grads, grads_version, sample_version)` | Serialize with `serialize_gradients`, `LPUSH` to `grads` list | ✅ |
| `pop_grads()` | `BRPOP grads timeout=5` → `(grads, grads_version, sample_version)` or all None | ❌ (read) |

Same timeout tracking as `pop_exps`.

### Shutdown & Management

| Method | Description |
|--------|-------------|
| `set_exit_flag(bool)` | Write `0` or `1` to `exit` key (with error logging) |
| `get_exit_flag()` | Read and parse `exit` key → `int` or `None` |
| `health_check()` | `PING` → `True`/`False` |
| `clear_data()` | `FLUSHALL` — deletes everything. **Handle with care.** |
| `_reconnect()` | Close old connection, create new single-client connection (no pool) |

### Retry Decorator

```python
@_retry(max_retries=3, base_delay=1.0)
```

Applied to write operations (`set_train_version_model`, `push_exps`, `push_grads`). Behavior:

| Attempt | Delay | Action |
|---------|-------|--------|
| 1 | 0 | Try immediately |
| 2 | 1s | Retry after 1s |
| 3 | 2s | Retry after 2s |
| (fail) | — | Log exception and re-raise |

On `ConnectionError`, the decorator calls `self._reconnect()` and tries `self.conn.ping()` before retrying.

---

## Three-Queue Architecture (Distributed Mode)

```
┌──────────────┐     LPUSH         BRPOP     ┌──────────────┐
│              │ ──────────────> <────────── │              │
│   Sampler    │    exps queue                │   Trainer    │
│   (N proc)   │                              │   (M proc)   │
│              │                              │              │
└──────────────┘                              └──────┬───────┘
       │                                             │ LPUSH
       │ BRPOP (model)                               │
       ▼                                             ▼
┌──────────────┐                              ┌──────────────┐
│  Model Redis  │ ◄──── set_train_version ────│  Grads       │
│  (db 0)       │                              │  Aggregator  │
│  - TRAIN_VERSION │                           │  (1 proc)    │
│  - train_model   │                           └──────────────┘
└──────────────┘
```

Each queue uses a separate Redis database (db 0 = model, db 1 = exps, db 2 = grads) but can share the same Redis instance.

---

## Configuration Reference

```yaml
redis:
  model:
    host: "${REDIS_MODEL_HOST:localhost}"  # Model + versions
    port: 6379
    db: 0
    password: "${REDIS_PASSWORD:}"
  exps:
    host: "${REDIS_EXPS_HOST:localhost}"   # Experience queue
    port: 6379
    db: 1
    password: "${REDIS_PASSWORD:}"
  grads:
    host: "${REDIS_GRADS_HOST:localhost}"  # Gradient queue
    port: 6379
    db: 2
    password: "${REDIS_PASSWORD:}"
```

Each of the three queues can point to:
- **Same Redis instance** (dev/small-scale) — just different db indices
- **Separate Redis instances** (production) — separate hosts for model/exps/grads
- **Redis Cluster / Sentinel** — via any Redis-compatible proxy

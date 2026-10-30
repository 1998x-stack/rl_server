---
layout: default
title: Utilities
description: Checkpointing, structured logging, and process management
---

# Utilities

The `rl_server/utils/` package provides cross-cutting infrastructure: checkpoint persistence, structured logging with deduplication, and process lifecycle management.

## Package Layout

```
rl_server/utils/
  checkpoint.py  — save_model(), load_model(), get_model_state_path()
  logging.py     — Log class with rotating files, exception deduplication
  process.py     — setup_mp(), setup_seed(), signal handlers, heartbeat files
```

---

## checkpoint.py — Atomic Model Persistence

### File Format

Checkpoints are saved as `.td` (Torch Dump) files:

```
models/{prefix}/
  {prefix}_{version}_{timestamp}.td
```

Each `.td` file is a `torch.save` dict:

```python
{
    'state_dict': model.state_dict(),    # Always CPU tensors
    'version': str(version),             # Stored as string in file
    'timestamp': time.time(),            # Unix timestamp
}
```

Atomic write: `torch.save` → `.tmp` file → `os.rename()` → `.td`. If the process crashes mid-save, the `.tmp` is discarded and the last good `.td` remains intact.

### Functions

#### `save_model(model, prefix, version, base_dir='models', max_versions=5)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | required | `nn.Module` to save |
| `prefix` | required | Model name (also used as subdirectory name) |
| `version` | required | String version number |
| `base_dir` | `'models'` | Root model directory |
| `max_versions` | `5` | Max `.td` files to retain (0 = unlimited) |

Returns saved file path on success, `None` on failure.

**Version retention**: After save, sorts existing `.td` files by mtime descending and deletes files beyond `max_versions`. This keeps disk usage bounded.

#### `load_model(model, prefix, version=None, base_dir='models', map_location=None)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | required | `nn.Module` to load weights into |
| `prefix` | required | Same prefix used during save |
| `version` | `None` | Specific version or `None` for latest |
| `base_dir` | `'models'` | Root model directory |
| `map_location` | `None` | Device mapping for `torch.load` |

Returns the checkpoint version as int, or `None` if no checkpoint found or load failed.

**Version type note**: Version is stored as string in the file. The main loop casts it to int after loading (`current_train_version = int(current_train_version)`).

**Error handling**: Catches `IOError`, `RuntimeError`, `ValueError`, `pickle.UnpicklingError` and returns `None` (logged via print).

#### `get_model_state_path(prefix, version=None, base_dir='models')`

Finds the latest `.td` file matching the prefix (and optional version) by modification time. Used internally by `load_model`.

---

## logging.py — Structured Logging

### `Log` Class

```python
log = Log('train_main_local')            # 或 Log.from_config(config, 'train_main_local')
log.info('training started')
log.info_throttled('queue_empty', 'queue empty')   # 60s 内只输出一次，其余计数
log.info_every('progress', 10, 'version=10')       # 每 10 次输出一次
log.exception('train main loop')                   # 带上下文 + 折叠重复 traceback
log.summary({'version': 100})                      # 退出前汇总折叠统计
```

#### Features

| Feature | Detail |
|---------|--------|
| **Rotating file handler** | 50 MB per file, 3 backups, UTF-8, `logs/{dir_name}/{dir_name}.log` |
| **Per-process files** | 子进程写 `logs/{dir_name}/{dir_name}.{pid}.log`，避免多进程同时轮转同一个文件 |
| **Console output** | Same format as file, level from `logging.level` |
| **Format** | `[2026-01-01 12:00:00.123] [INFO] [train_main_local:MainProcess:12345] message` |
| **Spawn-safe** | `Log` is pickled into workers and rebuilds its handlers there (worker logs used to vanish) |
| **Throttling** | `info_throttled` / `warning_throttled` / `info_every` / `once` |
| **Legacy compat** | `log_info()` / `log_exception()` aliases for old code |

#### Levels

`logger` is always at DEBUG; the **file handler** records DEBUG, the **console handler** honours
`logging.level`. That is what makes "file always DEBUG" actually true — before this, the logger's
own level filtered DEBUG out of the file as well.

#### Exception Deduplication

Repeated exceptions in tight loops must not bury the log, but must not disappear either.

| Condition | Behavior |
|-----------|----------|
| First occurrence | Full traceback + `type=`, `origin=`, `value=` |
| Repeat | Counted, not printed |
| Every `exc_report_secs` (default 60s) | One line: `repeated=N suppressed=M first=HH:MM:SS last=HH:MM:SS` |
| `log.summary()` at exit | One line per distinct exception with total `count=` |

Dedup key: `(exception type, deepest stack frame, digit-normalized message)`. The old key
`f"{type}:{value}"` never folded messages that embed ids/counters, and merged unrelated call sites
that happened to share text.

#### `setup_logging(dir_name, level='INFO', max_bytes=52428800, backup_count=3, base_dir=None, pid_suffix='')`

Lower-level function that creates a `logging.Logger` with `RotatingFileHandler` and `StreamHandler`.
Returns the `Logger` object. Called by `Log.__init__()`.

---

## process.py — Process Lifecycle

### `setup_mp()`

Configures the multiprocessing environment for CUDA compatibility:

```python
mp.set_start_method('spawn', force=True)  # Required for CUDA
os.environ["OMP_NUM_THREADS"] = "1"       # Prevent BLAS over-subscription
os.environ["MKL_NUM_THREADS"] = "1"
```

Called at the very start of `train.py`'s `main()`.

### `setup_seed(seed=1970010101)`

Fixes all random seeds for reproducibility:

| RNG | Seed | Notes |
|-----|------|-------|
| Python `random` | ✅ | |
| NumPy | ✅ | |
| PyTorch (CPU) | ✅ | |
| PyTorch (CUDA) | ✅ | `manual_seed_all` |
| cuDNN | ✅ | `deterministic=True`, `benchmark=False` |
| `PYTHONHASHSEED` | ✅ | Set via `os.environ` |

### Signal Handlers

```python
setup_signal_handlers()  # SIGTERM + SIGINT → _shutdown_event.set()
should_exit()            # Returns bool
request_shutdown()       # Manual trigger for tests
```

Threading-based (`threading.Event`), safe for use in both main process and worker processes. Checked in main loops:

```python
if should_exit():
    save_model(...)
    model_dict['is_exit'] = True
    break
```

### Heartbeat Files

```python
HEARTBEAT_DIR = '/tmp/rl_server/'   # Platform temp dir

write_heartbeat(worker_type, worker_id)
    # Creates: /tmp/rl_server/{type}_{id}_{pid}
    # Content: current timestamp

cleanup_heartbeat(worker_type, worker_id)
    # Removes the file
```

Used for external monitoring — a watchdog can check if heartbeat files have been updated recently to detect hung or dead workers.

### Worker Loop Conventions

All workers follow the same exit pattern:

1. Check `model_dict['is_exit']` at the top of each iteration
2. On exit, attempt cleanup (`del agent`, close resources)
3. Log exit message with PID and worker ID

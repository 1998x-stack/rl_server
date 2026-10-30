---
layout: default
title: Config System
description: YAML configuration loading, environment variable interpolation, and schema validation
---

# Config System

The `rl_server/config/` package provides a YAML-based configuration system with environment variable interpolation, override merging, and optional schema validation.

## Package Layout

```
rl_server/config/
  loader.py     — load_config(), _deep_merge(), interpolate_env_vars()
  schema.py     — validate_config() with REQUIRED_KEYS
  default.yaml  — Production defaults (all keys)
  dev.yaml      — Development overrides (subset)
```

---

## loader.py — Config Loading

### Architecture

```
load_config(base_path, override_path=None)
     │
     ├── Read base YAML ─────────────────────────── yaml.safe_load()
     │
     ├── If override_path exists ────────────────── _deep_merge(base, override)
     │
     └── _interpolate_recursive(result) ────────── Replace ${VAR:default}
                                                      in all string leaves
```

### `load_config(base_path, override_path=None)`

Main entry point. Loads a YAML file, optionally merges an override, and interpolates environment variables.

```python
from rl_server.config.loader import load_config

# Single config
config = load_config('config/default.yaml')

# With override (dev.yaml merges on top — only specify changed keys)
config = load_config('config/default.yaml', override='config/dev.yaml')

# path discovery in entrypoints:
config_path = args.config or os.path.join(
    os.path.dirname(__file__), '..', 'config', 'default.yaml'
)
```

Raises `FileNotFoundError` if `base_path` doesn't exist.

### `_deep_merge(base, override)`

Recursive dictionary merge. Rules:

- If both base and override have a dict at the same key → recurse
- Otherwise → override value replaces base value
- Returns a new dict (base is not mutated)

```python
base = {'a': 1, 'b': {'x': 10, 'y': 20}}
override = {'b': {'y': 99, 'z': 30}, 'c': 3}
result = _deep_merge(base, override)
# {'a': 1, 'b': {'x': 10, 'y': 99, 'z': 30}, 'c': 3}
```

### `interpolate_env_vars(value)` and `_interpolate_recursive(obj)`

Replace `${VAR}` and `${VAR:default}` patterns with environment variable values.

```
Pattern         Behavior
${REDIS_HOST}      → os.environ['REDIS_HOST'] or ''
${REDIS_HOST:localhost}  → os.environ.get('REDIS_HOST', 'localhost')
```

Recursive walk: strings → interpolate, dicts → recurse values, lists → recurse items, everything else → pass through.

---

## schema.py — Config Validation

### `validate_config(config)`

Performs pre-flight validation against `REQUIRED_KEYS`:

```python
REQUIRED_KEYS = {
    'redis': {'model', 'exps'},
    'training': {'env_name', 'num_samplers', 'num_trainers'},
}
```

Raises `ValueError` if a section or required key is missing.

> **Note:** `validate_config()` is not yet wired into any entrypoint. It exists as a utility for future stricter startup validation.

---

## default.yaml — Production Config

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `redis.model` | host | `${REDIS_MODEL_HOST:localhost}` | Model broadcast Redis |
| | port | 6379 | |
| | db | 0 | |
| | password | `${REDIS_PASSWORD:}` | Empty string if unset |
| `redis.exps` | host | `${REDIS_EXPS_HOST:localhost}` | Experience queue Redis |
| | port | 6379 | |
| | db | 1 | Separated from model by DB index |
| | password | `${REDIS_PASSWORD:}` | Same password as model |
| `redis.grads` | host | `${REDIS_GRADS_HOST:localhost}` | Gradient queue Redis |
| | port | 6379 | |
| | db | 2 | |
| | password | `${REDIS_PASSWORD:}` | |
| `training` | env_name | `DQNGymClassic` | Algorithm registry key |
| | device | `cpu` | `cpu` or `cuda:0`, etc. |
| | num_samplers | 2 | Sampler worker count |
| | num_trainers | 1 | Trainer worker count |
| | num_update_grads | 1 | Grads accumulated before update |
| | batch_update_grads_server | 10 | Grads batch for grads server mode |
| | max_gradient_lag | auto | 允许的梯度版本滞后；`auto` 按实测到达滞后自适应，整数则固定；0 = 严格同步（会丢掉在途批次） |
| | max_gradient_lag_cap | 64 | `auto` 模式下的滞后上限 |
| | version_log_interval | 10 | 每 N 个版本输出一行进度 |
| | check_log_interval | 100 | checker 每 N 次评估输出一条进度 |
| | cold_start_versions | 20 | 启动前 N 个版本的梯度拒绝单独计为 `cold_start_rejected` |
| | stall_warn_secs | 60 | 无梯度被接受多久后告警 |
| | stall_fatal_secs | 300 | 无梯度被接受多久后中止（退出码 1） |
| | enable_checker | true | Whether to run eval worker |
| | checkpoint_interval | 100 | Save every N versions |
| | max_versions | 10000 | Max checkpoint files retained |
| `queues` | len_grads_queue | 1000 | Maxsize for gradient mp.Queue |
| | len_sample_queue | 1000 | Maxsize for sample mp.Queue |
| `logging` | level | INFO | Console log level (file always records DEBUG) |
| | dir | logs | Log root directory |
| | throttle_secs | 60 | 循环内重复消息的最小输出间隔 |
| | tensorboard | true | Enable TensorBoard writer |

## dev.yaml — Development Override

Minimal override that only specifies dev-different values:

```yaml
redis:
  model:
    host: "localhost"     # Fixed, no env var
  exps:
    host: "localhost"
  grads:
    host: "localhost"

training:
  env_name: "DQNGymClassic"
  num_samplers: 1         # Fewer workers in dev
  num_trainers: 1

logging:
  level: "DEBUG"          # More verbose
```

Usage: `python -m rl_server.entrypoints.train --override rl_server/config/dev.yaml`

---

## Config Flow Per Entrypoint

All entrypoints follow the same pattern:

```python
config_path = args.config or os.path.join(
    os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml'
)
config = load_config(config_path, args.override)

# Extract sections
redis_cfg = config.get('redis', {})
training_cfg = config.get('training', {})
queues_cfg = config.get('queues', {})

# Override env_name from CLI if provided
env_name = args.env_name or training_cfg.get('env_name', 'DQNGymClassic')
```

The `--env-name` CLI flag takes precedence over config file — this is how `python -m rl_server.entrypoints.train --env-name MujocoNormal` works without touching YAML.

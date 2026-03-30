# RL Server Production Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the distributed RL training system into a production-ready, well-tested, cleanly structured Python package.

**Architecture:** Hybrid incremental approach — fix bugs first, add tests to lock behavior, then restructure into a clean `rl_server/` package with YAML config, proper error handling, gymnasium migration, and production monitoring. Each stage is independently committable.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, gymnasium 0.29+, Redis 5+, PyYAML, pytest, fakeredis, tensorboardX

---

## File Structure

### Stage 1-3: In-place fixes (modify existing files)

- Modify: `grads_main/grads_main.py` — fix undefined variable
- Modify: `train_main_redis/trainer_redis.py` — add missing import
- Modify: `libs/redis_cache.py` — add brpop timeout, retry logic
- Modify: `algo_envs/ppo_microrts.py` — fix environment leak
- Modify: `libs/config.py` — remove hardcoded credentials, use YAML loader
- Modify: `libs/log.py` — replace with Python logging
- Modify: `libs/utils.py` — replace file-based exit with signal handling
- Create: `config/default.yaml` — default configuration
- Create: `config/dev.yaml` — dev overrides
- Create: `libs/config_loader.py` — YAML loading with env var interpolation

### Stage 4: Tests (new files against current structure)

- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_actions.py`
- Create: `tests/unit/test_exps.py`
- Create: `tests/unit/test_redis_cache.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_utils.py`
- Create: `tests/unit/test_algo_base.py`
- Create: `tests/unit/test_dqn.py`
- Create: `tests/unit/test_ppo_mujoco.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_local_pipeline.py`
- Create: `tests/integration/test_checkpoint.py`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`

### Stage 5: Package restructure

- Create: `rl_server/__init__.py`
- Create: `rl_server/core/__init__.py`, `base.py`, `actions.py`, `buffers.py`, `noisy.py`
- Create: `rl_server/algorithms/__init__.py`
- Create: `rl_server/algorithms/dqn/__init__.py`, `network.py`, `agent.py`, `calculator.py`
- Create: `rl_server/algorithms/ppo/__init__.py`, `mujoco_normal.py`, `microrts.py`, `mujoco_beta.py`, `mobile.py`
- Create: `rl_server/algorithms/sac/__init__.py`, `mujoco_normal.py`
- Create: `rl_server/algorithms/td3/__init__.py`, `mujoco_normal.py`
- Create: `rl_server/workers/__init__.py`, `sampler.py`, `trainer.py`, `checker.py`, `grads_aggregator.py`
- Create: `rl_server/transport/__init__.py`, `redis_cache.py`, `serialization.py`
- Create: `rl_server/config/__init__.py`, `loader.py`, `schema.py`, `default.yaml`
- Create: `rl_server/entrypoints/__init__.py`, `train.py`, `sample.py`, `check.py`, `grads.py`
- Create: `rl_server/utils/__init__.py`, `checkpoint.py`, `logging.py`, `process.py`

### Stage 6: Gymnasium migration

- Modify: All algorithm files to use `gymnasium` instead of `gym`

### Stage 7: Production hardening

- Modify: `rl_server/utils/checkpoint.py` — atomic writes, retention policy
- Modify: `rl_server/utils/process.py` — signal handling, heartbeats
- Modify: `rl_server/transport/redis_cache.py` — connection pool, health checks
- Update: `requirements.txt`

### Stage 8: Updated tests

- Update: All test files to use new `rl_server.*` import paths
- Create: `tests/integration/test_redis_pipeline.py`
- Create: `tests/integration/test_grad_aggregation.py`

---

## Stage 1: Fix Critical Bugs

### Task 1: Fix undefined variable in grads_main.py

**Files:**
- Modify: `grads_main/grads_main.py:88-93`

- [ ] **Step 1: Fix the undefined variable bug**

In `grads_main/grads_main.py`, lines 91 and 93 reference `current_grads_version` which is never defined. The correct variable is `current_train_version`. Also line 92 passes `current_train_version` to `update_state` but then line 93 sets the model with `current_grads_version` — both should use the same incremented version.

Replace lines 88-96 in `grads_main/grads_main.py`:

```python
            if grads_count >= queue_config['batch_update_grads_server']:

                # 更新版本
                current_train_version = current_train_version + 1
                grads_net.update_state(current_train_version, grads_buffer)
                model_redis_cache.set_train_version_model(current_train_version, grads_net)

                grads_buffer = None
                grads_count = 0
```

The change: `current_grads_version` -> `current_train_version` on lines 91 and 93.

- [ ] **Step 2: Verify the fix is syntactically correct**

Run: `python -c "import ast; ast.parse(open('grads_main/grads_main.py').read()); print('OK')"`
Expected: `OK`

### Task 2: Add missing import in trainer_redis.py

**Files:**
- Modify: `train_main_redis/trainer_redis.py:1-23`

- [ ] **Step 1: Add the missing redis_cache import**

In `train_main_redis/trainer_redis.py`, add `import libs.redis_cache as redis_cache` after the existing imports. The file uses `redis_cache.RedisCache()` on line 61 but never imports the module.

Add after line 22 (`import libs.config as config`):

```python
import libs.redis_cache as redis_cache
```

- [ ] **Step 2: Verify the fix**

Run: `python -c "import ast; ast.parse(open('train_main_redis/trainer_redis.py').read()); print('OK')"`
Expected: `OK`

### Task 3: Add timeout to Redis brpop operations

**Files:**
- Modify: `libs/redis_cache.py:145-155` and `libs/redis_cache.py:176-186`

- [ ] **Step 1: Add timeout to pop_exps**

In `libs/redis_cache.py`, modify the `pop_exps` method to use a 5-second timeout on `brpop`:

```python
    # 获取采样经验
    def pop_exps(self):
        try:
            # 返回为  Tuple(key,value) key 为 RedisCache.exps_name
            # 调用为阻塞模式，5秒超时
            exps_info = self.conn.brpop(RedisCache.exps_name, timeout=5)
            if exps_info is None:
                return None, None
            exps_info = zlib.decompress(exps_info[1])
            exps_info = pickle.loads(exps_info)
            return exps_info['exps'], exps_info['sample_version']
        except Exception:
            self.log.log_exception()
            return None, None
```

- [ ] **Step 2: Add timeout to pop_grads**

In `libs/redis_cache.py`, modify the `pop_grads` method similarly:

```python
    # 获取梯度信息
    def pop_grads(self):
        try:
            # 返回为  Tuple(key,value) key 为 RedisCache.grads_name
            # 调用为阻塞模式，5秒超时
            grads_info = self.conn.brpop(RedisCache.grads_name, timeout=5)
            if grads_info is None:
                return None, None, None
            grads_info = zlib.decompress(grads_info[1])
            grads_info = pickle.loads(grads_info)
            return grads_info['grads'], grads_info['grads_version'], grads_info['sample_version']
        except Exception:
            self.log.log_exception()
            return None, None, None
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('libs/redis_cache.py').read()); print('OK')"`
Expected: `OK`

### Task 4: Fix environment memory leak in ppo_microrts.py

**Files:**
- Modify: `algo_envs/ppo_microrts.py:251-253`

- [ ] **Step 1: Implement proper cleanup in __del__**

In `algo_envs/ppo_microrts.py`, replace the commented-out `__del__` method:

```python
    def __del__(self):
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except Exception:
            pass
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('algo_envs/ppo_microrts.py').read()); print('OK')"`
Expected: `OK`

### Task 5: Commit Stage 1

- [ ] **Step 1: Stage and commit all bug fixes**

```bash
git add grads_main/grads_main.py train_main_redis/trainer_redis.py libs/redis_cache.py algo_envs/ppo_microrts.py
git commit -m "fix: resolve critical bugs in grads_main, trainer_redis, redis_cache, ppo_microrts

- Fix undefined variable current_grads_version in grads_main.py (use current_train_version)
- Add missing import libs.redis_cache in trainer_redis.py
- Add 5-second timeout to brpop in redis_cache.py pop_exps/pop_grads
- Fix environment memory leak in ppo_microrts.py __del__ method"
```

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Stage 2: YAML Configuration System

### Task 6: Create the YAML config loader

**Files:**
- Create: `libs/config_loader.py`

- [ ] **Step 1: Write tests for config loader**

Create `tests/unit/test_config_loader.py`:

```python
import os
import sys
import pytest
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from libs.config_loader import load_config, interpolate_env_vars


class TestInterpolateEnvVars:
    def test_simple_substitution(self):
        os.environ['TEST_HOST'] = 'myhost'
        assert interpolate_env_vars('${TEST_HOST}') == 'myhost'
        del os.environ['TEST_HOST']

    def test_default_value(self):
        result = interpolate_env_vars('${NONEXISTENT_VAR:default_val}')
        assert result == 'default_val'

    def test_no_substitution(self):
        assert interpolate_env_vars('plain_string') == 'plain_string'

    def test_empty_default(self):
        result = interpolate_env_vars('${NONEXISTENT_VAR:}')
        assert result == ''

    def test_integer_conversion(self):
        os.environ['TEST_PORT'] = '6379'
        result = interpolate_env_vars('${TEST_PORT:6379}')
        assert result == '6379'
        del os.environ['TEST_PORT']


class TestLoadConfig:
    def test_load_yaml_file(self):
        config_data = {
            'redis': {'host': 'localhost', 'port': 6379},
            'training': {'env_name': 'CartPole'}
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)
            assert config['redis']['host'] == 'localhost'
            assert config['training']['env_name'] == 'CartPole'
        os.unlink(f.name)

    def test_env_var_interpolation_in_yaml(self):
        os.environ['TEST_REDIS_HOST'] = 'redis.prod.internal'
        config_data = {'redis': {'host': '${TEST_REDIS_HOST:localhost}'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config = load_config(f.name)
            assert config['redis']['host'] == 'redis.prod.internal'
        del os.environ['TEST_REDIS_HOST']
        os.unlink(f.name)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path.yaml')

    def test_merge_configs(self):
        base = {'redis': {'host': 'localhost', 'port': 6379}, 'training': {'lr': 0.001}}
        override = {'redis': {'host': 'prod-host'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as bf:
            yaml.dump(base, bf)
            bf.flush()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as of:
            yaml.dump(override, of)
            of.flush()
        config = load_config(bf.name, of.name)
        assert config['redis']['host'] == 'prod-host'
        assert config['redis']['port'] == 6379
        assert config['training']['lr'] == 0.001
        os.unlink(bf.name)
        os.unlink(of.name)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_config_loader.py -v`
Expected: FAIL (module `libs.config_loader` not found)

- [ ] **Step 3: Implement config_loader.py**

Create `libs/config_loader.py`:

```python
# -*- coding: utf-8 -*-
"""
YAML configuration loader with environment variable interpolation.
"""
import os
import re
import copy
import yaml
from typing import Any, Dict, Optional


_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def interpolate_env_vars(value: str) -> str:
    """Replace ${VAR:default} patterns with environment variable values."""
    if not isinstance(value, str):
        return value

    def _replace(match):
        expr = match.group(1)
        if ':' in expr:
            var_name, default = expr.split(':', 1)
        else:
            var_name, default = expr, ''
        return os.environ.get(var_name, default)

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _interpolate_recursive(obj: Any) -> Any:
    """Recursively interpolate env vars in a nested dict/list structure."""
    if isinstance(obj, str):
        return interpolate_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_recursive(item) for item in obj]
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str, override_path: Optional[str] = None) -> Dict:
    """Load YAML config with optional override file and env var interpolation.

    Args:
        base_path: Path to the base YAML config file.
        override_path: Optional path to an override YAML file.

    Returns:
        Merged and interpolated configuration dictionary.

    Raises:
        FileNotFoundError: If base_path does not exist.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with open(base_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    if override_path and os.path.exists(override_path):
        with open(override_path, 'r') as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    return _interpolate_recursive(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config_loader.py -v`
Expected: All PASS

### Task 7: Create default YAML config files

**Files:**
- Create: `config/default.yaml`
- Create: `config/dev.yaml`

- [ ] **Step 1: Create config/default.yaml**

```yaml
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
    password: "${REDIS_PASSWORD:}"
  grads:
    host: "${REDIS_GRADS_HOST:localhost}"
    port: 6379
    db: 2
    password: "${REDIS_PASSWORD:}"

training:
  env_name: "DQNGymClassic"
  num_samplers: 2
  num_trainers: 1
  num_update_grads: 1
  batch_update_grads_server: 10
  enable_checker: true
  version_update_sample_model: 1
  version_update_calculate_model: 1
  checkpoint_interval: 100
  max_versions: 10000

queues:
  len_grads_queue: 1000
  len_batch_queue: 1000
  len_sample_queue: 1000

logging:
  level: "INFO"
  dir: "logs"
  tensorboard: true
```

- [ ] **Step 2: Create config/dev.yaml**

```yaml
redis:
  model:
    host: "localhost"
  exps:
    host: "localhost"
  grads:
    host: "localhost"

training:
  env_name: "DQNGymClassic"
  num_samplers: 1
  num_trainers: 1

logging:
  level: "DEBUG"
```

### Task 8: Update libs/config.py to use YAML loader

**Files:**
- Modify: `libs/config.py`

- [ ] **Step 1: Refactor config.py to load from YAML**

Replace the entire hardcoded Redis configuration section (lines 57-122) in `libs/config.py` and update the getter functions to use the YAML loader:

```python
import libs.config_loader as config_loader

# Load configuration from YAML
_CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml')
_OVERRIDE_PATH = os.environ.get('RL_SERVER_CONFIG', None)
_config = config_loader.load_config(_CONFIG_PATH, _OVERRIDE_PATH)


def get_current_env_name() -> str:
    return _config.get('training', {}).get('env_name', 'DQNGymClassic')


def get_current_queue_config() -> dict:
    training = _config.get('training', {})
    queues = _config.get('queues', {})
    return {
        'len_grads_queue': queues.get('len_grads_queue', 1000),
        'len_batch_queue': queues.get('len_batch_queue', 1000),
        'len_sample_queue': queues.get('len_sample_queue', 1000),
        'batch_update_grads_server': training.get('batch_update_grads_server', 10),
        'enable_checker': training.get('enable_checker', True),
        'num_trainer': training.get('num_trainers', 1),
        'num_sampler': training.get('num_samplers', 2),
        'num_update_grads': training.get('num_update_grads', 1),
        'version_update_sample_model': training.get('version_update_sample_model', 1),
        'version_update_calculate_model': training.get('version_update_calculate_model', 1),
    }


def _redis_dict_from_config(section: str) -> dict:
    redis_cfg = _config.get('redis', {}).get(section, {})
    return {
        'ip': redis_cfg.get('host', 'localhost'),
        'port': str(redis_cfg.get('port', 6379)),
        'db': str(redis_cfg.get('db', 0)),
        'pw': redis_cfg.get('password', ''),
    }


def get_current_redis_MODEL_CONFIG() -> dict:
    return _redis_dict_from_config('model')


def get_current_redis_exps_config() -> dict:
    return _redis_dict_from_config('exps')


def get_current_redis_grads_config() -> dict:
    return _redis_dict_from_config('grads')
```

Remove all the old `redis_args_dict_*` dictionaries and `queue_args_dict`. Keep the `create_net`, `create_agent`, `create_calculate` factory functions and algorithm imports unchanged.

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('libs/config.py').read()); print('OK')"`
Expected: `OK`

### Task 9: Commit Stage 2

- [ ] **Step 1: Stage and commit**

```bash
git add libs/config_loader.py libs/config.py config/ tests/unit/test_config_loader.py
git commit -m "feat: add YAML config system, remove hardcoded credentials

- Add config_loader.py with env var interpolation (${VAR:default})
- Create config/default.yaml and config/dev.yaml
- Refactor config.py to load from YAML files
- Support RL_SERVER_CONFIG env var for config overrides
- Remove all hardcoded Redis IPs and passwords"
```

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Stage 3: Error Handling, Logging, Graceful Shutdown

### Task 10: Replace custom Log class with Python logging

**Files:**
- Modify: `libs/log.py`

- [ ] **Step 1: Rewrite log.py using Python logging module**

Replace the entire `libs/log.py` with:

```python
# -*- coding: utf-8 -*-
"""
Structured logging with worker identity.
Backwards-compatible: Log class interface preserved.
"""
import os
import logging
import sys


def setup_logging(dir_name: str, level: str = 'INFO') -> logging.Logger:
    """Create a logger with both file and console handlers."""
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs', dir_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(dir_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # File handler
        fh = logging.FileHandler(
            os.path.join(log_dir, f'{dir_name}.log'),
            encoding='utf-8'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


class Log:
    """Backwards-compatible wrapper around Python logging."""

    def __init__(self, dir_name: str):
        self.logger = setup_logging(dir_name)

    def log_info(self, message: str, print_screen: bool = False):
        self.logger.info(message)

    def log_exception(self, print_screen: bool = False):
        self.logger.exception("Exception occurred")
```

This preserves the existing `Log` interface so all callers (`train_main_local.py`, `grads_main.py`, etc.) continue to work without modification. The `print_screen` parameter is accepted but ignored since the console handler always outputs.

- [ ] **Step 2: Verify syntax and import**

Run: `python -c "import ast; ast.parse(open('libs/log.py').read()); print('OK')"`
Expected: `OK`

### Task 11: Add retry logic to Redis operations

**Files:**
- Modify: `libs/redis_cache.py`

- [ ] **Step 1: Add retry decorator to redis_cache.py**

Add a retry helper at the top of `libs/redis_cache.py` (after the imports):

```python
import time as _time
import functools


def _retry(max_retries=3, base_delay=1.0):
    """Retry decorator with exponential backoff for Redis operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except redis.ConnectionError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.log.log_info(f"Redis connection error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        _time.sleep(delay)
                        try:
                            self.conn.ping()
                        except Exception:
                            self._reconnect()
            self.log.log_exception()
            raise last_exception
        return wrapper
    return decorator
```

- [ ] **Step 2: Add reconnection and health check methods to RedisCache**

Add these methods to the `RedisCache` class:

```python
    def _reconnect(self):
        """Attempt to reconnect to Redis."""
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = redis.Redis(
            host=self.redis_config['ip'],
            port=self.redis_config['port'],
            db=self.redis_config['db'],
            password=self.redis_config['pw']
        )

    def health_check(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return self.conn.ping()
        except Exception:
            return False
```

- [ ] **Step 3: Apply @_retry to push_exps, push_grads, set_train_version_model**

Add `@_retry()` decorator before these methods:

```python
    @_retry()
    def push_exps(self, exps: List, sample_version: int):
        # ... existing implementation unchanged ...

    @_retry()
    def push_grads(self, grads: List, grads_version: int, sample_version: int):
        # ... existing implementation unchanged ...

    @_retry()
    def set_train_version_model(self, version: int, model: nn.Module):
        # ... existing implementation unchanged ...
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('libs/redis_cache.py').read()); print('OK')"`
Expected: `OK`

### Task 12: Replace file-based exit with signal handling

**Files:**
- Modify: `libs/utils.py`

- [ ] **Step 1: Add signal-based shutdown support**

Add to the top of `libs/utils.py` (after existing imports):

```python
import signal
import threading

_shutdown_event = threading.Event()


def setup_signal_handlers():
    """Install SIGTERM/SIGINT handlers that set the shutdown event."""
    def _handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def exit_run() -> bool:
    """Check if shutdown was requested (via signal or legacy exit.cmd file)."""
    if _shutdown_event.is_set():
        return True
    # Legacy fallback: check for exit.cmd file
    path = os.path.abspath(os.path.dirname(__file__) + '/' + '../')
    if os.path.exists(path + "/exit.cmd"):
        return True
    return False
```

Replace the existing `exit_run()` function with the version above, and add the `setup_signal_handlers` function. The legacy file check is kept for backward compatibility.

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('libs/utils.py').read()); print('OK')"`
Expected: `OK`

### Task 13: Commit Stage 3

- [ ] **Step 1: Stage and commit**

```bash
git add libs/log.py libs/redis_cache.py libs/utils.py
git commit -m "feat: add production error handling, graceful shutdown, structured logging

- Replace custom Log class with Python logging (structured format, file+console)
- Add retry decorator with exponential backoff for Redis operations
- Add Redis health_check() and _reconnect() methods
- Add signal-based shutdown (SIGTERM/SIGINT) while keeping exit.cmd fallback"
```

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Stage 4: Comprehensive Test Suite

### Task 14: Set up test infrastructure

**Files:**
- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pytest.ini**

```ini
[pytest]
testpaths = tests
markers =
    integration: marks tests as integration tests (require external services)
timeout = 30
```

- [ ] **Step 2: Create requirements-dev.txt**

```
-r requirements.txt
fakeredis>=2.0
pytest>=8.0
pytest-timeout>=2.2
```

- [ ] **Step 3: Create test package files**

Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` as empty files.

- [ ] **Step 4: Create tests/conftest.py**

```python
import os
import sys
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model checkpoints."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def small_dqn_net():
    """Small DQN network for fast testing."""
    from algo_envs.dqn_gym_classic import DQNGymClassicNet
    net = DQNGymClassicNet()
    return net
```

- [ ] **Step 5: Install dev dependencies**

Run: `pip install -r requirements-dev.txt`

### Task 15: Write unit tests for actions.py

**Files:**
- Create: `tests/unit/test_actions.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from libs.actions import ArgmaxActionSelector, EpsilonGreedyActionSelector, ProbabilityActionSelector, EpsilonTracker


class TestArgmaxActionSelector:
    def test_selects_max_index(self):
        selector = ArgmaxActionSelector()
        scores = np.array([[0.1, 0.9, 0.3], [0.7, 0.2, 0.1]])
        actions = selector(scores)
        assert actions[0] == 1
        assert actions[1] == 0

    def test_batch_output_shape(self):
        selector = ArgmaxActionSelector()
        scores = np.random.randn(5, 4)
        actions = selector(scores)
        assert actions.shape == (5,)


class TestEpsilonGreedyActionSelector:
    def test_zero_epsilon_is_greedy(self):
        selector = EpsilonGreedyActionSelector(epsilon=0.0)
        scores = np.array([[0.1, 0.9], [0.8, 0.2]])
        actions = selector(scores)
        assert actions[0] == 1
        assert actions[1] == 0

    def test_full_epsilon_is_random(self):
        np.random.seed(42)
        selector = EpsilonGreedyActionSelector(epsilon=1.0)
        scores = np.zeros((100, 4))
        actions = selector(scores)
        # With full randomness, we should see variety in actions
        assert len(set(actions)) > 1


class TestProbabilityActionSelector:
    def test_samples_from_distribution(self):
        np.random.seed(42)
        selector = ProbabilityActionSelector()
        # Heavily skewed distribution
        probs = np.array([[0.01, 0.01, 0.98]])
        actions = [selector(probs)[0] for _ in range(100)]
        # Action 2 should dominate
        assert actions.count(2) > 80

    def test_output_shape(self):
        selector = ProbabilityActionSelector()
        probs = np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]])
        actions = selector(probs)
        assert actions.shape == (3,)


class TestEpsilonTracker:
    def test_epsilon_decays(self):
        selector = EpsilonGreedyActionSelector(epsilon=1.0)
        tracker = EpsilonTracker(selector, eps_start=1.0, eps_final=0.01, eps_frames=100)
        tracker.frame(0)
        assert selector.epsilon == 1.0
        tracker.frame(50)
        assert selector.epsilon == 0.5
        tracker.frame(200)
        assert selector.epsilon == 0.01  # clamped at eps_final
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_actions.py -v`
Expected: All PASS

### Task 16: Write unit tests for exps.py

**Files:**
- Create: `tests/unit/test_exps.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from libs.exps import Experience, ExperienceBuffer, TrajectoryBuffer


class TestExperienceBuffer:
    def test_append_and_len(self):
        buf = ExperienceBuffer(capacity=100)
        assert len(buf) == 0
        exp = Experience(state=np.zeros(4), action=1, reward=1.0, done=False, next_state=np.zeros(4))
        buf.append(exp)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ExperienceBuffer(capacity=3)
        for i in range(5):
            exp = Experience(state=np.array([i]), action=i, reward=float(i), done=False, next_state=np.array([i+1]))
            buf.append(exp)
        assert len(buf) == 3

    def test_sample_returns_correct_shapes(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(20):
            exp = Experience(
                state=np.array([i, i+1]),
                action=i % 3,
                reward=float(i),
                done=False,
                next_state=np.array([i+1, i+2])
            )
            buf.append(exp)
        states, actions, rewards, dones, next_states = buf.sample(5)
        assert states.shape == (5, 2)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert dones.shape == (5,)
        assert next_states.shape == (5, 2)

    def test_sample_raises_on_insufficient_data(self):
        buf = ExperienceBuffer(capacity=100)
        exp = Experience(state=np.zeros(2), action=0, reward=0.0, done=False, next_state=np.zeros(2))
        buf.append(exp)
        with pytest.raises(ValueError):
            buf.sample(5)


class TestTrajectoryBuffer:
    def test_append_and_len(self):
        buf = TrajectoryBuffer(capacity=10)
        buf.append([1, 2, 3])
        assert len(buf) == 1

    def test_sample(self):
        buf = TrajectoryBuffer(capacity=100)
        for i in range(10):
            buf.append([i, i*2])
        samples = buf.sample(3)
        assert len(samples) == 3
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_exps.py -v`
Expected: All PASS

### Task 17: Write unit tests for algo_base.py

**Files:**
- Create: `tests/unit/test_algo_base.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algo_envs.algo_base import NoisyLinear, layer_init, AlgoBaseNet, GradCoef


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(4, 8)
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_noise_changes_output_in_train_mode(self):
        layer = NoisyLinear(4, 8)
        layer.train()
        x = torch.randn(1, 4)
        layer.sample_noise()
        out1 = layer(x).detach().clone()
        layer.sample_noise()
        out2 = layer(x).detach().clone()
        # Different noise should produce different outputs (with very high probability)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_eval_mode_is_deterministic(self):
        layer = NoisyLinear(4, 8)
        layer.eval()
        x = torch.randn(1, 4)
        out1 = layer(x).detach().clone()
        out2 = layer(x).detach().clone()
        assert torch.allclose(out1, out2)


class TestLayerInit:
    def test_orthogonal_init(self):
        linear = torch.nn.Linear(4, 8)
        initialized = layer_init(linear, method='orthogonal')
        assert initialized is linear
        # Orthogonal matrix: W^T W should be close to identity (scaled)
        w = linear.weight.data
        product = w @ w.T
        # Check diagonal elements are non-zero
        assert torch.all(torch.diag(product) > 0)

    def test_kaiming_init(self):
        linear = torch.nn.Linear(4, 8)
        initialized = layer_init(linear, method='kaiming')
        assert initialized is linear


class TestGradCoef:
    def test_forward_preserves_values(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = GradCoef.apply(x, 0.5)
        assert torch.allclose(x, y)

    def test_backward_scales_gradient(self):
        x = torch.randn(3, 4, requires_grad=True)
        coeff = 0.5
        y = GradCoef.apply(x, coeff)
        loss = y.sum()
        loss.backward()
        # Gradient should be coeff * 1.0 for each element
        expected_grad = torch.full_like(x, coeff)
        assert torch.allclose(x.grad, expected_grad)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_algo_base.py -v`
Expected: All PASS

### Task 18: Write unit tests for DQN

**Files:**
- Create: `tests/unit/test_dqn.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algo_envs.dqn_gym_classic import DQNGymClassicNet, DQNGymClassicCalculate


class TestDQNGymClassicNet:
    def test_forward_output_shape(self):
        net = DQNGymClassicNet()
        states = torch.randn(4, 4)  # batch_size=4, obs_dim=4 (CartPole)
        actions = net(states)
        assert actions.shape == (4,)

    def test_get_q_values_shape(self):
        net = DQNGymClassicNet()
        states = torch.randn(4, 4)
        q_values = net.get_q_values(states)
        assert q_values.shape == (4, 2)  # 2 actions for CartPole

    def test_update_state_modifies_parameters(self):
        net = DQNGymClassicNet()
        params_before = [p.data.clone() for p in net.parameters()]
        # Create fake gradients
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(version=1, grads_buffer=grads)
        params_after = [p.data.clone() for p in net.parameters()]
        # At least one parameter should have changed
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed

    def test_gradients_are_computed(self):
        net = DQNGymClassicNet()
        states = torch.randn(2, 4)
        q_values = net.get_q_values(states)
        loss = q_values.sum()
        loss.backward()
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grads


class TestDQNGymClassicCalculate:
    def test_generate_grads_returns_numpy_arrays(self):
        share_net = DQNGymClassicNet()
        calc = DQNGymClassicCalculate(share_net)
        # Fill buffer with enough samples
        for i in range(300):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            done = False
            version = 0
            calc.exps_buffer.append(
                __import__('libs.exps', fromlist=['Experience']).Experience(
                    state=state, action=action, reward=reward, done=done, next_state=state
                )
            )
        # Create sample batch
        samples = []
        for i in range(10):
            samples.append([
                np.random.randn(4).astype(np.float32),  # state
                np.random.randint(0, 2),  # action
                np.random.randn(),  # reward
                False,  # done (index 3 is unused, index 4 is done in DQN)
                False,  # done
            ])
        model_dict = {'TRAIN_VERSION': 0}
        grads_list, version = calc.generate_grads(samples, model_dict)
        assert len(grads_list) == 1
        assert version == 0
        # Grads should be numpy arrays
        for grad in grads_list[0]:
            assert isinstance(grad, np.ndarray) or grad is None
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_dqn.py -v`
Expected: All PASS

### Task 19: Write unit tests for PPO Mujoco

**Files:**
- Create: `tests/unit/test_ppo_mujoco.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algo_envs.ppo_mujoco_normal import MujocoNormalNet


class TestMujocoNormalNet:
    def test_forward_output_shape(self):
        net = MujocoNormalNet()
        # current_env_name='Pusher', OBS_DIM=23, ACT_DIM=7
        states = torch.randn(4, 23)
        actions = net(states)
        assert actions.shape == (4, 7)

    def test_get_distributions(self):
        net = MujocoNormalNet()
        states = torch.randn(4, 23)
        dists = net.get_distributions(states)
        sample = dists.sample()
        assert sample.shape == (4, 7)

    def test_get_sample_data(self):
        net = MujocoNormalNet()
        states = torch.randn(4, 23)
        actions, log_probs = net.get_sample_data(states)
        assert actions.shape == (4, 7)
        assert log_probs.shape == (4, 7)

    def test_get_calculate_data(self):
        net = MujocoNormalNet()
        states = torch.randn(4, 23)
        actions = torch.randn(4, 7)
        values, log_probs, entropy = net.get_calculate_data(states, actions)
        assert values.shape == (4, 1)
        assert log_probs.shape == (4, 7)
        assert entropy.shape == (4, 7)

    def test_update_state_modifies_parameters(self):
        net = MujocoNormalNet()
        params_before = [p.data.clone() for p in net.parameters()]
        grads = [np.random.randn(*p.shape).astype(np.float32) for p in net.parameters()]
        net.update_state(version=1, grads_buffer=grads)
        params_after = [p.data.clone() for p in net.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed

    def test_gradients_flow(self):
        net = MujocoNormalNet()
        states = torch.randn(2, 23)
        actions = torch.randn(2, 7)
        values, log_probs, entropy = net.get_calculate_data(states, actions)
        loss = values.sum() + log_probs.sum()
        loss.backward()
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
        assert has_grads
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_ppo_mujoco.py -v`
Expected: All PASS

### Task 20: Write unit tests for utils.py

**Files:**
- Create: `tests/unit/test_utils.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import torch
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import libs.utils as utils


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path):
        # Create a simple model
        model = torch.nn.Linear(4, 2)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Patch the base_dir to use tmp_path
        save_dir = tmp_path / "models" / "test_prefix"
        save_dir.mkdir(parents=True)

        # Save directly using torch.save (bypassing path logic)
        checkpoint = {'state_dict': model.state_dict(), 'version': 1, 'timestamp': 0}
        save_path = save_dir / "test_prefix_1_20260330.td"
        torch.save(checkpoint, save_path)

        # Load into a fresh model
        fresh_model = torch.nn.Linear(4, 2)
        loaded = torch.load(save_path)
        fresh_model.load_state_dict(loaded['state_dict'])

        # Verify weights match
        for key in original_state:
            assert torch.allclose(original_state[key], fresh_model.state_dict()[key])


class TestSetupSeed:
    def test_deterministic_output(self):
        utils.setup_seed(42)
        t1 = torch.randn(3)
        utils.setup_seed(42)
        t2 = torch.randn(3)
        assert torch.allclose(t1, t2)


class TestExitRun:
    def test_no_exit_file(self, tmp_path):
        # exit_run checks for exit.cmd in project root
        # Without the file, it should return False (unless signal was sent)
        # Just test the signal path
        assert not utils._shutdown_event.is_set()

    def test_signal_shutdown(self):
        utils._shutdown_event.clear()
        assert not utils.exit_run()
        utils._shutdown_event.set()
        assert utils.exit_run()
        utils._shutdown_event.clear()
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_utils.py -v`
Expected: All PASS

### Task 21: Write unit tests for redis_cache.py

**Files:**
- Create: `tests/unit/test_redis_cache.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import torch
import pytest
import pickle
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False

from libs.log import Log


@pytest.fixture
def fake_redis_cache():
    if not HAS_FAKEREDIS:
        pytest.skip("fakeredis not installed")
    from libs.redis_cache import RedisCache
    log = Log("test_redis")
    config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
    cache = RedisCache.__new__(RedisCache)
    cache.log = log
    cache.redis_config = config
    cache.conn = fakeredis.FakeRedis()
    return cache


class TestRedisCache:
    def test_set_and_get_exit_flag(self, fake_redis_cache):
        cache = fake_redis_cache
        cache.set_exit_flag(True)
        flag = cache.get_exit_flag()
        assert flag == 1

    def test_get_exit_flag_when_not_set(self, fake_redis_cache):
        cache = fake_redis_cache
        flag = cache.get_exit_flag()
        assert flag is None

    def test_set_and_get_train_version_model(self, fake_redis_cache):
        cache = fake_redis_cache
        model = torch.nn.Linear(4, 2)
        cache.set_train_version_model(5, model)
        version = cache.get_train_version()
        assert version == 5

    def test_get_train_model_loads_state(self, fake_redis_cache):
        cache = fake_redis_cache
        model = torch.nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        cache.set_train_version_model(1, model)

        # Load into fresh model
        fresh_model = torch.nn.Linear(4, 2)
        result = cache.get_train_model(fresh_model)
        assert result is True
        assert torch.allclose(original_weight, fresh_model.weight.data)

    def test_push_and_pop_exps(self, fake_redis_cache):
        cache = fake_redis_cache
        exps = [[1, 2, 3], [4, 5, 6]]
        cache.push_exps(exps, sample_version=1)
        # pop_exps uses brpop which fakeredis may not support with timeout
        # Test the push side at minimum
        assert cache.conn.llen('exps') == 1

    def test_clear_data(self, fake_redis_cache):
        cache = fake_redis_cache
        cache.conn.set('test_key', 'test_value')
        cache.clear_data()
        assert cache.conn.get('test_key') is None

    def test_health_check(self, fake_redis_cache):
        cache = fake_redis_cache
        assert cache.health_check() is True
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_redis_cache.py -v`
Expected: All PASS

### Task 22: Write unit tests for config.py

**Files:**
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write tests**

```python
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import libs.config as config


class TestConfigFactories:
    def test_create_net_dqn(self):
        net = config.create_net("DQNGymClassic")
        assert net is not None
        # Should have parameters
        params = list(net.parameters())
        assert len(params) > 0

    def test_create_net_invalid_raises(self):
        with pytest.raises(SystemExit):
            config.create_net("NonExistentAlgo")

    def test_get_current_env_name(self):
        name = config.get_current_env_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_queue_config_has_required_keys(self):
        qc = config.get_current_queue_config()
        required_keys = ['num_trainer', 'num_sampler', 'len_grads_queue',
                         'len_sample_queue', 'num_update_grads']
        for key in required_keys:
            assert key in qc, f"Missing key: {key}"

    def test_get_redis_config_has_required_keys(self):
        rc = config.get_current_redis_MODEL_CONFIG()
        required_keys = ['ip', 'port', 'db', 'pw']
        for key in required_keys:
            assert key in rc, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: All PASS

### Task 23: Write integration test for local pipeline

**Files:**
- Create: `tests/integration/test_local_pipeline.py`

- [ ] **Step 1: Write test**

```python
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from algo_envs.dqn_gym_classic import DQNGymClassicNet, DQNGymClassicAgent, DQNGymClassicCalculate
import libs.exps as Exps


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestLocalDQNPipeline:
    """Test full local pipeline: create net -> sample -> compute grads -> update."""

    def test_sample_train_cycle(self):
        # 1. Create network
        net = DQNGymClassicNet()

        # 2. Create agent and sample
        agent = DQNGymClassicAgent(net, is_checker=False)
        model_dict = {'TRAIN_VERSION': 0}
        exps_list = agent.sample_multi_envs(model_dict)
        assert len(exps_list) > 0
        assert len(exps_list[0]) > 0

        # 3. Compute gradients
        calc = DQNGymClassicCalculate(net)
        # Feed enough samples into buffer first
        all_samples = []
        for env_exps in exps_list:
            all_samples.extend(env_exps)

        # Need at least batch_size samples in buffer
        for sample in all_samples[:500]:
            state = np.array(sample[0], dtype=np.float32)
            action = sample[1]
            reward = sample[2]
            done = sample[3]
            next_state = state  # approximate
            exp = Exps.Experience(state, action, reward, done, next_state)
            calc.exps_buffer.append(exp)

        if len(calc.exps_buffer) >= calc.batch_size:
            grads_list, version = calc.generate_grads(all_samples[:10], model_dict)
            assert len(grads_list) == 1

            # 4. Update network
            params_before = [p.data.clone() for p in net.parameters()]
            net.update_state(1, grads_list[0])
            params_after = [p.data.clone() for p in net.parameters()]
            changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
            assert changed


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestCheckpointRoundtrip:
    """Test save and reload produces identical forward pass."""

    def test_checkpoint_roundtrip(self, tmp_path):
        net = DQNGymClassicNet()
        states = torch.randn(2, 4)
        output_before = net(states).detach().clone()

        # Save
        checkpoint = {'state_dict': net.state_dict(), 'version': 1}
        path = tmp_path / "checkpoint.td"
        torch.save(checkpoint, path)

        # Load into fresh network
        fresh_net = DQNGymClassicNet()
        loaded = torch.load(path)
        fresh_net.load_state_dict(loaded['state_dict'])
        output_after = fresh_net(states).detach().clone()

        assert torch.allclose(output_before, output_after)
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/integration/test_local_pipeline.py -v -m integration`
Expected: All PASS

### Task 24: Run full test suite and commit Stage 4

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 2: Stage and commit**

```bash
git add pytest.ini requirements-dev.txt tests/
git commit -m "test: add unit and integration tests for all core components

- Unit tests: actions, exps, algo_base, DQN, PPO Mujoco, redis_cache, config, utils
- Integration tests: local DQN pipeline, checkpoint roundtrip
- Test infrastructure: conftest.py with fixtures, pytest.ini, requirements-dev.txt
- Uses fakeredis for Redis unit tests"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Stage 5: Package Restructure

### Task 25: Create rl_server package skeleton

**Files:**
- Create: All `__init__.py` files for the new package structure

- [ ] **Step 1: Create all package directories and __init__.py files**

```bash
mkdir -p rl_server/core rl_server/algorithms/dqn rl_server/algorithms/ppo rl_server/algorithms/sac rl_server/algorithms/td3 rl_server/workers rl_server/transport rl_server/config rl_server/entrypoints rl_server/utils
```

Create empty `__init__.py` in each:
- `rl_server/__init__.py`
- `rl_server/core/__init__.py`
- `rl_server/algorithms/__init__.py`
- `rl_server/algorithms/dqn/__init__.py`
- `rl_server/algorithms/ppo/__init__.py`
- `rl_server/algorithms/sac/__init__.py`
- `rl_server/algorithms/td3/__init__.py`
- `rl_server/workers/__init__.py`
- `rl_server/transport/__init__.py`
- `rl_server/config/__init__.py`
- `rl_server/entrypoints/__init__.py`
- `rl_server/utils/__init__.py`

### Task 26: Move core modules

**Files:**
- Create: `rl_server/core/base.py` (from `algo_envs/algo_base.py`)
- Create: `rl_server/core/actions.py` (from `libs/actions.py`)
- Create: `rl_server/core/buffers.py` (from `libs/exps.py`)
- Create: `rl_server/core/noisy.py` (extracted from `algo_envs/algo_base.py`)

- [ ] **Step 1: Create rl_server/core/noisy.py**

Extract `NoisyLinear` and `GradCoef` from `algo_base.py`:

```python
# -*- coding: utf-8 -*-
"""Noisy layers and gradient utilities."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class NoisyLinear(nn.Linear):
    """Linear layer with learnable noise for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input: torch.Tensor):
        if not self.training:
            return super().forward(input)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()


class GradCoef(autograd.Function):
    """Scale gradients during backpropagation."""
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.coeff * grad_output, None
```

- [ ] **Step 2: Create rl_server/core/base.py**

```python
# -*- coding: utf-8 -*-
"""Base classes for RL algorithms."""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List


def layer_init(linear_layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0, method: str = 'orthogonal') -> nn.Linear:
    """Initialize linear layer weights."""
    if method == 'orthogonal':
        nn.init.orthogonal_(linear_layer.weight, std)
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, bias_const)
    else:
        nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)
    return linear_layer


class AlgoBaseNet(nn.Module):
    """Base network for RL algorithms."""
    def __init__(self):
        super().__init__()

    def forward(self, states):
        raise NotImplementedError

    def update_state(self, version: int, grads_buffer: List):
        raise NotImplementedError


class AlgoBaseAgent:
    """Base agent for RL algorithms."""
    def __init__(self):
        pass

    def sample_multi_envs(self, model_dict: Dict):
        raise NotImplementedError

    def check_single_env(self):
        raise NotImplementedError


class AlgoBaseCalculate:
    """Base calculator for RL algorithms."""
    def __init__(self):
        pass

    def generate_grads(self, samples: List, model_dict: Dict):
        raise NotImplementedError
```

- [ ] **Step 3: Create rl_server/core/actions.py**

Copy `libs/actions.py` content, removing the `sys.path` hack:

```python
# -*- coding: utf-8 -*-
"""Action selection strategies."""
import numpy as np
from typing import Union


class ActionSelector:
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    def __call__(self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    def __call__(self, probs: np.ndarray):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)
```

- [ ] **Step 4: Create rl_server/core/buffers.py**

Copy `libs/exps.py` content, removing the `sys.path` hack:

```python
# -*- coding: utf-8 -*-
"""Experience and trajectory buffers."""
import collections
import numpy as np
from typing import List

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exps: List[Experience]):
        self.buffer.append(exps)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


class TrajectoryBuffer:
    def __init__(self, capacity: int = 1000000):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, trajectory: List):
        self.buffer.append(trajectory)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
```

### Task 27: Move transport modules

**Files:**
- Create: `rl_server/transport/redis_cache.py` (from `libs/redis_cache.py`)
- Create: `rl_server/transport/serialization.py`

- [ ] **Step 1: Create rl_server/transport/redis_cache.py**

Copy the updated `libs/redis_cache.py` (with timeout and retry from Stage 3), replacing `sys.path` hack and `from libs.log import Log` with `from rl_server.utils.logging import Log`.

- [ ] **Step 2: Create rl_server/transport/serialization.py**

```python
# -*- coding: utf-8 -*-
"""Serialization utilities for Redis transport."""
import pickle
import zlib
from typing import Any


def serialize(obj: Any) -> bytes:
    """Serialize object using pickle + zlib compression."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(data)


def deserialize(data: bytes) -> Any:
    """Deserialize object from compressed pickle bytes."""
    return pickle.loads(zlib.decompress(data))
```

### Task 28: Move utils modules

**Files:**
- Create: `rl_server/utils/logging.py` (from `libs/log.py`)
- Create: `rl_server/utils/checkpoint.py` (from `libs/utils.py` save/load)
- Create: `rl_server/utils/process.py` (from `libs/utils.py` setup/exit)

- [ ] **Step 1: Create rl_server/utils/logging.py**

Copy the updated `libs/log.py` (from Stage 3) without the `sys.path` hack.

- [ ] **Step 2: Create rl_server/utils/checkpoint.py**

Extract model save/load logic from `libs/utils.py`:

```python
# -*- coding: utf-8 -*-
"""Model checkpoint management."""
import os
import time
import glob
import torch
from typing import Optional, Dict


def get_model_state_path(prefix: str, version: Optional[str] = None, base_dir: str = 'models') -> Optional[str]:
    """Get the path to a model checkpoint file."""
    model_dir = os.path.normpath(os.path.join(base_dir, prefix))
    pattern = f"{prefix}_{version}_*.td" if version else f"{prefix}_*.td"
    candidates = glob.glob(os.path.join(model_dir, pattern))
    if version:
        candidates = [f for f in candidates if os.path.basename(f).startswith(f"{prefix}_{version}_")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime, default=None)


def load_model(model: torch.nn.Module, prefix: str, version: Optional[str] = None,
               base_dir: str = 'models', map_location: Optional[str] = None) -> Optional[int]:
    """Load model state dict from checkpoint."""
    file_path = get_model_state_path(prefix, version, base_dir)
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        checkpoint = torch.load(file_path, map_location=map_location)
        if 'state_dict' not in checkpoint or 'version' not in checkpoint:
            raise ValueError("Invalid checkpoint format")
        model.load_state_dict(checkpoint['state_dict'])
        return checkpoint['version']
    except (IOError, RuntimeError, ValueError) as e:
        print(f"Load model failed: {str(e)}")
        return None


def save_model(model: torch.nn.Module, prefix: str, version: str,
               base_dir: str = 'models', max_versions: int = 5) -> Optional[str]:
    """Save model state dict to checkpoint with atomic write."""
    save_dir = os.path.normpath(os.path.join(base_dir, prefix))
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{version}_{timestamp}.td"
    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + '.tmp'

    checkpoint = {
        'state_dict': model.state_dict(),
        'version': version,
        'timestamp': time.time(),
    }
    try:
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, save_path)  # atomic on same filesystem
        if max_versions > 0:
            existing = sorted(
                glob.glob(os.path.join(save_dir, f"{prefix}_*.td")),
                key=os.path.getmtime, reverse=True
            )
            for old_file in existing[max_versions:]:
                os.remove(old_file)
        return save_path
    except (IOError, RuntimeError) as e:
        print(f"Save model failed: {str(e)}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None
```

- [ ] **Step 3: Create rl_server/utils/process.py**

```python
# -*- coding: utf-8 -*-
"""Process management, signal handling, and seed setup."""
import os
import signal
import random
import threading
import numpy as np
import torch
import torch.multiprocessing as mp

_shutdown_event = threading.Event()


def setup_signal_handlers():
    """Install SIGTERM/SIGINT handlers that set the shutdown event."""
    def _handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def should_exit() -> bool:
    """Check if shutdown was requested."""
    return _shutdown_event.is_set()


def request_shutdown():
    """Programmatically request shutdown."""
    _shutdown_event.set()


def setup_seed(seed: int = 1970010101):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_mp():
    """Configure multiprocessing for PyTorch."""
    mp.set_start_method('spawn', force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
```

### Task 29: Move algorithm modules

- [ ] **Step 1: Create rl_server/algorithms/dqn/network.py, agent.py, calculator.py**

Split `algo_envs/dqn_gym_classic.py` into three files. Each file imports from `rl_server.core.base` instead of `algo_envs.algo_base`. Update `import gym` to `import gymnasium as gym` (for Stage 6).

For now, keep `import gym` and update in Stage 6. The split itself is:

- `network.py`: `TRAIN_ENVS`, `current_env_name`, `TRAIN_CONFIG`, `MODEL_CONFIG`, `DQNGymClassicNet`
- `agent.py`: `DQNGymClassicAgent` (imports network configs from `network.py`)
- `calculator.py`: `DQNGymClassicCalculate` (imports from `network.py`)

- [ ] **Step 2: Create rl_server/algorithms/ppo/mujoco_normal.py**

Copy `algo_envs/ppo_mujoco_normal.py`, updating imports to `from rl_server.core.base import ...` and `from rl_server.core.noisy import NoisyLinear`.

- [ ] **Step 3: Create rl_server/algorithms/ppo/microrts.py**

Copy `algo_envs/ppo_microrts.py`, updating imports similarly.

- [ ] **Step 4: Copy remaining algorithm files**

- `rl_server/algorithms/ppo/mujoco_beta.py` (from `algo_envs/ppo_mujoco_beta.py` + `beta_alpha.py` + `beta_relative.py`, merged with mode parameter)
- `rl_server/algorithms/ppo/mobile.py` (from `algo_envs/ppo_mobile_300.py`)
- `rl_server/algorithms/sac/mujoco_normal.py` (from `algo_envs/sac_mujoco_normal.py`)
- `rl_server/algorithms/td3/mujoco_normal.py` (from `algo_envs/td3_mujoco_normal.py`)

### Task 30: Move config module

- [ ] **Step 1: Create rl_server/config/loader.py**

Copy `libs/config_loader.py` as `rl_server/config/loader.py`.

- [ ] **Step 2: Create rl_server/config/schema.py**

```python
# -*- coding: utf-8 -*-
"""Configuration validation."""
from typing import Dict


REQUIRED_KEYS = {
    'redis': {'model', 'exps'},
    'training': {'env_name', 'num_samplers', 'num_trainers'},
}


def validate_config(config: Dict) -> None:
    """Validate config has all required sections and keys."""
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: {section}.{key}")
```

- [ ] **Step 3: Copy config/default.yaml to rl_server/config/default.yaml**

### Task 31: Create worker modules

- [ ] **Step 1: Create rl_server/workers/sampler.py**

Unified sampler that supports both local queue and Redis transport:

```python
# -*- coding: utf-8 -*-
"""Unified sampling worker."""
import time
import queue
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Dict

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed


class SamplerWorker:
    """Sampling worker that supports local queue or Redis transport."""

    def __init__(self, idx: int, model_dict: Dict, share_model: nn.Module,
                 env_name: str, log: Log, sample_queue: mp.Queue = None,
                 redis_cache=None):
        self.sampler_id = idx
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.sample_queue = sample_queue
        self.redis_cache = redis_cache
        self.process = None

    def process_function(self):
        setup_seed()
        # Import algorithm factories - these must be available in the new package
        from rl_server.algorithms import create_agent
        sample_agent = create_agent(self.env_name, self.share_model)

        while not self.model_dict['is_exit']:
            try:
                exps_list = sample_agent.sample_multi_envs(self.model_dict)
                if exps_list is None:
                    continue
                for exps in exps_list:
                    if self.redis_cache is not None:
                        self.redis_cache.push_exps(exps, self.model_dict['TRAIN_VERSION'])
                    elif self.sample_queue is not None:
                        exps_info = {'sample_version': self.model_dict['TRAIN_VERSION'], 'exps': exps}
                        self.sample_queue.put(exps_info)
                time.sleep(0)
            except queue.Full:
                time.sleep(5)
            except Exception:
                self.log.log_exception()

        try:
            del sample_agent
        except Exception:
            self.log.log_exception()
        self.log.log_info(f'exit sampler pid={self.process.pid} id={self.sampler_id}')

    def start(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start sampler pid={self.process.pid} id={self.sampler_id}')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
```

- [ ] **Step 2: Create rl_server/workers/trainer.py**

```python
# -*- coding: utf-8 -*-
"""Unified training worker."""
import time
import queue
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Dict

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed


class TrainerWorker:
    """Training worker that supports local queue or Redis transport."""

    def __init__(self, idx: int, model_dict: Dict, share_model: nn.Module,
                 env_name: str, log: Log, sample_queue: mp.Queue = None,
                 grads_queue: mp.Queue = None, redis_cache=None):
        self.trainer_id = idx
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.sample_queue = sample_queue
        self.grads_queue = grads_queue
        self.redis_cache = redis_cache
        self.process = None

    def process_function(self):
        setup_seed()
        from rl_server.algorithms import create_calculate
        calculate = create_calculate(self.env_name, self.share_model)

        # If using Redis for exps, set up Redis cache for pulling samples
        exps_redis_cache = None
        if self.redis_cache is not None:
            from rl_server.transport.redis_cache import RedisCache
            from rl_server.config.loader import load_config
            config = load_config('config/default.yaml')
            redis_cfg = config.get('redis', {}).get('exps', {})
            exps_redis_cache = RedisCache(self.log, {
                'ip': redis_cfg.get('host', 'localhost'),
                'port': str(redis_cfg.get('port', 6379)),
                'db': str(redis_cfg.get('db', 1)),
                'pw': redis_cfg.get('password', ''),
            })

        while not self.model_dict['is_exit']:
            try:
                samples = None
                exps_version = 0

                if exps_redis_cache is not None:
                    samples, exps_version = exps_redis_cache.pop_exps()
                elif self.sample_queue is not None:
                    try:
                        samples_info = self.sample_queue.get(timeout=1)
                        samples = samples_info['exps']
                        exps_version = samples_info['sample_version']
                    except queue.Empty:
                        continue

                if samples is not None:
                    grads_list, train_version = calculate.generate_grads(samples, self.model_dict)
                    for grads in grads_list:
                        grads_info = {
                            'grads': grads,
                            'grads_version': train_version,
                            'sample_version': exps_version,
                        }
                        self.grads_queue.put(grads_info)
                time.sleep(0)
            except queue.Full:
                time.sleep(1)
            except Exception:
                self.log.log_exception()

        if exps_redis_cache is not None:
            del exps_redis_cache
        self.log.log_info(f'exit trainer pid={self.process.pid} id={self.trainer_id}')

    def start(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start trainer pid={self.process.pid} id={self.trainer_id}')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
```

- [ ] **Step 3: Create rl_server/workers/checker.py**

```python
# -*- coding: utf-8 -*-
"""Model evaluation worker."""
import time
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed


class CheckerWorker:
    """Periodically evaluates model performance."""

    def __init__(self, model_dict, share_model: nn.Module, env_name: str, log: Log):
        self.model_dict = model_dict
        self.share_model = share_model
        self.self_version = model_dict['TRAIN_VERSION']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"

    def process_function(self):
        setup_seed()
        from rl_server.algorithms import create_net, create_agent
        check_net = create_net(self.env_name)
        check_agent = create_agent(self.env_name, check_net, is_checker=True)
        writer = SummaryWriter(comment=self.comment)

        while not self.model_dict['is_exit']:
            try:
                check_net.load_state_dict(self.share_model.state_dict())
                check_net.train(False)
                self.self_version = self.model_dict['TRAIN_VERSION']
                infos = check_agent.check_single_env()
                if isinstance(infos, dict):
                    for key, value in infos.items():
                        writer.add_scalar(key, value, self.self_version)
                time.sleep(0)
            except Exception:
                self.log.log_exception()

        writer.close()
        self.log.log_info(f'exit checker pid={self.process.pid}')

    def start(self, comment: str = None):
        if comment:
            self.comment = comment
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start checker pid={self.process.pid}')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join()
```

- [ ] **Step 4: Create rl_server/workers/grads_aggregator.py**

```python
# -*- coding: utf-8 -*-
"""Gradient aggregation worker (for distributed gradient server mode)."""
import time
import torch.nn as nn

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, should_exit
from rl_server.utils.checkpoint import load_model, save_model
from rl_server.transport.redis_cache import RedisCache


class GradsAggregator:
    """Pulls gradients from Redis, accumulates, and updates the network."""

    def __init__(self, net: nn.Module, grads_redis: RedisCache,
                 model_redis: RedisCache, log: Log,
                 batch_size: int = 10, model_prefix: str = "grads_main",
                 env_name: str = ""):
        self.net = net
        self.grads_redis = grads_redis
        self.model_redis = model_redis
        self.log = log
        self.batch_size = batch_size
        self.model_prefix = model_prefix
        self.env_name = env_name

    def run(self, initial_version: int = 0):
        current_version = initial_version
        grads_buffer = None
        grads_count = 0

        while not should_exit():
            try:
                if grads_count >= self.batch_size:
                    current_version += 1
                    self.net.update_state(current_version, grads_buffer)
                    self.model_redis.set_train_version_model(current_version, self.net)
                    grads_buffer = None
                    grads_count = 0
                else:
                    grads_item, grads_version, sample_version = self.grads_redis.pop_grads()
                    if grads_item is not None:
                        grads_count += 1
                        if grads_buffer is None:
                            grads_buffer = grads_item
                        else:
                            for target_grad, grad in zip(grads_buffer, grads_item):
                                target_grad += grad
            except Exception:
                self.log.log_exception()

        # Save on exit
        save_model(self.net, f"{self.model_prefix}_{self.env_name}", str(current_version))
        self.model_redis.set_exit_flag(1)
        self.log.log_info("grads aggregator exited")
```

### Task 32: Create entrypoints

- [ ] **Step 1: Create rl_server/entrypoints/train.py**

```python
# -*- coding: utf-8 -*-
"""Unified training entry point."""
import argparse
import torch.multiprocessing as mp

from rl_server.config.loader import load_config
from rl_server.config.schema import validate_config
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_mp, setup_signal_handlers, should_exit
from rl_server.utils.checkpoint import load_model, save_model


def main():
    parser = argparse.ArgumentParser(description='RL Server Training')
    parser.add_argument('--config', default='config/default.yaml', help='Path to config YAML')
    parser.add_argument('--override', default=None, help='Path to override config YAML')
    parser.add_argument('--mode', choices=['local', 'redis'], default='redis', help='Training mode')
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    validate_config(config)

    setup_mp()
    setup_seed()
    setup_signal_handlers()

    log = Log(f"train_{args.mode}")
    log.log_info(f"Starting training in {args.mode} mode")

    env_name = config['training']['env_name']
    model_prefix = f"train_{args.mode}"

    from rl_server.algorithms import create_net
    train_net = create_net(env_name)
    train_net.share_memory()

    version = load_model(train_net, f"{model_prefix}_{env_name}")
    current_version = version if version is not None else 0

    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = current_version

    grads_queue = mp.Queue(maxsize=config['queues']['len_grads_queue'])

    from rl_server.workers.trainer import TrainerWorker
    from rl_server.workers.sampler import SamplerWorker
    from rl_server.workers.checker import CheckerWorker

    redis_cache_obj = None
    if args.mode == 'redis':
        from rl_server.transport.redis_cache import RedisCache
        redis_cfg = config['redis']['model']
        redis_cache_obj = RedisCache(log, {
            'ip': redis_cfg.get('host', 'localhost'),
            'port': str(redis_cfg.get('port', 6379)),
            'db': str(redis_cfg.get('db', 0)),
            'pw': redis_cfg.get('password', ''),
        })
        redis_cache_obj.clear_data()
        redis_cache_obj.set_train_version_model(current_version, train_net)

    sample_queue = None
    samplers = []
    if args.mode == 'local':
        sample_queue = mp.Queue(maxsize=config['queues']['len_sample_queue'])
        for i in range(config['training']['num_samplers']):
            s = SamplerWorker(i, model_dict, train_net, env_name, log, sample_queue=sample_queue)
            samplers.append(s)
            s.start()

    trainers = []
    for i in range(config['training']['num_trainers']):
        t = TrainerWorker(i, model_dict, train_net, env_name, log,
                          sample_queue=sample_queue, grads_queue=grads_queue,
                          redis_cache=redis_cache_obj)
        trainers.append(t)
        t.start()

    if config['training'].get('enable_checker', False):
        checker = CheckerWorker(model_dict, train_net, env_name, log)
        checker.start()

    log.log_info(f"Training started: {env_name} in {args.mode} mode")

    import queue as queue_mod
    grads_buffer = None
    grads_count = 0
    num_update = config['training']['num_update_grads']

    while not should_exit():
        try:
            if grads_count >= num_update:
                current_version += 1
                train_net.update_state(current_version, grads_buffer)
                model_dict['TRAIN_VERSION'] = current_version
                if redis_cache_obj:
                    redis_cache_obj.set_train_version_model(current_version, train_net)
                grads_count = 0
                grads_buffer = None
            else:
                try:
                    grads_info = grads_queue.get(block=False)
                    grads_item = grads_info['grads']
                    grads_count += 1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for tg, g in zip(grads_buffer, grads_item):
                            tg += g
                except queue_mod.Empty:
                    pass
            time.sleep(0)
        except Exception:
            log.log_exception()

    # Shutdown
    log.log_info("Shutting down...")
    save_model(train_net, f"{model_prefix}_{env_name}", str(current_version))
    model_dict['is_exit'] = True
    for s in samplers:
        s.stop()
    for t in trainers:
        t.stop()
    if config['training'].get('enable_checker', False):
        checker.stop()
    if redis_cache_obj:
        redis_cache_obj.set_exit_flag(1)
    log.log_info("Shutdown complete")


if __name__ == '__main__':
    import time
    main()
```

- [ ] **Step 2: Create rl_server/entrypoints/sample.py**

```python
# -*- coding: utf-8 -*-
"""Standalone sampling worker entry point."""
import argparse
import time
import torch.multiprocessing as mp

from rl_server.config.loader import load_config
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_mp, setup_signal_handlers, should_exit
from rl_server.transport.redis_cache import RedisCache


def main():
    parser = argparse.ArgumentParser(description='RL Server Sampler')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--override', default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    setup_mp()
    setup_seed()
    setup_signal_handlers()

    log = Log("sample_main")
    env_name = config['training']['env_name']

    from rl_server.algorithms import create_net
    sample_net = create_net(env_name)
    sample_net.share_memory()

    redis_cfg = config['redis']['model']
    model_redis = RedisCache(log, {
        'ip': redis_cfg.get('host', 'localhost'),
        'port': str(redis_cfg.get('port', 6379)),
        'db': str(redis_cfg.get('db', 0)),
        'pw': redis_cfg.get('password', ''),
    })

    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = 0

    # Wait for initial model
    while True:
        version = model_redis.get_train_version()
        if version is not None and model_redis.get_train_model(sample_net):
            model_dict['TRAIN_VERSION'] = version
            break
        time.sleep(1)

    from rl_server.workers.sampler import SamplerWorker
    samplers = []
    for i in range(config['training']['num_samplers']):
        s = SamplerWorker(i, model_dict, sample_net, env_name, log, redis_cache=model_redis)
        samplers.append(s)
        s.start()

    log.log_info("Samplers started")
    current_version = model_dict['TRAIN_VERSION']

    while not should_exit():
        exit_flag = model_redis.get_exit_flag()
        if exit_flag and int(exit_flag):
            break
        new_version = model_redis.get_train_version()
        if new_version and new_version > current_version:
            if model_redis.get_train_model(sample_net):
                current_version = new_version
                model_dict['TRAIN_VERSION'] = current_version
        time.sleep(0)

    model_dict['is_exit'] = True
    for s in samplers:
        s.stop()
    log.log_info("Samplers shutdown complete")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Create rl_server/entrypoints/check.py**

```python
# -*- coding: utf-8 -*-
"""Model evaluation entry point."""
import argparse
import time
import torch.multiprocessing as mp

from rl_server.config.loader import load_config
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_mp, setup_signal_handlers, should_exit
from rl_server.transport.redis_cache import RedisCache


def main():
    parser = argparse.ArgumentParser(description='RL Server Checker')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--override', default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    setup_mp()
    setup_seed()
    setup_signal_handlers()

    log = Log("check_main")
    env_name = config['training']['env_name']

    from rl_server.algorithms import create_net
    check_net = create_net(env_name)
    check_net.share_memory()

    redis_cfg = config['redis']['model']
    model_redis = RedisCache(log, {
        'ip': redis_cfg.get('host', 'localhost'),
        'port': str(redis_cfg.get('port', 6379)),
        'db': str(redis_cfg.get('db', 0)),
        'pw': redis_cfg.get('password', ''),
    })

    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = 0

    # Wait for initial model
    while True:
        version = model_redis.get_train_version()
        if version is not None and model_redis.get_train_model(check_net):
            model_dict['TRAIN_VERSION'] = version
            break
        time.sleep(1)

    from rl_server.workers.checker import CheckerWorker
    checker = CheckerWorker(model_dict, check_net, env_name, log)
    checker.start()

    log.log_info("Checker started")
    current_version = model_dict['TRAIN_VERSION']

    while not should_exit():
        new_version = model_redis.get_train_version()
        if new_version and new_version > current_version:
            if model_redis.get_train_model(check_net):
                current_version = new_version
                model_dict['TRAIN_VERSION'] = current_version
        time.sleep(0)

    model_dict['is_exit'] = True
    checker.stop()
    log.log_info("Checker shutdown complete")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Create rl_server/entrypoints/grads.py**

```python
# -*- coding: utf-8 -*-
"""Gradient aggregation server entry point."""
import argparse

from rl_server.config.loader import load_config
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_mp, setup_signal_handlers
from rl_server.utils.checkpoint import load_model
from rl_server.transport.redis_cache import RedisCache
from rl_server.workers.grads_aggregator import GradsAggregator


def main():
    parser = argparse.ArgumentParser(description='RL Server Gradient Aggregator')
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--override', default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    setup_mp()
    setup_seed()
    setup_signal_handlers()

    log = Log("grads_main")
    env_name = config['training']['env_name']

    from rl_server.algorithms import create_net
    net = create_net(env_name)

    model_prefix = "grads_main"
    version = load_model(net, f"{model_prefix}_{env_name}")
    initial_version = version if version is not None else 0

    def _redis_dict(section):
        rc = config['redis'][section]
        return {'ip': rc.get('host', 'localhost'), 'port': str(rc.get('port', 6379)),
                'db': str(rc.get('db', 0)), 'pw': rc.get('password', '')}

    grads_redis = RedisCache(log, _redis_dict('grads'))
    grads_redis.clear_data()
    model_redis = RedisCache(log, _redis_dict('model'))
    model_redis.clear_data()
    model_redis.set_train_version_model(initial_version, net)

    batch_size = config['training'].get('batch_update_grads_server', 10)
    aggregator = GradsAggregator(net, grads_redis, model_redis, log,
                                  batch_size=batch_size, model_prefix=model_prefix,
                                  env_name=env_name)
    log.log_info("Starting gradient aggregation")
    aggregator.run(initial_version)
    log.log_info("Gradient aggregation complete")


if __name__ == '__main__':
    main()
```

### Task 33: Remove unused proto directory

- [ ] **Step 1: Delete unused proto files**

```bash
git rm -r proto/
```

### Task 34: Verify, test, and commit Stage 5

- [ ] **Step 1: Run import test**

Run: `python -c "from rl_server.core.base import AlgoBaseNet; print('OK')"`
Expected: `OK`

- [ ] **Step 2: Run existing tests (they still use old imports)**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All PASS (old tests still work against old code)

- [ ] **Step 3: Stage and commit**

```bash
git add rl_server/ -A
git rm -r proto/
git commit -m "refactor: restructure into rl_server package

- Create rl_server/ package with core/, algorithms/, workers/, transport/, config/, entrypoints/, utils/
- Split algo_base.py into base.py and noisy.py
- Split algorithm files into network/agent/calculator modules
- Unify sampler/trainer workers for local+Redis modes
- Create unified training entry point with argparse
- Remove unused proto/ directory
- Old code preserved for backward compatibility during migration"
```

- [ ] **Step 4: Push**

```bash
git push origin main
```

---

## Stage 6: Gymnasium Migration

### Task 35: Update all gym imports to gymnasium

**Files:**
- Modify: `rl_server/algorithms/dqn/agent.py`
- Modify: `rl_server/algorithms/ppo/mujoco_normal.py`
- Modify: `algo_envs/dqn_gym_classic.py`
- Modify: `algo_envs/ppo_mujoco_normal.py`
- Modify: All algorithm files that use `import gym`

- [ ] **Step 1: Replace `import gym` with `import gymnasium as gym` in all files**

In every algorithm file under `rl_server/algorithms/` and `algo_envs/`:

```python
# Old:
import gym

# New:
import gymnasium as gym
```

- [ ] **Step 2: Update env.step() calls**

In every agent's `sample_multi_envs` and `check_single_env`, update step return values:

```python
# Old:
next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])

# New (already correct for gym 0.26+ format):
next_state_n, reward_n, terminated, truncated, info = self.envs[i].step(actions[i])
done_n = terminated or truncated
```

- [ ] **Step 3: Update env.reset() calls**

Already correct — current code already uses `self.envs[i].reset()[0]` format.

- [ ] **Step 4: Update requirements.txt**

```
numpy>=1.26.4
gymnasium>=0.29.0
torch>=2.6.0
tensorboardX>=2.6
redis>=5.0.0
pyyaml>=6.0
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 6: Commit and push**

```bash
git add -A
git commit -m "feat: migrate from gym to gymnasium

- Replace import gym with import gymnasium as gym
- Update step() to handle terminated/truncated separately
- Update requirements.txt: gym->gymnasium>=0.29.0"
git push origin main
```

---

## Stage 7: Production Hardening

### Task 36: Implement atomic checkpoint management

**Files:**
- Modify: `rl_server/utils/checkpoint.py`

- [ ] **Step 1: Already implemented in Task 28 Step 2**

The `save_model` function in `rl_server/utils/checkpoint.py` already uses atomic writes (`.tmp` + `os.rename`) and retention policy. Verify it works.

### Task 37: Add worker heartbeats

**Files:**
- Modify: `rl_server/utils/process.py`

- [ ] **Step 1: Add heartbeat functions**

Add to `rl_server/utils/process.py`:

```python
import tempfile

HEARTBEAT_DIR = os.path.join(tempfile.gettempdir(), 'rl_server')


def write_heartbeat(worker_type: str, worker_id: int):
    """Write heartbeat file for monitoring."""
    os.makedirs(HEARTBEAT_DIR, exist_ok=True)
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    with open(path, 'w') as f:
        f.write(str(time.time()))


def cleanup_heartbeat(worker_type: str, worker_id: int):
    """Remove heartbeat file on shutdown."""
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    if os.path.exists(path):
        os.remove(path)
```

### Task 38: Add Redis connection pooling

**Files:**
- Modify: `rl_server/transport/redis_cache.py`

- [ ] **Step 1: Use ConnectionPool in RedisCache constructor**

```python
    def __init__(self, log, redis_config: dict):
        self.log = log
        self.redis_config = redis_config
        pool_size = int(redis_config.get('pool_size', 10))
        self.pool = redis.ConnectionPool(
            host=redis_config['ip'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config['pw'],
            max_connections=pool_size
        )
        self.conn = redis.Redis(connection_pool=self.pool)
        if not self.conn.ping():
            self.log.log_info("Redis connect fail, will exit")
            raise ConnectionError("Cannot connect to Redis")
```

### Task 39: Final dependency update and commit

- [ ] **Step 1: Verify final requirements.txt**

```
numpy>=1.26.4
gymnasium>=0.29.0
torch>=2.6.0
tensorboardX>=2.6
redis>=5.0.0
pyyaml>=6.0
```

- [ ] **Step 2: Commit and push**

```bash
git add -A
git commit -m "feat: add checkpoint management, monitoring, production dependencies

- Atomic checkpoint writes with retention policy
- Worker heartbeat files for monitoring
- Redis connection pooling
- Updated requirements.txt with all production dependencies"
git push origin main
```

---

## Stage 8: Update Tests for New Structure

### Task 40: Update all test imports to new package

**Files:**
- Modify: All test files in `tests/`

- [ ] **Step 1: Update imports in all test files**

Replace old imports:
```python
# Old:
from libs.actions import ArgmaxActionSelector
from algo_envs.dqn_gym_classic import DQNGymClassicNet

# New:
from rl_server.core.actions import ArgmaxActionSelector
from rl_server.algorithms.dqn.network import DQNGymClassicNet
```

- [ ] **Step 2: Add integration test for Redis pipeline**

Create `tests/integration/test_redis_pipeline.py`:

```python
import pytest
import torch
from rl_server.transport.redis_cache import RedisCache
from rl_server.utils.logging import Log

try:
    import fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False


@pytest.mark.integration
class TestRedisPipeline:
    @pytest.fixture
    def fake_cache(self):
        if not HAS_FAKEREDIS:
            pytest.skip("fakeredis not installed")
        log = Log("test")
        cache = RedisCache.__new__(RedisCache)
        cache.log = log
        cache.redis_config = {'ip': 'localhost', 'port': '6379', 'db': '0', 'pw': ''}
        cache.conn = fakeredis.FakeRedis()
        return cache

    def test_model_sync_roundtrip(self, fake_cache):
        model = torch.nn.Linear(4, 2)
        original_weight = model.weight.data.clone()
        fake_cache.set_train_version_model(1, model)
        version = fake_cache.get_train_version()
        assert version == 1

        fresh = torch.nn.Linear(4, 2)
        assert fake_cache.get_train_model(fresh)
        assert torch.allclose(original_weight, fresh.weight.data)
```

- [ ] **Step 3: Add integration test for grad aggregation**

Create `tests/integration/test_grad_aggregation.py`:

```python
import pytest
import torch
import numpy as np
from rl_server.algorithms.dqn.network import DQNGymClassicNet


@pytest.mark.integration
class TestGradAggregation:
    def test_accumulate_and_apply_grads(self):
        net = DQNGymClassicNet()
        params_before = [p.data.clone() for p in net.parameters()]

        # Generate two sets of gradients
        grads1 = [np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in net.parameters()]
        grads2 = [np.random.randn(*p.shape).astype(np.float32) * 0.01 for p in net.parameters()]

        # Accumulate
        accumulated = [g1 + g2 for g1, g2 in zip(grads1, grads2)]

        # Apply
        net.update_state(1, accumulated)
        params_after = [p.data.clone() for p in net.parameters()]

        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All PASS

- [ ] **Step 5: Commit and push**

```bash
git add tests/ -A
git commit -m "test: update tests for new package structure, add pipeline integration tests

- Update all test imports to rl_server.* paths
- Add Redis pipeline integration test
- Add gradient aggregation integration test
- All tests passing against new package structure"
git push origin main
```

---

## Summary

| Stage | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-5 | Fix critical bugs |
| 2 | 6-9 | YAML config system |
| 3 | 10-13 | Error handling, logging, shutdown |
| 4 | 14-24 | Comprehensive test suite |
| 5 | 25-34 | Package restructure |
| 6 | 35 | Gymnasium migration |
| 7 | 36-39 | Production hardening |
| 8 | 40 | Updated tests |

Total: 40 tasks, 8 git pushes.

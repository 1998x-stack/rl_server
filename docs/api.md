---
layout: default
title: API Reference
description: Core API reference for RL Server
---

# 📘 API Reference

## Algorithm Registry

```python
from rl_server.algorithms import create_net, create_agent, create_calculate, register

# Factory functions (lazy-load algorithms on first use)
net    = create_net('DQNGymClassic')
agent  = create_agent('DQNGymClassic', net, is_checker=False)
calc   = create_calculate('DQNGymClassic', net)

# Register a custom algorithm
register('MyAlgo', MyNetClass, MyAgentClass, MyCalculateClass)
```

## Base Classes

### AlgoBaseNet

```python
class AlgoBaseNet(nn.Module):
    def forward(self, states)       # Required: inference
    def update_state(self, version, grads_buffer)  # Required: apply gradients

def layer_init(linear, std=np.sqrt(2), bias_const=0.0, method='orthogonal') -> nn.Linear
```

### AlgoBaseAgent

```python
class AlgoBaseAgent:
    def sample_multi_envs(self, model_dict) -> List  # Required: N-env rollout
    def check_single_env(self) -> Dict               # Required: single-episode eval
```

### AlgoBaseCalculate

```python
class AlgoBaseCalculate:
    def generate_grads(self, samples, model_dict) -> Tuple[List, int]  # Required
```

## Action Selectors

```python
from rl_server.core.actions import (
    ArgmaxActionSelector,          # Greedy: argmax(scores)
    EpsilonGreedyActionSelector,   # ε-random, otherwise delegate
    ProbabilityActionSelector,     # Sample from distribution
    EpsilonTracker,                # Linear ε-decay scheduler
)
```

## Config

```python
from rl_server.config.loader import load_config

config = load_config('config/default.yaml', override='config/dev.yaml')
# Values like "${REDIS_HOST:localhost}" are interpolated from env vars
```

## Checkpoint

```python
from rl_server.utils.checkpoint import save_model, load_model

# Atomic save (tmp → rename, max version retention)
save_model(net, prefix='my_model', version='100', base_dir='models', max_versions=5)

# Load latest or specific version
version = load_model(net, prefix='my_model', version='100', base_dir='models')
```

## Transport (Redis)

```python
from rl_server.transport.redis_cache import RedisCache

cache = RedisCache(log, {'ip': 'localhost', 'port': 6379, 'db': 0, 'pw': ''})
cache.set_train_version_model(version=42, model=net)
cache.push_exps(experiences, sample_version=42)
cache.push_grads(gradients, grads_version=42, sample_version=41)
```

## Process Utilities

```python
from rl_server.utils.process import (
    setup_mp,              # spawn start method, limit BLAS threads
    setup_seed,            # fix all RNG seeds
    setup_signal_handlers, # SIGTERM/SIGINT → threading.Event
    should_exit,           # check shutdown flag
    write_heartbeat,       # write /tmp/rl_server/{type}_{id}_{pid}
    cleanup_heartbeat,     # remove heartbeat file
)
```

## Logging

```python
from rl_server.utils.logging import Log

log = Log('my_worker')
log.log_info('Training started')
log.log_exception()  # Logs current exception traceback
```

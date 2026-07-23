---
layout: default
title: Algorithm Registry
description: Lazy-loading algorithm registry and factory functions
---

# Algorithm Registry

The algorithm registry (`rl_server/algorithms/__init__.py`) is the central dispatch point for algorithm construction. It implements a **lazy-loading pattern** — algorithm subpackages are imported only on first use, keeping startup time and memory minimal.

## Architecture

```
_REGISTRY = {}  # env_name → (NetClass, AgentClass, CalculateClass)

create_net('DQNGymClassic')       → _lazy_load → DQNGymClassicNet()
create_agent('MujocoNormal', net) → _lazy_load → MujocoNormalAgent(net)
create_calculate('SACMujocoNormal', net) → _lazy_load → MujocoNormalQCalculate(net)
```

## Factory Functions

### `create_net(env_name)`

```python
net = create_net('MujocoNormal')
```

- Returns a fresh network instance
- Network parameters are initialized by the class constructor (random weights)
- Called by: all entrypoints, worker constructors

### `create_agent(env_name, net, is_checker=False)`

```python
agent = create_agent('MujocoNormal', net, is_checker=True)
```

- Returns an agent bound to the given network
- `is_checker=True` → agent creates a single evaluation environment (instead of N parallel envs)
- Called by: SamplerWorker, CheckerWorker, standalone sample.py/check.py

### `create_calculate(env_name, net)`

```python
calc = create_calculate('MujocoNormal', net)
```

- Returns a gradient calculator bound to the shared model
- Calculator creates its own mutable copy of the network for gradient computation
- Called by: TrainerWorker, GradsAggregator

### `register(env_name, net_cls, agent_cls, calculate_cls)`

Registers a triple of classes for a given environment name:

```python
register('DQNGymClassic', DQNGymClassicNet, DQNGymClassicAgent, DQNGymClassicCalculate)
```

### `set_device(env_name, device)`

Overrides `MODEL_CONFIG['DEVICE']` before network construction:

```python
set_device('MujocoNormal', 'cuda:0')
```

Uses `__import__` with `net_cls.__module__` introspection to find and modify the `MODEL_CONFIG_DEVICE` variable in the algorithm's module. This generic approach works for all registered algorithms without per-algorithm branches.

**Must be called before `create_net()`**, because the network's `__init__` places parameters on `MODEL_CONFIG['DEVICE']`.

## Lazy Loading

`_lazy_load()` is called automatically when a factory function encounters an unknown `env_name`.

### Registered Algorithms

| env_name | Module | Net | Agent | Calculate |
|----------|--------|-----|-------|-----------|
| `DQNGymClassic` | `rl_server.algorithms.dqn.network` | `DQNGymClassicNet` | `DQNGymClassicAgent` | `DQNGymClassicCalculate` |
| `DQNAtari` | `rl_server.algorithms.dqn.atari` | `AtariDQNNet` | `AtariDQNAgent` | `AtariDQNCalculate` |
| `MujocoNormal` | `rl_server.algorithms.ppo.mujoco_normal` | `MujocoNormalNet` | `MujocoNormalAgent` | `MujocoNormalCalculate` |
| `MicroRTS` | `rl_server.algorithms.ppo.microrts` | `MicroRTSNet` | `MicroRTSAgent` | `MicroRTSCalculate` |
| `MujocoBeta` | `rl_server.algorithms.ppo.mujoco_beta` | `MujocoBetaNet` | `MujocoBetaAgent` | `MujocoBetaCalculate` |
| `SACMujocoNormal` | `rl_server.algorithms.sac.mujoco_normal` | `MujocoNormalQNet` | `MujocoNormalQAgent` | `MujocoNormalQCalculate` |
| `TD3MujocoNormal` | `rl_server.algorithms.td3.mujoco_normal` | `MujocoNormalQNet` | `MujocoNormalQAgent` | `MujocoNormalQCalculate` |

### Error Handling

```python
if env_name == '...':
    ...
else:
    raise ValueError(f"Unknown algorithm: {env_name}")
```

`ImportError` from missing dependencies (e.g., `gym_microrts` for MicroRTS) is caught and re-raised as `ValueError` with the original message.

## Adding a New Algorithm

1. **Create a subpackage**: `rl_server/algorithms/my_algo/` with files defining `MyNet(AlgoBaseNet)`, `MyAgent(AlgoBaseAgent)`, `MyCalculate(AlgoBaseCalculate)`
2. **Add registration**: One `elif` branch in `_lazy_load()`:
   ```python
   elif env_name == 'MyAlgo':
       from rl_server.algorithms.my_algo import MyNet, MyAgent, MyCalculate
       register(env_name, MyNet, MyAgent, MyCalculate)
   ```
3. **Train**: `python -m rl_server.entrypoints.train --env-name MyAlgo`

### Network Module Requirements

Each algorithm's network module must export a module-level `MODEL_CONFIG_DEVICE` variable (used by `set_device()`). This is typically:

```python
MODEL_CONFIG_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

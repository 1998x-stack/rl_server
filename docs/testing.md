---
layout: default
title: Testing
description: Test suite structure, test layers, and test conventions
---

# Testing

RL Server has 174+ tests organized into three layers: unit, smoke, and integration.

## Test Layers

```bash
# Full suite
pytest tests/ -v

# By layer
pytest tests/unit/ -v            # Fast, isolated — no I/O
pytest tests/smoke/ -v           # Algorithm forward + update validation
pytest tests/integration/ -v     # Multi-component pipelines (fakeredis)
```

Conftest: `tests/conftest.py` — shared fixtures (not read during exploration).

## Unit Tests — `tests/unit/`

15+ files covering individual components in isolation.

| Area | Key Files | What They Test |
|------|-----------|----------------|
| **Registry** | `test_rl_server_package.py` | Factory functions (`create_net`, `create_agent`, `create_calculate`), lazy loading, error handling for unknown env names |
| **Config** | `test_config_loader.py` | YAML loading, env var interpolation, deep merge, override files |
| **Checkpoint** | `test_checkpoint.py` | `save_model` / `load_model` roundtrip, version retention, atomic writes, missing file handling |
| **DQN** | `test_dqn.py` | DQN network forward pass, agent sampling, calculator gradients |
| **PPO** | `test_ppo.py` | PPO normal GPU/CPU forward, agent sample, calculator grads |
| **Actions** | `test_actions.py` | EpsilonGreedy, Argmax, Probability selectors, EpsilonTracker |
| **Buffers** | `test_buffer.py` | ExperienceBuffer capacity, sampling, ordering |
| **Transport** | `test_redis.py` | RedisCache push/pop, serialization roundtrip, retry behavior |
| **NoisyNet** | `test_noisy.py` | NoisyLinear forward shapes, noise sampling |
| **Process** | `test_heartbeat.py` | Heartbeat write/cleanup, signal handlers |
| **Run PPO** | `test_run_ppo.py` | Smoke test for `run_ppo_swimmer.py` |

### What Unit Tests Verify

- **Network**: Forward pass output shape and dtype
- **Agent**: Sample/check return formats and rollouts
- **Calculator**: Gradient generation produces expected structure
- **Config**: Correct loading, merging, and interpolation
- **Checkpoint**: File I/O correctness, version handling
- **Transport**: Serialization and Redis operations via fakeredis

## Smoke Tests — `tests/smoke/`

One file testing all 6 algorithms at a basic functional level.

| Test | What It Verifies |
|------|------------------|
| `test_algorithm_smoke.py` | Each registered algo: net forward pass + `update_state` with generated grads |

Smoke tests ensure that after adding a new algorithm, the basic train loop (forward + backward + apply) works without environment dependencies.

## Integration Tests — `tests/integration/`

4 files testing multi-component interactions.

| File | What It Tests |
|------|---------------|
| `test_pipeline.py` | End-to-end local multiprocess pipeline with fakeredis |
| `test_redis_pipeline.py` | Distributed-mode pipeline (sampler → Redis → trainer → Redis → grads) |
| `test_grad_aggregation.py` | Gradient accumulation logic (multi-batch → averaged update) |

Integration tests use `fakeredis` — no Redis server process needed. This makes them suitable for CI environments without Redis.

## Test Configuration

From `pytest.ini`:

```ini
[pytest]
timeout = 30           # 30-second timeout per test
markers =
    integration: marks tests as integration tests
```

All tests must complete within 30 seconds. Redis-dependent tests use `fakeredis` which operates in-memory, so there's no network dependency.

## Running Specific Tests

```bash
# Pattern matching
pytest tests/ -v -k "ppo"       # All tests with "ppo" in name
pytest tests/ -v -k "dqn"       # All tests with "dqn" in name

# By layer + pattern
pytest tests/unit/ -v -k "checkpoint"  # Unit tests about checkpointing

# Single file
pytest tests/unit/test_dqn.py -v
```

## Adding Tests for a New Algorithm

When adding a new algorithm, create at minimum:

1. **Unit test file**: `tests/unit/test_<algo>.py` — network forward, agent sample/check, calculator grads
2. **Smoke test entry**: The algorithm will be covered by `tests/smoke/test_algorithm_smoke.py` if registered

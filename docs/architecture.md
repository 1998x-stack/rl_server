---
layout: default
title: Architecture
description: RL Server architecture and data flow
---

# 🏗 Architecture

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        MAIN PROCESS                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐ │
│  │ load_config │───▶│ create_net  │───▶│ while not exit:       │ │
│  │ + override  │    │ share_mem() │    │   grads ← grads_queue │ │
│  └─────────────┘    └─────────────┘    │   accumulate          │ │
│                                        │   update_state()      │ │
│                                        │   TRAIN_VERSION += 1  │ │
│                                        └──────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       │ mp.Queue
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   SAMPLERS   │ │   TRAINERS   │ │   CHECKER    │
│   (N proc)   │ │   (M proc)   │ │   (1 proc)   │
│              │ │              │ │              │
│ sample_multi │ │ generate     │ │ check_single │
│ _envs() ─────┼▶│ _grads() ────┼▶│ _env() ──────┼──▶ TensorBoard
│              │ │              │ │              │
│ [Gym envs]   │ │ [Calculate]  │ │ [Eval env]   │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Three-Component Abstraction

Every algorithm implements three classes:

### AlgoBaseNet (nn.Module)
- `forward(states)` → actions — inference in all workers
- `update_state(version, grads_buffer)` — apply aggregated gradients in main process

### AlgoBaseAgent
- `sample_multi_envs(model_dict)` → experiences — rollout N parallel environments
- `check_single_env()` → metrics — single-episode evaluation

### AlgoBaseCalculate
- `generate_grads(samples, model_dict)` → gradients — compute loss and gradients

## Data Flow

```
Sampler Workers                  Trainer Workers
┌────────────┐                   ┌────────────┐
│   Agent    │ ──experiences──▶  │ Calculator │ ──gradients──▶
│ (rollout)  │    sample_queue   │ (loss+grad)│   grads_queue
└────────────┘                   └────────────┘
      △                                               │
      │                                               ▼
      │              Main Process                     │
      └──── shared ── [  Network  ] ◀── update ──────┘
              model      (PyTorch)     (avg grads)
```

- `model_dict` is an `mp.Manager().dict()` shared across processes
- Keys: `is_exit` (bool), `TRAIN_VERSION` (int)
- Each gradient batch carries the model version it was computed from

## Deployment Modes

### Local Multiprocess
Uses `torch.multiprocessing.Queue` for IPC. Network weights shared via `share_memory()`. Best for single-machine training.

### Redis Distributed
Each worker type runs as an independent process connecting to Redis. Communication via Redis lists (LPUSH/BRPOP). Model sync through compressed pickle state_dicts.

### Gradient Aggregation Server
A standalone daemon that aggregates gradients from multiple sources and broadcasts updated models.

## Resilience Patterns

- **Graceful shutdown**: SIGTERM → set exit flag → workers complete iteration → save checkpoint → exit
- **Atomic checkpoints**: Write to `.tmp` → `os.rename()` — no corruption on crash
- **Redis retry**: Exponential backoff (1s, 2s, 4s) on ConnectionError
- **BRPOP timeout**: 5-second timeout prevents infinite blocking

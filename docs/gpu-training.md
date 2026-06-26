---
layout: default
title: GPU Training
description: GPU training setup, architecture, known issues, and fixes
---

# GPU Training

## Quick Start

```bash
# GPU training on physical GPU 2 (A100-80GB)
CUDA_VISIBLE_DEVICES=2 python -m rl_server.entrypoints.train \
  --env-name MujocoNormal \
  --config config/default.yaml \
  --override <(echo 'training: {device: "cuda:0"}')

# CPU training (default)
python -m rl_server.entrypoints.train --env-name MujocoNormal
```

`CUDA_VISIBLE_DEVICES` remaps physical GPUs — with `CUDA_VISIBLE_DEVICES=2`, `cuda:0` in PyTorch points to physical GPU 2. Configure `training.device` in YAML to select the GPU.

## Architecture

### Device selection flow

```
config YAML           algorithms/__init__.py      algorithm module
training:             set_device(env, device)     MODEL_CONFIG['DEVICE']
  device: "cuda:0" ──────────────────────────────► = torch.device("cuda:0")
                                                         │
                         create_net(env_name) ◄──────────┘
                              │
                         Net.__init__()
                         self.to(MODEL_CONFIG['DEVICE'])
```

`set_device()` uses `net_cls.__module__` introspection to find the right `MODEL_CONFIG` dict generically — works for all registered algorithms without per-algorithm branches.

### CPU vs GPU: process model divergence

| Aspect | CPU mode | GPU mode |
|---|---|---|
| `share_memory()` | Called on main net | Skipped (CUDA tensors can't be shared) |
| Sampler sync | Reads shared memory directly | `load_state_dict()` each rollout |
| Trainer sync | `load_state_dict()` each `generate_grads` | Same (already GPU-safe) |
| Checker sync | `load_state_dict()` each eval loop | Same (already GPU-safe) |
| Worker net creation | Reuses main process model | Each worker creates own net via `create_net` |

**Why `load_state_dict` works across processes:** `nn.Module.state_dict()` always returns CPU tensors regardless of the model's device. Loading CPU state_dict into a GPU model in another process works because `load_state_dict` places tensors on the target model's device.

### Gradient flow (GPU-safe)

```
Trainer worker (GPU)                  Main process (GPU)
┌──────────────────────┐              ┌──────────────────────┐
│ calculate_net (GPU)  │              │ train_net (GPU)      │
│ loss.backward()      │              │                      │
│ grads = [p.grad      │              │ update_state(ver,    │
│   .cpu().numpy()     │──mp.Queue──►│   grads_buffer)      │
│   for p in params]   │              │ param.grad =         │
│                      │              │   torch.FloatTensor  │
└──────────────────────┘              │   (grad).to(device)  │
                                      └──────────────────────┘
```

Gradients are transferred as CPU numpy arrays via `mp.Queue` (pickled). The main process reconstructs them as tensors on the correct device.

### Checkpoint resume on GPU

`load_model()` loads to CPU by default (`map_location=None`), then `model.load_state_dict()` moves weights to the GPU model:

```python
checkpoint = torch.load(file_path)           # CPU tensors
model.load_state_dict(checkpoint['state_dict'])  # CPU → GPU
```

This works correctly with `CUDA_VISIBLE_DEVICES` — no special handling needed.

## Fixed Issues

### 1. `and False` GPU guard (removed)

All 5 algorithm modules had:
```python
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')
#                                                                     ^^^^^^^^
```
The `and False` forced CPU-only. Removed; GPU now activates when CUDA is available AND config requests it.

### 2. Version type coercion

`load_model()` returns the checkpoint version as a string (stored that way in `.td` files). The main loop does `current_train_version += 1` which fails with `TypeError` if the version is a string. Fixed: `current_train_version = int(current_train_version)` after non-None load.

### 3. TensorBoard flush

`SummaryWriter` buffers writes and only flushes periodically (~120s) or on close. Checker loops are short — most runs never triggered a flush, leaving 0-byte event files. Fixed: explicit `writer.flush()` after each scalar write in checker worker.

### 4. Training progress visibility

The main loop had no version logging — impossible to tell if training was making progress. Fixed: log every 10 versions and on every `checkpoint_interval` save.

### 5. Periodic checkpoint (local mode)

Previously, checkpoints were only saved on SIGTERM. If the process crashed, all progress was lost. Fixed: `save_model()` is now called every `checkpoint_interval` versions (default: 100) during training.

## MicroRTS GPU notes

MicroRTS has a CNN backbone and follows a slightly different GPU pattern than the MuJoCo MLPs:

| Component | Device | Rationale |
|---|---|---|
| `MicroRTSNet` (main process) | CPU | `share_memory()` not called; weights synced via `state_dict()` |
| `MicroRTSNet` (trainer) | GPU | `calculate_net.to(device)` in `MicroRTSCalculate.__init__` |
| `update_state` | CPU | `torch.FloatTensor(grad)` — no `.to(DEVICE)`, params stay CPU |
| `generate_grads` tensors | GPU | All intermediate tensors need explicit `.to(device=self.device)` |

The CNN's `forward` does `x.permute((0, 3, 1, 2))` expecting `[N, H, W, C]` input. Ensure `reset()` returns `[N, H, W, C]` directly (gym_microrts 0.3.2 behavior — do NOT do `reset()[0]`).

## Potential Issues (monitoring)

### Multi-GPU collision

Without `CUDA_VISIBLE_DEVICES`, PyTorch defaults to `cuda:0` (physical GPU 0). On multi-tenant machines, this conflicts with other workloads. Always set `CUDA_VISIBLE_DEVICES` explicitly.

### VRAM sizing

| Model | obs_dim × act_dim | Hidden | VRAM (approx) |
|---|---|---|---|
| Pusher | 23 × 7 | 128 | 500 MiB |
| HalfCheetah | 17 × 6 | 64 | 300 MiB |
| Humanoid | 376 × 17 | 512 | 2-3 GiB |
| Ant | 27 × 8 | 256 | 1 GiB |
| MicroRTS | 10×10×27 | CNN+256 | 500 MiB |

These are per-worker. In local mode with 1 trainer, expect ~1× model VRAM. The sampler and checker do CPU inference (envs are CPU-bound), so they don't consume GPU memory.

### Sampler CPU inference

On GPU, the sampler syncs model weights but runs `sample_multi_envs` on CPU. The network is tiny (MLP), so GPU inference overhead (CPU→GPU transfer, CUDA kernel launch) would outweigh any benefit. This is intentional — environments are CPU-bound, not GPU-bound.

### JVM memory (MicroRTS only)

MicroRTS starts a JVM per spawned worker (~200-500 MB each). With 2 samplers + 1 checker, expect ~1-2 GB extra RSS. Not a GPU issue, but relevant for resource planning.

### Checkpoint version retention

`max_versions` controls how many `.td` files are kept per model prefix. Old files are deleted on each save. The `checkpoint_interval` config controls save frequency. Default: save every 100 versions, keep 10000 files.

## Testing GPU mode

```bash
# Quick smoke: single forward/backward on GPU
python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
m = torch.nn.Linear(10, 10).cuda()
x = torch.randn(3, 10).cuda()
loss = m(x).sum()
loss.backward()
print('GPU forward/backward OK, shape:', x.shape)
print('Device:', next(m.parameters()).device)
"

# Verify rl_server GPU setup
python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import rl_server.algorithms.ppo.mujoco_normal as ppo
ppo.MODEL_CONFIG['DEVICE'] = torch.device('cuda:0')
net = ppo.MujocoNormalNet()
print('Net device:', next(net.parameters()).device)
x = torch.randn(2, ppo.TRAIN_ENVS['Swimmer'].OBS_DIM).cuda()
y = net(x)
print('Forward shape:', y.shape)
"
```

## Config reference

```yaml
# GPU training override (e.g., config/gpu.yaml)
training:
  device: "cuda:0"          # PyTorch device string

# Usage:
# CUDA_VISIBLE_DEVICES=2 python -m rl_server.entrypoints.train --override config/gpu.yaml
```

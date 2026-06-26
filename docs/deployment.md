---
layout: default
title: Deployment
description: Deployment guide — Docker Compose, environment setup, and known gotchas
---

# Deployment

## Docker Compose (Distributed Mode)

```bash
docker compose up -d          # Redis + grads + 1 sampler
docker compose --profile monitor up -d  # + checker (TensorBoard)
docker compose up -d --scale sampler=4  # 4 sampler workers
```

Services communicate via Redis. Environment variables (`REDIS_MODEL_HOST=redis` etc.) route all Redis traffic to the `redis` service container. Config interpolation (`${REDIS_MODEL_HOST:localhost}`) picks these up automatically.

**Scaling reference:**

| Scale | Samplers | Command |
|---|---|---|
| Dev | 1–2 | `docker compose up -d` |
| Medium | 4–8 | `docker compose up -d --scale sampler=4` |
| High | 16–64 | Deploy sampler across multiple hosts, point `REDIS_*_HOST` to a shared Redis |

### Build

```bash
docker build -t rl_server:latest .
```

The image is ~1.5 GB (PyTorch + MuJoCo deps). MuJoCo rendering libs (`libgl1-mesa-glx`, `libglib2.0-0`, `libosmesa6`) are included for headless environments.

---

## Environment Setup

### MuJoCo (Gymnasium) — works out of the box

```bash
pip install -r requirements.txt
pip install gymnasium[mujoco]      # Swimmer, HalfCheetah, Ant, Hopper, ...
python -m rl_server.entrypoints.train --env-name MujocoNormal
```

No additional system dependencies needed. MuJoCo ships its own engine via `mujoco` PyPI package.

### MicroRTS — requires gym_microrts 0.3.2 (PyPI)

MicroRTS uses the old hierarchical-action `MicroRTSVecEnv` API, NOT the grid-based `MicroRTSGridModeVecEnv` from git-HEAD.

```bash
pip install "gym-microrts==0.3.2" dacite
```

The package needs numpy compat patches (see gotcha #1). The rl_server code (`rl_server/algorithms/ppo/microrts.py`) was written for this API — no code changes needed.

```bash
python -m rl_server.entrypoints.train --env-name MicroRTS
```

**Why 0.3.2, not git-HEAD:** The gym_microrts git-HEAD introduced `MicroRTSGridModeVecEnv` which uses a fundamentally different grid-based action space (action per map cell `[N, H*W, dim]`). rl_server's PPO MicroRTS implementation uses hierarchical actions (select unit → select action). These are architecturally incompatible — the network architecture, action selection, and experience format would all need rewriting.

---

## MicroRTS Gotchas (gym_microrts 0.3.2)

### 1. numpy 2.0 compatibility

gym_microrts 0.3.2 uses `np.int` and `np.float` which were removed in numpy 2.0. The `rl_server/algorithms/ppo/microrts.py` file includes a shim at the top of the file:

```python
import numpy
numpy.int = int
numpy.float = float
```

Additionally, the installed gym_microrts source must be patched for spawned subprocesses (which start fresh Python interpreters):

```bash
python3 -c "
import gym_microrts, os
pkg_dir = os.path.dirname(gym_microrts.__file__)
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            content = open(path).read()
            if 'np.int' in content or 'np.float' in content:
                content = content.replace('np.int', 'int')
                content = content.replace('np.float', 'float')
                open(path, 'w').write(content)
                print(f'Patched: {path}')
"
```

Important: replace `np.int` with `int` but keep `np.int32`, `np.float32` etc. — only the bare `np.int`/`np.float` were removed.

### 2. Old gym step() returns 4 values

Old gym's `env.step()` returns `(obs, rewards, dones, infos)` — 4 values, not 5 (no `truncated`). The microrts.py code unpacks 4 values:

```python
next_obs, rs, done, _ = self.env.step(action.T)
```

### 3. reset() returns numpy array directly

gym_microrts 0.3.2 `MicroRTSVecEnv.reset()` returns a numpy array `[N, H, W, C]` directly — NOT a tuple containing a list. Code must NOT do `reset()[0]`:

```python
self.obs = self.env.reset()  # [N, H, W, C] directly
```

### 4. GPU training: device consistency

MicroRTS uses a CNN backbone (Conv2d). For GPU training:

- `MicroRTSNet` parameters stay on CPU (no `.to(device)` in init — matches MuJoCo pattern)
- `MicroRTSCalculate.calculate_net` moved to GPU via `.to(self.device)` for forward/backward
- `generate_grads` returns CPU numpy arrays (`param.grad.data.cpu().numpy()`)
- `update_state` assigns CPU tensors (`torch.FloatTensor(grad)`, no `.to(DEVICE)`)
- All intermediate tensors in `generate_grads` must explicitly call `.to(device=self.device)`

### 5. MaskedCategorical NaN handling

When ALL actions are masked for a batch element, logits become all `-inf` → softmax produces NaN. The `MaskedCategorical._apply_action_masks` includes two guards:

```python
# 1. Replace NaN in input logits with large negative value
clamped_logits = torch.nan_to_num(clamped_logits, nan=-1e9)

# 2. If all actions masked, fall back to original logits
all_masked = ~self.action_masks.any(dim=-1)
if all_masked.any():
    masked[all_masked] = clamped_logits[all_masked]
```

### 6. Slow rollout speed

MicroRTS with coacAI is CPU-intensive. Each rollout: 512 steps × 8 envs, each step involves full Java MicroRTS simulation. Typical throughput: ~0.5 versions/min on CPU. GPU helps trainer but not samplers (envs are CPU-bound).

### 7. JVM per spawned worker

Each spawned worker starts its own JVM via JPype. With 2 samplers + 1 checker + 1 trainer, expect 4 JVMs consuming ~200-500 MB each = 1-2 GB extra RSS.

### 8. grid API (git-HEAD) is architecturally incompatible

gym_microrts git-HEAD (`MicroRTSGridModeVecEnv`) uses grid-based actions per map cell `[N, H*W, action_dim]`. rl_server's PPO MicroRTS uses hierarchical actions (select unit → select action). These are incompatible at the design level — porting requires ~200+ lines rewritten (network architecture, action selection, experience format, loss computation). Use PyPI 0.3.2 instead.

---

## GitHub Pages

Site: `https://1998x-stack.github.io/rl_server/`
Source: `/docs` directory on `main` branch
Theme: Jekyll Cayman (`remote_theme: pages-themes/cayman@v0.2.0`)

### Setup reference

```bash
# Enable Pages via API (repo must be public on Free plan)
gh api repos/1998x-stack/rl_server/pages -X POST \
  -F "source[branch]=main" \
  -F "source[path]=/docs"

# Check build status
gh api repos/1998x-stack/rl_server/pages/builds/latest \
  --jq '{status: .status, error: .error.message}'

# Site updates automatically on push to main
```

**Gotcha:** GitHub Pages on private repos requires Pro/Team/Enterprise plan. Make the repo public first if you're on Free (`gh api repos/{owner}/{repo} -X PATCH -F "private=false"`).

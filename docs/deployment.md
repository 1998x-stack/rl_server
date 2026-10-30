---
layout: default
title: Deployment
description: Deployment guide — Docker Compose, environment setup, and known gotchas
---

# Deployment

## Docker Compose (Distributed Mode)

```bash
docker compose up -d          # Redis + grads + trainer + sampler
docker compose --profile monitor up -d  # + checker (TensorBoard)
docker compose up -d --scale sampler=4  # 4 sampler workers
```

Services communicate via Redis. Environment variables (`REDIS_MODEL_HOST=redis` etc.) route all Redis traffic to the `redis` service container. Config interpolation (`${REDIS_MODEL_HOST:localhost}`) picks these up automatically.

Hosts **and ports** are interpolated: `REDIS_MODEL_HOST`/`REDIS_MODEL_PORT`, `REDIS_EXPS_HOST`/`REDIS_EXPS_PORT`, `REDIS_GRADS_HOST`/`REDIS_GRADS_PORT` (ports default to `6379`). Point the three workers at a Redis on a non-default port when `6379` is taken locally:

```bash
docker run -d --name rl_server_redis -p 127.0.0.1:6479:6379 redis:7-alpine
export REDIS_MODEL_PORT=6479 REDIS_EXPS_PORT=6479 REDIS_GRADS_PORT=6479 RL_RUN_ID=local_dev
python -m rl_server.entrypoints.grads   --env-name DQNGymClassic &   # then trainer + sample
```

Set the same unique `RL_RUN_ID` for every service in a training job. `queues.len_sample_queue` bounds pending Redis experience batches; samplers wait when that queue reaches capacity. Scale trainers with samplers to keep the queue from staying full.

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

### Reproducible local env (no sudo, no Docker)

```bash
bash scripts/bootstrap_env.sh --all      # base+dev, MuJoCo, MicroRTS (portable JDK 17), Atari
bash scripts/bootstrap_env.sh            # base + dev deps only
source .venv/bin/activate
```

The script creates a Python 3.11 `.venv` (gitignored), installs the CPU PyTorch wheel, `requirements-dev.txt`, and optionally `gymnasium[mujoco]` / `gym-microrts==0.3.2`. It is idempotent, so re-running it only fills gaps. Network flakiness on the ~185 MB JDK download is handled with resume + retry.

### MuJoCo (Gymnasium) — works out of the box

```bash
pip install -r requirements.txt
pip install gymnasium[mujoco]      # Swimmer, HalfCheetah, Ant, Hopper, ...
python -m rl_server.entrypoints.train --env-name MujocoNormal
```

No additional system dependencies needed. MuJoCo ships its own engine via `mujoco` PyPI package.

### Atari (ALE) — DQNAtari

```bash
pip install "gymnasium[atari]" opencv-python-headless   # ale-py >=0.10 ships the ROMs
python -m rl_server.entrypoints.train --env-name DQNAtari
```

`ale-py` 0.12 bundles the ROMs, so AutoROM is not required. Frame preprocessing uses OpenCV — without
`opencv-python-headless` every worker dies at startup with `ModuleNotFoundError: cv2` (and, since the
worker health check, the main log reports `worker died ... exitcode=1` instead of running silently).

### MicroRTS — requires gym_microrts 0.3.2 (PyPI)

MicroRTS uses the old hierarchical-action `MicroRTSVecEnv` API, NOT the grid-based `MicroRTSGridModeVecEnv` from git-HEAD.

```bash
pip install "gym-microrts==0.3.2" "gym==0.23.1" dacite
```

Requirements beyond pip, all handled by `scripts/bootstrap_env.sh --microrts`:

- **A JDK on the machine.** JPype starts a JVM per worker; with no JDK you get `JVMNotFoundException` at env construction. The bootstrap script unpacks a portable Temurin 17 into `~/.local/opt/jdk17` (no sudo) and links `java`/`javac` into `~/.local/bin`. Set `JAVA_HOME=~/.local/opt/jdk17` if JPype cannot locate the JVM.
- **`gym==0.23.1`.** gym_microrts 0.3.2 imports `gym.envs.registry.env_specs` at module load; gym 0.24+ turned the registry into a plain dict, so 0.26.x raises `AttributeError` on `import gym_microrts`. 0.23.1 is the last release with the old registry, and it also matches the 4-tuple `step()` the algorithm expects.
- **numpy compat patches** (see gotcha #1).

The rl_server code (`rl_server/algorithms/ppo/microrts.py`) was written for this API — no code changes needed.

```bash
python -m rl_server.entrypoints.train --env-name MicroRTS
```

**Why 0.3.2, not git-HEAD:** The gym_microrts git-HEAD introduced `MicroRTSGridModeVecEnv` which uses a fundamentally different grid-based action space (action per map cell `[N, H*W, dim]`). rl_server's PPO MicroRTS implementation uses hierarchical actions (select unit → select action). These are architecturally incompatible — the network architecture, action selection, and experience format would all need rewriting.

---

## MicroRTS Gotchas (gym_microrts 0.3.2)

### 1. numpy 2.0 compatibility

gym_microrts 0.3.2 uses bare `np.int` / `np.float`, removed in numpy 2.0, in three files:

- `gym_microrts/envs/vec_env.py`
- `gym_microrts/envs/local_agent_env.py`
- `gym_microrts/envs/global_agent_env.py`

Patch the **installed package source** (`scripts/bootstrap_env.sh --microrts` does this automatically and is idempotent). A runtime shim inside `rl_server/algorithms/ppo/microrts.py` would not be enough: sampler workers are separate interpreters, so the aliases must live in the package that they import.

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

Every MicroRTS env needs a JDK: JPype starts a JVM inside each spawned worker, so a machine without `java` fails at env construction (`JVMNotFoundException`). `scripts/bootstrap_env.sh --microrts` installs a portable JDK 17 without sudo; otherwise `export JAVA_HOME=/path/to/jdk-17`.

With 2 samplers + 1 checker + 1 trainer, expect 4 JVMs consuming ~200-500 MB each = 1-2 GB extra RSS.

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

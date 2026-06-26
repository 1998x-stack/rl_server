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

### MicroRTS — requires manual JAR build

MicroRTS needs a compiled Java engine (`microrts.jar`). The gym_microrts package expects this JAR but cannot build it automatically in all environments.

**Known gotchas (see below):** Python version constraint, old-gym API changes, JAR compilation, Java API mismatch.

---

## MicroRTS: Full Setup Walkthrough

### Step 1 — Install gym-microrts

```bash
# Must ignore Python version constraint (package says <3.10 but works on 3.10+)
pip install --ignore-requires-python \
    "gym-microrts @ git+https://github.com/Farama-Foundation/gym-microrts.git"
```

The package installs old `gym` (0.23.x) as a dependency alongside `gymnasium`. They coexist — rl_server uses gymnasium for the training loop; gym_microrts uses old gym for its vec_env wrapper.

### Step 2 — Install JDK and compile MicroRTS JAR

```bash
# Download JDK (no sudo needed)
curl -sL "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.26%2B4/OpenJDK11U-jdk_x64_linux_hotspot_11.0.26_4.tar.gz" \
    -o /tmp/jdk11.tar.gz
mkdir -p /tmp/jdk11 && tar xzf /tmp/jdk11.tar.gz -C /tmp/jdk11 --strip-components=1

# Clone MicroRTS source
git clone --depth 1 https://github.com/Farama-Foundation/MicroRTS.git /tmp/MicroRTS

# Compile with classpath including all lib JARs
JAVA_HOME=/tmp/jdk11 PATH=/tmp/jdk11/bin:$PATH
cd /tmp/MicroRTS
CP="lib/jdom.jar:lib/minimal-json-0.9.4.jar:lib/weka.jar"
for jar in lib/bots/*.jar; do CP="$CP:$jar"; done

mkdir -p /tmp/microrts-build
find src -name "*.java" | grep -v "/test/" > /tmp/sources.txt
javac -cp "$CP" -d /tmp/microrts-build @/tmp/sources.txt

# Bundle into fat JAR
cd /tmp/microrts-build
jar xf /tmp/MicroRTS/lib/jdom.jar
jar xf /tmp/MicroRTS/lib/minimal-json-0.9.4.jar
jar xf /tmp/MicroRTS/lib/weka.jar
for jar in /tmp/MicroRTS/lib/bots/*.jar; do jar xf "$jar"; done
jar cf /tmp/microrts.jar .

# Place where gym_microrts expects it
MICRORTS_PKG=$(python -c "import gym_microrts; import os; print(os.path.dirname(gym_microrts.__file__))")
cp /tmp/microrts.jar "$MICRORTS_PKG/microrts/microrts.jar"
```

### Step 3 — Code changes for gym-microrts API compatibility

The rl_server MicroRTS implementation was written for the old `gym_microrts.envs.vec_env.MicroRTSVecEnv` API. The current gym_microrts uses `MicroRTSGridModeVecEnv` with a different constructor.

**Required changes in `rl_server/algorithms/ppo/microrts.py`:**

```python
# 1. Import: MicroRTSVecEnv → MicroRTSGridModeVecEnv
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# 2. Constructor: num_envs → num_selfplay_envs=0 + num_bot_envs=N
#    map_path  → map_paths (list)
#    Add autobuild=False (prevents JAR deletion)
self.env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=0,
    num_bot_envs=self.num_envs,
    autobuild=False,
    ai2s=[microrts_ai.coacAI for _ in range(self.num_envs)],
    map_paths=['maps/10x10/basesWorkers10x10.xml'],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
```

### Step 4 — Run

```bash
python -m rl_server.entrypoints.train --env-name MicroRTS
```

---

## MicroRTS Gotchas

### 1. Python version constraint

Package metadata says `Requires-Python: <3.10`. Use `--ignore-requires-python` with pip, or install from GitHub directly. The code works fine on 3.10+.

### 2. Old gym vs gymnasium

gym_microrts imports old `gym` (unmaintained). This produces deprecation warnings about NumPy 2.0 and distutils. These are cosmetic — the environments work. rl_server uses gymnasium everywhere else; the two packages coexist without conflict.

### 3. MicroRTSVecEnv → MicroRTSGridModeVecEnv

Old API: `MicroRTSVecEnv(num_envs=N, map_path='...', ai2s=[...])`
New API: `MicroRTSGridModeVecEnv(num_selfplay_envs=0, num_bot_envs=N, map_paths=['...'], ai2s=[...])`

Key differences:
- `num_envs` split into `num_selfplay_envs` + `num_bot_envs`
- `ai2s` is for bot environments only (`assert num_bot_envs == len(ai2s)`)
- `map_path` (str) → `map_paths` (list of str)
- New: `autobuild` (bool), `partial_obs`, `frame_skip`, `cycle_maps`

### 4. autobuild deletes your JAR

`MicroRTSGridModeVecEnv(autobuild=True)` (default) will:
1. Delete `microrts.jar`  
2. Run `bash build.sh` (which does not exist)
3. Fail because the JAR is gone

Always pass `autobuild=False` and manage the JAR yourself.

### 5. No JDK in standard environments

The `build.sh` approach requires both JDK (`javac`) and Ant. Most deployments have only JRE. The JDK is needed only for the one-time JAR build — not at runtime. Pre-build the JAR and bake it into your Docker image.

### 6. Java API version mismatch

The latest MicroRTS `HEAD` may differ from what gym_microrts expects. If you see `AttributeError: 'JNIGridnetVecClient' object has no attribute 'getUnitLocationMasks'`, the JAR was compiled from a MicroRTS version that doesn't match the gym_microrts wrapper.

**Workaround:** compile MicroRTS from a specific commit that matches your gym_microrts version. Check the gym_microrts source for which MicroRTS API it uses:

```bash
grep -r "UnitTypeTable\|getUnitLocationMasks\|JNIGridnetVecClient" \
    $(python -c "import gym_microrts; print(gym_microrts.__path__[0])")
```

### 7. JVM startup in subprocesses

Each spawned worker (sampler, trainer, checker) starts its own JVM via JPype. This means:
- N samplers × 1 JVM = N JVMs
- Each JVM consumes ~200–500 MB heap
- With `num_envs=16` per sampler and 2 samplers, expect ~1–2 GB extra memory

### 8. CoacAI bot JAR

The CoacAI opponent bot is bundled as `lib/bots/Coac.jar`. This JAR is in the gym_microrts package and included in the fat JAR we build. The `microrts_ai.coacAI` constant maps to this bot — it still exists in current gym_microrts.

### 9. Observation shape: reset() return type changed

Old API: `reset()` returns `(obs_list,)` — a tuple containing a list of N arrays, each `[H, W, C]`.
New API: `reset()` returns a single numpy array `[N, H, W, C]`.

Code like `self.obs = self.env.reset()[0]` breaks silently — it slices the FIRST env's observation `[H, W, C]` instead of the full batch `[N, H, W, C]`. The network then gets a 3D tensor when it expects 4D `[N, H, W, C]`.

**Fix:**
```python
obs_result = self.env.reset()
self.obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
```

### 10. step() action shape: grid-based vs hierarchical

This is a **fundamental architectural difference** between the two APIs:

| Aspect | Old API (MicroRTSVecEnv) | New API (MicroRTSGridModeVecEnv) |
|---|---|---|
| Action space | Hierarchical: select unit → select action | Grid-based: action per map cell |
| Action shape | `[num_action_components, num_envs]` | `[num_envs, H*W, action_dim]` |
| Mask API | `getUnitLocationMasks()`, `getUnitActionMasks(units)` | `getMasks(player)` returns `[N, H, W, 1+A+P]` |

The rl_server PPO MicroRTS implementation (`microrts.py`) is architecturally coupled to the old hierarchical action space. **Porting it to the grid-based API requires rewriting:**
- Network forward pass (logit shapes)
- `get_action` / `_get_single_action` (action selection logic)
- `sample_multi_envs` / `check_single_env` (experience storage format)
- `MicroRTSCalculate.generate_grads` (loss computation from experiences)

This is a ~200+ line refactor, not a drop-in fix. The old and new action semantics are incompatible at the design level.

### 11. Mask caching pattern

The new API's `getMasks(0)` calls Java via JPype on every invocation — expensive across hundreds of steps. The monkey-patch wrappers in the fixed microrts.py implement a cache-invalidate pattern:

```python
# After each env.step(), invalidate the mask cache
self.env.invalidate_masks()

# Next call to getUnitLocationMasks() or getUnitActionMasks()
# triggers a fresh getMasks(0), caches the result, and indexes into it
```

This avoids duplicate JNI calls within a single step (both `getUnitLocationMasks` and `getUnitActionMasks` read from the same `getMasks(0)` result).

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

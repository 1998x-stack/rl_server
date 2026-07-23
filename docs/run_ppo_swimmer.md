---
layout: default
title: Single-Process PPO Quick Test
description: run_ppo_swimmer.py — a self-contained, single-process PPO training script
---

# Single-Process PPO Quick Test

`run_ppo_swimmer.py` is a standalone script for quickly testing PPO training on the Swimmer-v4 environment without the multiprocess infrastructure. It runs the full training loop in a single process.

## Purpose

- **Fast iteration**: No `mp.Queue`, no subprocess orchestration, no Redis setup
- **Debugging**: Single process makes debugging with `pdb` straightforward
- **Smoke testing**: Used by CI (`test_run_ppo.py`) to verify PPO works end-to-end
- **Baseline**: Performance reference point for the distributed pipeline

## Usage

```bash
python run_ppo_swimmer.py
```

## Implementation

### Architecture

The script implements a minimal PPO trainer in a single ~200-line Python file:

```
main():
  1. Create MuJoCo Swimmer-v4 env (single)
  2. Build MujocoNormalNet from rl_server.algorithms.ppo.mujoco_normal
  3. Training loop (2000 iterations):
     a. Collect trajectory by running policy in env
     b. Compute GAE advantages
     c. Compute PPO clipped loss
     d. Optimize with SGD
     e. Log progress
  4. Close env
```

### Key Components

| Component | Detail |
|-----------|--------|
| **Algorithm** | PPO Normal (Gaussian policy) |
| **Environment** | Swimmer-v4 (obs_dim=8, act_dim=2) |
| **Network** | `MujocoNormalNet` from `rl_server.algorithms.ppo.mujoco_normal` |
| **Optimizer** | Adam, lr=2.5e-4 |
| **Total iterations** | 2000 |
| **Steps per iteration** | 2048 |
| **Batch size** | 64 |
| **Epochs per iteration** | 10 |
| **Clip ε** | 0.2 |
| **γ** | 0.99 |
| **GAE λ** | 0.95 |

### Training Loop

```python
for iteration in range(2000):
    # 1. Collect trajectory
    states, actions, log_probs, rewards, dones = collect_trajectory(env, net, 2048)

    # 2. Compute advantages (GAE)
    advantages, returns = compute_gae(rewards, dones, values, γ, λ)

    # 3. Optimize (10 epochs)
    for epoch in range(10):
        for batch in minibatches(states, actions, log_probs, advantages, returns, 64):
            new_log_probs, new_values, entropy = net.get_calculate_data(batch_states, batch_actions)
            ratio = exp(new_log_probs - batch_log_probs)

            # PPO clipped loss
            pg_loss = -mean(min(ratio * adv, clamp(ratio, 1-ε, 1+ε) * adv))
            v_loss = MSE(new_values, batch_returns)
            loss = pg_loss + v_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Dependencies

- `gymnasium[mujoco]` for the Swimmer environment
- PyTorch for network computation
- `rl_server` package for algorithm classes (imports from `rl_server.algorithms.ppo.mujoco_normal`)

### Expected Output

The script prints iteration number and loss/scalar stats. After 2000 iterations, the policy should show improvement over random initialization (positive rewards in Swimmer).

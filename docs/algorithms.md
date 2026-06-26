---
layout: default
title: Algorithms
description: Algorithm details and environment support
---

# 🧠 Algorithms

## DQN — Dueling + NoisyNet

**Registry key:** `DQNGymClassic`

- Dueling architecture (shared backbone → state value + advantage streams)
- Factorized Gaussian noise layers (NoisyNet) for exploration
- Epsilon-greedy with linear decay
- Target network sync every 10 updates
- Experience replay buffer (capacity: 100k)

**Environments:** CartPole, MountainCar, Acrobot, Pendulum, LunarLander

## PPO — Proximal Policy Optimization

### PPO Normal (`MujocoNormal`)
- Gaussian policy with learnable `log_std`
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Value function clipping
- Tanh output squash to `[-1, 1]`

### PPO Beta (`MujocoBeta`)
- Beta distribution policy (naturally bounded to `[0, 1]`)
- Same GAE + clipping as Normal variant
- Suitable for environments with inherently bounded actions

### PPO MicroRTS (`MicroRTS`)
- Convolutional + dense network for 10×10×27 observation space
- Masked categorical distribution (invalid actions get `-inf` logit)
- Hierarchical action space: unit type → sub-action

**MuJoCo Environments:** Swimmer, HalfCheetah, Ant, Hopper, Walker2d, Humanoid, HumanoidStandup, Pusher, Reacher, InvertedPendulum, InvertedDoublePendulum

## SAC — Soft Actor-Critic

**Registry key:** `SACMujocoNormal`

- Dual Q-networks (clip to minimum)
- Entropy regularization with learned alpha
- Gaussian policy
- Soft target network updates

## TD3 — Twin Delayed DDPG

**Registry key:** `TD3MujocoNormal`

- Clipped double Q-learning
- Delayed policy updates (policy every 2 Q-updates)
- Target policy smoothing (Gaussian noise on target actions)
- Deterministic policy with exploration noise

## Adding a Custom Algorithm

See the [Tutorial](https://github.com/1998x-stack/rl_server/blob/main/TUTORIAL.md#7-adding-a-new-algorithm) for a step-by-step walkthrough.

Quick steps:
1. Create `rl_server/algorithms/my_algo/` with `network.py`, `agent.py`, `calculator.py`
2. Implement `AlgoBaseNet`, `AlgoBaseAgent`, `AlgoBaseCalculate`
3. Add an `elif` branch in `rl_server/algorithms/__init__.py:_lazy_load()`
4. Train: `python -m rl_server.entrypoints.train --env-name MyAlgo`

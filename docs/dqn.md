---
layout: default
title: DQN Algorithm
description: Dueling DQN with NoisyNet — discrete-action deep Q-learning
---

# DQN — Deep Q-Network

**Registry keys:** `DQNGymClassic`, `DQNAtari`

DQN in RL Server uses Dueling architecture (state value + advantage streams) with optional NoisyNet and ε-greedy exploration.

## Files

```
rl_server/algorithms/dqn/
  network.py    — DQNGymClassicNet: Dueling DQN network
  agent.py      — DQNGymClassicAgent: ε-greedy rollout + checker
  calculator.py — DQNGymClassicCalculate: experience replay, target network
  atari.py      — AtariDQN variants: CNN-based Dueling DQN for Atari games
```

---

## Network — `DQNGymClassicNet`

### Architecture

```
Input (obs_dim)
    ↓
nn.Linear(obs_dim, hidden_dim) + ReLU      ← layer_init (orthogonal, std=√2)
nn.Linear(hidden_dim, hidden_dim) + ReLU
nn.Linear(hidden_dim, hidden_dim) + ReLU
    ↓
┌────────────────┬──────────────────────┐
│  Value Head     │  Advantage Head       │
│  Linear(h, 1)   │  Linear(h, act_dim)   │
└────────┬───────┘└──────────┬───────────┘
         └──────────┬────────┘
                    ▼
     Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
                    ↓
            argmax(Q) → action
```

### Key Methods

| Method | Returns | Used By |
|--------|---------|---------|
| `forward(states)` | `actions` (argmax of Q) | Sampler, Checker (inference) |
| `get_q_values(states)` | Full Q-values | Calculator (training loss) |
| `update_state(version, grads_buffer)` | None | Main process |

### Supported Environments

| Env Key | OBS_DIM | ACT_DIM | HIDDEN_DIM |
|---------|---------|---------|------------|
| CartPole | 4 | 2 | 16 |
| MountainCar | 2 | 3 | 32 |
| Acrobot | 6 | 3 | 64 |
| Pendulum | 3 | 1 | 64 |
| LunarLander | 8 | 4 | 128 |

### `update_state()` Detail

Creates a new `Adam` optimizer each call (no optimizer state persistence — intentional choice for the gradient-buffer architecture):

```python
def update_state(self, version, grads_buffer):
    train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG.LEARNING_RATE)
    train_optim.zero_grad()
    for param, grad in zip(self.parameters(), grads_buffer):
        param.grad = torch.FloatTensor(grad)
    train_optim.step()
```

---

## Agent — `DQNGymClassicAgent`

### Multi-Environment Sampling (`sample_multi_envs`)

```python
exps = [[] for _ in range(num_envs)]
for step in range(NUM_STEPS):
    actions = self._get_sample_actions(states)
    for i in range(num_envs):
        next_state, reward, done, ... = envs[i].step(action)
        if done: next_state = envs[i].reset()
        exps[i].append([state, action, reward, done, train_version])
        states[i] = next_state
return exps
```

Each environment's trajectory is a separate list of `[state, action, reward, done, version]` tuples. The outer list has `num_envs` elements.

### Action Selection

```python
@torch.no_grad()
def _get_sample_actions(self, states):
    if random() > self.epsilon:
        actions = self.sample_net(t_states)    # argmax of Q
    else:
        actions = np.random.choice(act_dim)    # random exploration
```

### Checker (`check_single_env`)

Single episode until `done`. Returns:
- `sum_rewards`: total episode return
- `average_mus`: mean action value

---

## Calculator — `DQNGymClassicCalculate`

### Training Components

| Component | Detail |
|-----------|--------|
| **Replay buffer** | `ExperienceBuffer(capacity=100000)` |
| **Batch size** | 256 |
| **Target network** | Copies from online every `version_diff=10` updates |
| **Repeat count** | 64 minibatch samples per `generate_grads` call |
| **Loss** | `F.mse_loss` (MSE TD error) |
| **Gamma** | 0.99 |

### Gradient Computation Flow

```
generate_grads(samples, model_dict):
  1. Unpack samples → state/action/reward/done arrays
  2. Create Experience tuples from consecutive state pairs
  3. Append to ExperienceBuffer (FIFO, max 100k)
  4. If buffer < batch_size: raise ValueError
  5. Sync calculate_net ← share_model.state_dict()
  6. Every 10th call: sync target_net ← calculate_net.state_dict()
  7. Repeat 64 times:
     a. Sample batch from ExperienceBuffer
     b. Compute Q(s,a) via calculate_net.get_q_values()
     c. Compute expected Q via Double DQN:
        - next_actions = argmax(Q(s') from calculate_net)
        - target_Q(s', next_actions) from target_net
        - expected = r + γ * target_Q (0 if done)
     d. loss = MSE(Q(s,a), expected) / 64
     e. loss.backward()
  8. Collect parameter.grad as numpy arrays
  9. Return [grads], train_version
```

### Dueling Double DQN

The calculator implements **Double DQN** action selection:

```python
# Current net selects the action
next_q_values = self.calculate_net.get_q_values(next_states_v)
expected_actions = torch.max(next_q_values, 1)[1]

# Target net evaluates the action
target_next_q_values = self.target_net.get_q_values(next_states_v)
target_next_q_values = target_next_q_values.gather(1, expected_actions.unsqueeze(1))
```

This reduces the overestimation bias of standard DQN.

---

## Atari DQN (`DQNAtari`)

**Registered key:** `DQNAtari`

An extension for Atari games with CNN-based architecture.

### Architecture

```
Input: 84×84 grayscale, 4-frame stack (84×84×4)
    ↓
Conv2D(4, 32, 8, stride=4) + ReLU
Conv2D(32, 64, 4, stride=2) + ReLU
Conv2D(64, 64, 3, stride=1) + ReLU
Flatten
    ↓
┌────────────────┬──────────────────────┐
│  Value Head     │  Advantage Head       │
└────────┬───────┘└──────────┬───────────┘
         └──────────┬────────┘
                    ▼
            Q(s,a) = V + A - mean(A)
```

### Environment Preprocessing

```python
def make_atari_env(env_name):
    env = gym.make(env_name, obs_type="rgb", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    return env
```

### Config

| Parameter | Value |
|-----------|-------|
| NUM_ENVS | 4 |
| NUM_STEPS | 128 |
| Experience buffer | 500,000 |
| Batch size | 32 |
| Epsilon decay | 1.0 → 0.1 over 100k steps |
| Learning rate | 2.5e-4 |

### Supported Games

Pong, Breakout, SpaceInvaders, Boxing, Qbert, BeamRider, Enduro, Seaquest

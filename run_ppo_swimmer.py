#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Swimmer-v4 上 PPO 快速脚本：进程内若干迭代，用于冒烟或调试（无多进程）。"""
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import numpy as np
import gymnasium as gym

# Set Swimmer as the active environment before imports
import rl_server.algorithms.ppo.mujoco_normal as ppo_mod
ppo_mod.current_env_name = 'Swimmer'

from rl_server.algorithms.ppo.mujoco_normal import (
    MujocoNormalNet, MujocoNormalAgent, MujocoNormalCalculate,
    TRAIN_ENVS, MODEL_CONFIG
)
from rl_server.utils.process import setup_seed


def evaluate(net, env_name, num_episodes=3):
    """运行若干回合贪心策略评估，返回平均回报。

    Args:
        net: 策略网络，``forward`` 输出动作。
        env_name: ``gymnasium`` 环境 ID。
        num_episodes: 评估回合数。

    Returns:
        各回合回报的平均值。
    """
    rewards = []
    for _ in range(num_episodes):
        env = gym.make(env_name)
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        while not done and steps < 1000:
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                action = net(state_t).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(episode_reward)
        env.close()
    return np.mean(rewards)


def main():
    """构造 Swimmer 并行环境，循环采样—聚合梯度—``update_state``，最后对比训练前后评估回报。"""
    setup_seed(42)
    env_name = TRAIN_ENVS['Swimmer'].ENV_NAME
    obs_dim = TRAIN_ENVS['Swimmer'].OBS_DIM
    act_dim = TRAIN_ENVS['Swimmer'].ACT_DIM
    num_envs = 4       # small for quick test
    num_steps = 200     # steps per sample round
    num_iterations = 10  # training iterations

    print(f"PPO Training: {env_name}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  num_envs={num_envs}, num_steps={num_steps}, iterations={num_iterations}")
    print()

    # Create network
    net = MujocoNormalNet()
    calc = MujocoNormalCalculate(net)

    # Create parallel environments
    envs = [gym.make(env_name) for _ in range(num_envs)]
    states = [envs[i].reset(seed=42+i)[0] for i in range(num_envs)]

    # Evaluate before training
    pre_reward = evaluate(net, env_name)
    print(f"Pre-training eval reward: {pre_reward:.2f}")
    print()

    model_dict = {'TRAIN_VERSION': 0}
    start_time = time.time()

    for iteration in range(num_iterations):
        # === SAMPLE PHASE ===
        all_exps = [[] for _ in range(num_envs)]

        for step in range(num_steps):
            with torch.no_grad():
                states_t = torch.FloatTensor(np.array(states))
                actions, log_probs = net.get_sample_data(states_t)
                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()

            for i in range(num_envs):
                next_state, reward, terminated, truncated, _ = envs[i].step(actions_np[i])
                done = terminated or truncated
                if done:
                    next_state, _ = envs[i].reset()

                # PPO sample format: [state, action, reward, done, log_probs, version]
                all_exps[i].append([
                    states[i], actions_np[i], reward, done,
                    log_probs_np[i], model_dict['TRAIN_VERSION']
                ])
                states[i] = next_state

        # === TRAIN PHASE ===
        # Process each environment's experience
        total_grads = None
        grad_count = 0

        for env_exps in all_exps:
            try:
                grads_list, train_version = calc.generate_grads(env_exps, model_dict)
                for grads in grads_list:
                    grad_count += 1
                    if total_grads is None:
                        total_grads = [g.copy() if g is not None else None for g in grads]
                    else:
                        for j, g in enumerate(grads):
                            if g is not None and total_grads[j] is not None:
                                total_grads[j] += g
            except ValueError as e:
                # "exps is not Enough" - buffer not full yet
                pass

        if total_grads is not None and grad_count > 0:
            # Average gradients
            avg_grads = [g / grad_count if g is not None else None for g in total_grads]
            model_dict['TRAIN_VERSION'] += 1
            net.update_state(model_dict['TRAIN_VERSION'], avg_grads)

            elapsed = time.time() - start_time
            print(f"  Iter {iteration+1}/{num_iterations}: "
                  f"version={model_dict['TRAIN_VERSION']}, "
                  f"grads_from={grad_count} envs, "
                  f"time={elapsed:.1f}s")
        else:
            print(f"  Iter {iteration+1}/{num_iterations}: buffer filling...")

    # Cleanup envs
    for env in envs:
        env.close()

    elapsed = time.time() - start_time
    print()
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final version: {model_dict['TRAIN_VERSION']}")

    # Evaluate after training
    post_reward = evaluate(net, env_name)
    print(f"Post-training eval reward: {post_reward:.2f}")
    print(f"Improvement: {post_reward - pre_reward:+.2f}")


if __name__ == '__main__':
    main()

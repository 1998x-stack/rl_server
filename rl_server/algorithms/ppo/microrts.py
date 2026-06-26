# -*- coding: utf-8 -*-
"""MicroRTS 上的 PPO：掩码分类分布、向量环境与多段动作空间。
"""
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import numpy as np
from collections import deque
from typing import Optional, Union

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSVecEnv

from rl_server.core.base import AlgoBaseNet, AlgoBaseAgent, AlgoBaseCalculate, layer_init

# Training parameters
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAE_LAMBDA'] = 0.95
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['SAMPLE_REUSE'] = 1
TRAIN_CONFIG['BATCH_SIZE'] = 512
TRAIN_CONFIG['DISPATCH_SIZE'] = TRAIN_CONFIG['BATCH_SIZE'] * 2
TRAIN_CONFIG['CLIP_COEF'] = 0.2
TRAIN_CONFIG['MAX_CLIP_COEF'] = 4
TRAIN_CONFIG['ENT_COEF'] = 0.01
TRAIN_CONFIG['VLAUE_COEF'] = 1
TRAIN_CONFIG['IS_CLIP_VALUE_LOSS'] = False
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4
TRAIN_CONFIG['ENABLE_TARGET'] = False
TRAIN_CONFIG['UPDATE_ALPHA'] = 0.2

# Model and environment config
MODEL_CONFIG = dict()
MODEL_CONFIG['ENV_NAME'] = "MicroRTSEnv"
MODEL_CONFIG['NUM_ENVS'] = 8
MODEL_CONFIG['NUM_STEPS'] = 512
MODEL_CONFIG['OBS_SPACE'] = (10, 10, 27)
MODEL_CONFIG['ACTION_SHAPE'] = [100, 6, 4, 4, 4, 4, 7, 49]
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MaskedCategorical(Categorical):
    """在非法动作上施加 ``-inf`` logits 的分类分布，用于约束策略采样。"""

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
        device: Union[str, torch.device] = 'auto'
    ):
        # Support both 'action_masks' and 'masks' parameter names
        if masks is not None and action_masks is None:
            action_masks = masks

        self.device = self._detect_device(device, logits, probs, action_masks)
        self.action_masks = self._process_masks(action_masks)
        self._validate_input_shapes(logits, probs, self.action_masks)
        masked_logits = self._apply_action_masks(logits, probs)
        super().__init__(logits=masked_logits, probs=None, validate_args=validate_args)

    @classmethod
    def _detect_device(cls, device, *tensors):
        if isinstance(device, torch.device):
            return device
        if device == 'auto':
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    return t.device
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _process_masks(self, masks):
        if masks is None:
            return torch.tensor([], device=self.device)
        return masks.to(self.device).bool()

    def _validate_input_shapes(self, logits, probs, masks):
        base_tensor = logits if logits is not None else probs
        if base_tensor is None:
            raise ValueError("Must provide logits or probs")
        if masks.ndim != base_tensor.ndim:
            raise ValueError(
                f"Mask dimension mismatch: input shape {base_tensor.shape} vs mask shape {masks.shape}"
            )

    def _apply_action_masks(self, logits, probs):
        if logits is not None:
            clamped_logits = logits.to(self.device)
            # Replace NaN with a large negative value so masked softmax doesn't error
            clamped_logits = torch.nan_to_num(clamped_logits, nan=-1e9)
            masked = torch.where(
                self.action_masks,
                clamped_logits,
                torch.tensor(-torch.inf, device=self.device)
            )
            all_masked = ~self.action_masks.any(dim=-1)
            if all_masked.any():
                masked[all_masked] = clamped_logits[all_masked]
            return masked
        elif probs is not None:
            clamped_probs = probs.to(self.device)
            return torch.where(
                self.action_masks,
                torch.log(clamped_probs + 1e-8),
                torch.tensor(-torch.inf, device=self.device)
            )
        else:
            raise RuntimeError("Logic error, check initialization parameters")

    def entropy(self):
        if self.action_masks is None:
            return super().entropy()

        valid_probs = self.probs * self.action_masks
        valid_probs = valid_probs / valid_probs.sum(-1, keepdim=True)

        p_log_p = valid_probs * torch.log(valid_probs + 1e-8)
        return -p_log_p.nansum(dim=-1)

    def argmax(self):
        return (self.probs * self.action_masks).argmax(dim=-1)


class MicroRTSNet(AlgoBaseNet):
    """卷积骨干 + 多输出头，对应 MicroRTS 分段动作空间。"""

    def __init__(self):
        super(MicroRTSNet, self).__init__()
        self.device = MODEL_CONFIG['DEVICE']

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(),
        )

        self.actor_unit = layer_init(nn.Linear(256, 100), std=0.01)
        self.actor_type = layer_init(nn.Linear(256, 6), std=0.01)
        self.actor_move = layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_harvest = layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_return = layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_produce = layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_produce_type = layer_init(nn.Linear(256, 7), std=0.01)
        self.actor_attack = layer_init(nn.Linear(256, 49), std=0.01)

        self.critic = layer_init(nn.Linear(256, 1), std=1)
        self.train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])

    def forward(self, x: torch.Tensor):
        x = x.permute((0, 3, 1, 2))
        obs = self.network(x)
        return [
            self.actor_unit(obs),
            self.actor_type(obs),
            self.actor_move(obs),
            self.actor_harvest(obs),
            self.actor_return(obs),
            self.actor_produce(obs),
            self.actor_produce_type(obs),
            self.actor_attack(obs)
        ], self.critic(obs)

    def update_state(self, version, grads_buffer):
        self.train_optim.zero_grad()
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        self.train_optim.step()


class MicroRTSAgent(AlgoBaseAgent):
    """``MicroRTSVecEnv`` 上并行 rollout，支持 checker 统计与 TB 风格日志。"""

    def __init__(self, sample_net, is_checker=False):
        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.num_check_single_envs = 16
        self.num_envs = MODEL_CONFIG['NUM_ENVS']
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.action_shape = MODEL_CONFIG['ACTION_SHAPE']
        self.outcomes = deque(maxlen=100)
        self.rewards = deque(maxlen=100)
        self.total_rewards = 0
        self.steps = 0
        if not is_checker:
            self.env = MicroRTSVecEnv(
                num_envs=self.num_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_envs)],
                map_path='maps/10x10/basesWorkers10x10.xml',
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        else:
            self.env = MicroRTSVecEnv(
                num_envs=self.num_check_single_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_check_single_envs)],
                map_path='maps/10x10/basesWorkers10x10.xml',
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        self.obs = self.env.reset()

    def __del__(self):
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except Exception:
            pass

    def get_units_number(unit_type, bef_obs, ind_obs):
        return int(bef_obs[ind_obs][:, :, unit_type].sum())

    @torch.no_grad()
    def get_action(self, states, type_masks: torch.Tensor = None):
        split_logits, _ = self.sample_net(states)
        type_masks = torch.Tensor(type_masks)
        multi_categoricals = [
            MaskedCategorical(
                logits=split_logits[0],
                masks=type_masks
            )
        ]
        action_components = [multi_categoricals[0].sample()]
        action_masks = np.array(
            self.env.vec_client.getUnitActionMasks(
                action_components[0].cpu().numpy()
            )
        )
        action_masks = action_masks.reshape(len(action_components[0]), -1)
        action_masks = torch.Tensor(action_masks)
        split_suam = torch.split(action_masks, self.action_shape[1:], dim=1)
        masks = torch.cat(
            tensors=(type_masks, action_masks),
            dim=1
        )
        multi_categoricals = multi_categoricals + [MaskedCategorical(logits=logits, masks=iam)
                                                    for (logits, iam) in zip(split_logits[1:], split_suam)]
        action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
        actions = torch.stack(tensors=action_components, dim=0)
        probs = torch.stack(
            [
                multi_categorical.log_prob(action)
                for multi_categorical, action in zip(multi_categoricals, actions)
            ]
        )
        return actions.cpu().numpy(), masks.cpu().numpy(), probs.cpu().numpy()

    @torch.no_grad()
    def _get_single_action(self, states, type_masks=None):
        split_logits, _ = self.sample_net(states)
        type_masks = torch.Tensor(type_masks)
        multi_categoricals = [MaskedCategorical(logits=split_logits[0], masks=type_masks)]
        action_components = [multi_categoricals[0].argmax()]
        action_masks = np.array(
            self.env.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())
        )
        action_masks = action_masks.reshape(len(action_components[0]), -1)
        action_masks = torch.Tensor(action_masks)
        split_suam = torch.split(action_masks, self.action_shape[1:], dim=1)
        multi_categoricals = multi_categoricals + [MaskedCategorical(logits=logits, masks=iam)
                                                    for (logits, iam) in zip(split_logits[1:], split_suam)]
        action_components += [categorical.argmax() for categorical in multi_categoricals[1:]]
        actions = torch.stack(action_components, dim=0)
        return actions.cpu().numpy()

    def sample_multi_envs(self, model_dict):
        exps = [[] for _ in range(self.num_envs)]
        if self.num_steps > 0:
            for _ in range(0, self.num_steps):
                self.steps = self.steps + 1
                unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
                with torch.no_grad():
                    action, mask, prob = self.get_action(states=torch.Tensor(self.obs), type_masks=unit_mask)
                    next_obs, rs, done, _ = self.env.step(action.T)
                    for i in range(self.num_envs):
                        exps[i].append([self.obs[i], action.T[i], rs[i], mask[i], done[i], prob.T[i], model_dict['TRAIN_VERSION']])
                self.obs = next_obs
        return exps

    def check_single_env(self):
        step_record_dict = dict()
        step_record_dict['outcomes'] = 0
        step_record_dict['reward'] = 0
        step_record_dict['total_reward'] = 0

        for _ in range(0, 512):
            unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_check_single_envs, -1)
            with torch.no_grad():
                action = self._get_single_action(states=torch.Tensor(self.obs), type_masks=unit_mask)
                next_obs, rs, done, _ = self.env.step(action.T)
                self.rewards.append(sum(rs) / len(rs))
                self.total_rewards = self.total_rewards + rs[0]
            for i in range(self.num_check_single_envs):
                if done[i]:
                    if MicroRTSAgent.get_units_number(11, self.obs, i) > MicroRTSAgent.get_units_number(12, self.obs, i):
                        self.outcomes.append(1)
                    else:
                        self.outcomes.append(0)
                    print(sum(self.outcomes) / len(self.outcomes))
                    step_record_dict['outcomes'] = sum(self.outcomes) / len(self.outcomes)
                    step_record_dict['reward'] = sum(self.rewards) / len(self.rewards)
                    if i == 0:
                        step_record_dict['total_reward'] = self.total_rewards
                        self.total_rewards = 0
            self.obs = next_obs
        return step_record_dict


class MicroRTSCalculate(AlgoBaseCalculate):
    """PPO 多段动作掩码损失与价值损失，支持样本重用与分块 dispatch。"""

    def __init__(self, SHARE_MODEL):
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG
        self.device = self.model_config['DEVICE']
        self.num_reuse = self.train_config['SAMPLE_REUSE']
        self.batch_size = self.train_config['BATCH_SIZE']
        self.dispatch_size = self.train_config['DISPATCH_SIZE']

        self.share_model = SHARE_MODEL.to(device=self.device)
        self.states = torch.Tensor([]).to(device=self.device)
        self.actions = torch.Tensor([]).to(device=self.device)
        self.returns = torch.Tensor([]).to(device=self.device)
        self.masks = torch.Tensor([]).to(device=self.device)
        self.probs = torch.Tensor([]).to(device=self.device)
        self.advantages = torch.Tensor([]).to(device=self.device)
        self.calculate_net = MicroRTSNet().to(self.device)

    def generate_grads(self, samples, model_dict):

        self.calculate_net.load_state_dict(self.share_model.state_dict())
        train_version = model_dict['TRAIN_VERSION']
        ent_coef = self.train_config['ENT_COEF']
        vf_coef = self.train_config['VLAUE_COEF']

        len_samples = len(samples)
        dones = torch.zeros((len_samples,)).to(device=self.device)
        rewards = torch.zeros((len_samples,)).to(device=self.device)
        b_states = torch.zeros((len_samples,) + self.model_config['OBS_SPACE']).to(device=self.device)
        b_actions = torch.zeros((len_samples, len(self.model_config['ACTION_SHAPE']))).to(device=self.device)
        b_masks = torch.ones((len_samples, sum(self.model_config['ACTION_SHAPE']))).to(device=self.device)
        b_log_probs = torch.zeros((len_samples, len(self.model_config['ACTION_SHAPE']))).to(device=self.device)

        gamma = self.train_config['GAMMA']
        gae_lambda = self.train_config['GAE_LAMBDA']

        for i in range(len_samples):
            b_states[i] = torch.Tensor(samples[i][0]).to(device=self.device)
            b_actions[i] = torch.Tensor(samples[i][1]).to(device=self.device)
            rewards[i] = torch.Tensor(np.array(samples[i][2]).reshape(-1, 1)).to(device=self.device)
            b_masks[i] = torch.Tensor(samples[i][3]).to(device=self.device)
            dones[i] = torch.Tensor(np.array(samples[i][4]).reshape(-1, 1)).to(device=self.device)
            b_log_probs[i] = torch.Tensor(np.array(samples[i][5])).to(device=self.device)

        new_log_prob, entropy, new_values = self.get_prob_entropy_value(b_states, actions=b_actions.T, masks=b_masks)
        with torch.no_grad():
            last_gae_lam = 0
            b_advantages = torch.zeros((len_samples,)).to(device=self.device)
            b_returns = torch.zeros((len_samples,)).to(device=self.device)
            for t in reversed(range(len_samples - 1)):
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * new_values[t + 1] * next_non_terminal - new_values[t]
                b_advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                b_returns[t] = b_advantages[t] + new_values[t]

        b_advantages = b_advantages.reshape(-1, 1)
        new_log_prob = new_log_prob.to(self.model_config['DEVICE'])
        entropy = entropy.to(self.model_config['DEVICE'])
        ratio1 = (new_log_prob - b_log_probs).exp()
        ratio2 = (new_log_prob.sum(1) - b_log_probs.sum(1)).exp().reshape(-1, 1).expand_as(ratio1)
        ratio3 = (ratio1 + ratio2) / 2

        pg_loss3 = self.get_pg_loss(ratio3, b_advantages)

        pg_loss = -pg_loss3.mean()
        entropy_loss = -entropy.mean()
        v_loss = ((new_values - b_returns) ** 2).mean()
        loss = pg_loss + ent_coef * entropy_loss + v_loss * vf_coef
        self.calculate_net.zero_grad()
        loss.backward()

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]

        return [grads], train_version

    def get_prob_entropy_value(self, states, actions, masks, action_space=[100, 6, 4, 4, 4, 4, 7, 49]):
        split_logits, values = self.calculate_net(states)
        split_masks = torch.split(masks, action_space, dim=1)
        multi_categoricals = [
            MaskedCategorical(logits=logits, masks=iam)
            for (logits, iam) in zip(split_logits, split_masks)
        ]
        log_probs = torch.stack(
            [
                categorical.log_prob(a)
                for a, categorical in zip(actions, multi_categoricals)
            ]
        )
        entropys = torch.stack(
            [
                categorical.entropy()
                for categorical in multi_categoricals
            ]
        )

        return log_probs.T, entropys.T, values.reshape((-1,))

    def get_pg_loss(self, ratio, advantage):
        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']
        clip_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

        min_loss_policy = torch.min(ratio * advantage, clip_ratio * advantage)
        max_loss_policy = torch.max(min_loss_policy, max_clip_coef * advantage)
        return torch.where(advantage >= 0, min_loss_policy, max_loss_policy)

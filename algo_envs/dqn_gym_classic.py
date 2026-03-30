# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

import libs.exps as Exps
import algo_envs.algo_base as AlgoBase

TRAIN_ENVS = {
    'CartPole': SimpleNamespace(**{
        'ENV_NAME': "CartPole-v1", 
        'OBS_DIM':4,  # 
        'ACT_DIM':2, 
        'HIDDEN_DIM':16
    }),
    'MountainCar': SimpleNamespace(**{
        'ENV_NAME': "MountainCar-v1", 
        'OBS_DIM': 2,  # 位置和速度
        'ACT_DIM': 3,  # 左/无/右
        'HIDDEN_DIM': 32  # 因奖励稀疏需更复杂网络
    }),
    'Acrobot': SimpleNamespace(**{
        'ENV_NAME': "Acrobot-v1",
        'OBS_DIM': 6,  # 包含两个关节的sin/cos值和角速度
        'ACT_DIM': 3,  # -1/0/+1扭矩
        'HIDDEN_DIM': 64  # 高维状态需更深网络
    }),
    'Pendulum': SimpleNamespace(**{
        'ENV_NAME': "Pendulum-v1", 
        'OBS_DIM': 3,  # 角度sin/cos值+角速度
        'ACT_DIM': 1,  # 连续扭矩（需离散化处理）
        'HIDDEN_DIM': 64  # 需离散化动作空间适应DQN
    }),
    'LunarLander': SimpleNamespace(**{
        'ENV_NAME': "LunarLander-v2", 
        'OBS_DIM': 8,  # 坐标/速度/角度/触地状态等[6]
        'ACT_DIM': 4,  # 无动作/左/主引擎/右[6]
        'HIDDEN_DIM': 128  # 复杂状态需更大网络
    })
}

current_env_name = 'CartPole'

# 训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAMMA'] = 0.99
TRAIN_CONFIG['BATCH_SIZE'] = 256 # 批次大小
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4 # 学习率
TRAIN_CONFIG['epsilon'] = 0.01 # epsilon-greedy 

# 模型及环境
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32 # 环境数量
MODEL_CONFIG['NUM_STEPS'] = 512 # 一次采样的长度
MODEL_CONFIG['OBS_SPACE'] = (4,) # 状态空间 
MODEL_CONFIG['ACTION_SHAPE'] = [2] # 动作空间
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device

# replaybuffer 只支持 单个 trainer
MAX_BUFFER_SIZE = 100000

class DQNGymClassicNet(AlgoBase.AlgoBaseNet): 
    """
    """
    def __init__(self):
        super(DQNGymClassicNet,self).__init__()
        
        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM
        
        self.network = nn.Sequential(
            AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
        )
        
        # Dueling DQN
        self.value = AlgoBase.layer_init(nn.Linear(hide_dim, 1))
        self.advantage = AlgoBase.layer_init(nn.Linear(hide_dim, act_dim))
        
    def get_q_values(self, states):
        out = self.network(states)
        advantage = self.advantage(out)
        value = self.value(out)
        return value + advantage-advantage.mean()
    
    def forward(self, states):
        q_values = self.get_q_values(states)
        action  = torch.argmax(q_values,dim=-1)
        return action
            
    def update_state(self, version, grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        # 更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()
        
class DQNGymClassicAgent(AlgoBase.AlgoBaseAgent):
    """
    """
    def __init__(self, sample_net: DQNGymClassicNet, is_checker):
        super(DQNGymClassicAgent,self).__init__()
        
        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS']
        self.epsilon = TRAIN_CONFIG['epsilon']
        
        self.ACT_DIM = TRAIN_ENVS[current_env_name].ACT_DIM
        env_name = TRAIN_ENVS[current_env_name].ENV_NAME
        
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("DQNGymClassic check env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]
        
    def sample_multi_envs(self, model_dict):
        # 采样环境，保存 状态，动作，奖励，是否完成，训练版本号
        exps=[[] for _ in range(self.num_envs)]
        for _ in range(self.num_steps):
            actions = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()[0]
                    
                exps[i].append([self.states[i], actions[i], reward_n, done_n, model_dict['TRAIN_VERSION']])
                self.states[i] = next_state_n
                
        return exps
    
    def check_single_env(self):
        # 单一环境或者vectorized环境；
        actions = []
        rewards = []
        is_done = False
        step_record_dict = dict()

        while True:
            action = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(action)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            rewards.append(reward_n)
            actions.append(action)
            
            if is_done:
                break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['average_mus'] = np.mean(actions)
        
        return step_record_dict

    @torch.no_grad() # 此地用到模型推理
    def _get_sample_actions(self, states):
        # 采样 epsilon-greedy进行采样
        t_states = torch.Tensor(states)
        if np.random.random() > self.epsilon:
            actions = self.sample_net(t_states)
            actions  = actions.cpu().numpy()
        else:
            actions = np.random.choice(self.ACT_DIM, size = t_states.shape[0])
        return actions
    
    @torch.no_grad()
    def _get_single_action(self, state):
        action = self.sample_net(torch.Tensor(state))
        action  = action.cpu().numpy()
        return action
    
 
class DQNGymClassicCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self, SHARE_MODEL: DQNGymClassicNet):
        super(DQNGymClassicCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG # 训练设置参数
        self.model_config = MODEL_CONFIG # 模型设置参数
        self.share_model = SHARE_MODEL # 主模型，共享参数，版本统一
        self.calculate_net = DQNGymClassicNet()
        self.target_net = DQNGymClassicNet()
        self.exps_buffer = Exps.ExperienceBuffer(capacity=MAX_BUFFER_SIZE)

        self.gamma = self.train_config['GAMMA']
        self.batch_size = TRAIN_CONFIG['BATCH_SIZE']
        self.version_diff = 10 # 10版本迭代后，目标网络同步主模型
        self.num_repeat = 64
        self.update_version = 0
        
        
    def generate_grads(self, samples, model_dict):
        
        # 从样本中提取状态，动作，奖励和结束状态
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[4] for s in samples])
    
        for state, action, reward, done, next_state in zip(s_states[:-1], s_actions[:-1], s_rewards[:-1], s_dones[:-1], s_states[1:]):
            exp = Exps.Experience(state, action, reward, done, next_state)
            self.exps_buffer.append(exp)
        
        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")
            
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        if self.update_version % self.version_diff == 0:
            self.target_net.load_state_dict(self.calculate_net.state_dict())
        
        self.calculate_net.zero_grad()
        
        for _ in range(self.num_repeat):
            s_states, s_actions, s_rewards, s_dones, s_next_states = self.exps_buffer.sample(self.batch_size)
            
            states_v = torch.Tensor(s_states) # 类型为float32
            actions_v = torch.tensor(s_actions) # 类型为int64
            rewards_v = torch.Tensor(s_rewards) # 类型为float32
            next_states_v = torch.Tensor(s_next_states) # 类型为float32

            q_values: torch.Tensor = self.calculate_net.get_q_values(states_v) # 类型为float32
            q_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            
            with torch.no_grad():
                # 构建目标值，不需要梯度
                next_q_values: torch.Tensor = self.calculate_net.get_q_values(next_states_v) # 类型为float32
                expected_actions = torch.max(next_q_values, 1)[1]
                target_next_q_values: torch.Tensor = self.target_net.get_q_values(next_states_v) # 类型为float32
                target_next_q_values = target_next_q_values.gather(1, expected_actions.unsqueeze(1)).squeeze(1)
                target_next_q_values[s_dones] = 0.0 # 如果处于结束状态，重置为0
                expected_q_values = rewards_v + self.gamma * target_next_q_values
            
            loss = F.mse_loss(q_values, expected_q_values) / self.num_repeat
            loss.backward()

        # 不进行optimizer step操作，而且集合num_repeat下所有梯度，后续在主服务进行step
        grads = [
            param.grad.data.cpu().numpy()
                if param.grad is not None else None
                    for param in self.calculate_net.parameters()
        ]
        self.update_version = self.update_version +1
        
        return [grads], model_dict['TRAIN_VERSION']
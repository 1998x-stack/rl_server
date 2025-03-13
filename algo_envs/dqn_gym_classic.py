# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

import libs.exps as Exps
import algo_envs.algo_base as AlgoBase

TRAIN_ENVS = {
    'CartPole': SimpleNamespace(**{
        'env_name': "CartPole-v0", 
        'obs_dim':4,  # 
        'act_dim':2, 
        'hide_dim':16
    }),
    'MountainCar': SimpleNamespace(**{
        'env_name': "MountainCar-v0", 
        'obs_dim': 2,  # 位置和速度
        'act_dim': 3,  # 左/无/右
        'hide_dim': 32  # 因奖励稀疏需更复杂网络
    }),
    'Acrobot': SimpleNamespace(**{
        'env_name': "Acrobot-v1",
        'obs_dim': 6,  # 包含两个关节的sin/cos值和角速度
        'act_dim': 3,  # -1/0/+1扭矩
        'hide_dim': 64  # 高维状态需更深网络
    }),
    'Pendulum': SimpleNamespace(**{
        'env_name': "Pendulum-v1", 
        'obs_dim': 3,  # 角度sin/cos值+角速度
        'act_dim': 1,  # 连续扭矩（需离散化处理）
        'hide_dim': 64  # 需离散化动作空间适应DQN
    }),
    'LunarLander': SimpleNamespace(**{
        'env_name': "LunarLander-v2", 
        'obs_dim': 8,  # 坐标/速度/角度/触地状态等[6]
        'act_dim': 4,  # 无动作/左/主引擎/右[6]
        'hide_dim': 128  # 复杂状态需更大网络
    })
}

current_env_name = 'CartPole'

#训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['gamma'] = 0.99
TRAIN_CONFIG['batch_size'] = 256 # 批次大小
TRAIN_CONFIG['learning_rate'] = 2.5e-4 # 学习率
TRAIN_CONFIG['epsilon'] = 0.01 # epsilon-greedy 

#模型及环境
MODEL_CONFIG = dict()
MODEL_CONFIG['num_envs'] = 32 # 环境数量
MODEL_CONFIG['num_steps'] = 512 # 一次采样的长度
MODEL_CONFIG['obs_space'] = (4,) # 状态空间 
MODEL_CONFIG['action_shape'] = [2] # 动作空间
MODEL_CONFIG['device'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device

#replaybuffer 只支持 单个 trainer
MAX_BUFFER_SIZE = 100000

class DQNGymClassicNet(AlgoBase.AlgoBaseNet): 
    """
    """
    def __init__(self):
        super(DQNGymClassicNet,self).__init__()
        
        obs_dim = TRAIN_ENVS[current_env_name].obs_dim
        act_dim = TRAIN_ENVS[current_env_name].act_dim
        hide_dim = TRAIN_ENVS[current_env_name].hide_dim
        
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
        return value + advantage - advantage.mean()
    
    def forward(self, states):
        q_values = self.get_q_values(states)
        action  = torch.argmax(q_values,dim=-1)
        return action
            
    def update_state(self, version, grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['learning_rate'])
        train_optim.zero_grad()
        #更新网络参数
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
        self.num_steps = MODEL_CONFIG['num_steps']
        self.num_envs = MODEL_CONFIG['num_envs']
        self.epsilon = TRAIN_CONFIG['epsilon']
        
        self.rewards = []
        
        self.act_dim = TRAIN_ENVS[current_env_name].act_dim
        env_name = TRAIN_ENVS[current_env_name].env_name
        
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset() for i in range(self.num_envs)]
        else:
            print("DQNGymClassic check env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()
        
    def sample_env(self, model_dict): 
        exps=[[] for _ in range(self.num_envs)]
        
        for _ in range(self.num_steps):
            actions = self.get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()
                    
                exps[i].append([self.states[i], actions[i], reward_n, done_n, model_dict['train_version']])
                self.states[i] = next_state_n
                
        return exps
    
    def check_env(self):
        
        steps = 0
        mus = []
        rewards = []
        is_done = False
        step_record_dict = dict()

        while True:
            mu = self.get_check_action(self.states)
            next_state_n, reward_n, is_done, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            
            steps += 1
            if is_done:
                break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['average_mus'] = np.mean(mus)
        
        return step_record_dict

    @torch.no_grad()
    def get_sample_actions(self,states):
        t_states = torch.Tensor(states)
        if np.random.random() > self.epsilon:
            actions = self.sample_net(t_states)
            actions  = actions.cpu().numpy()
        else:
            actions = np.random.choice(self.act_dim,size = t_states.shape[0])
        return actions
    
    @torch.no_grad()
    def get_check_action(self,state):
        action = self.sample_net(torch.Tensor(state))
        action  = action.cpu().numpy()
        return action
    
 
class DQNGymClassicCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self,share_model: DQNGymClassicNet):
        super(DQNGymClassicCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG 
        self.share_model = share_model
        self.calculate_net = DQNGymClassicNet()
        self.target_net = DQNGymClassicNet()
        self.exps_buffer = Exps.ExperienceBuffer(capacity=MAX_BUFFER_SIZE)

        self.batch_size = TRAIN_CONFIG['batch_size']
        self.version_diff = 10
        self.num_repeat = 64
        self.update_version = 0
        
    def generate_grads(self, samples, model_dict):
        
        gamma = self.train_config['gamma']
        
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s for s in samples])
    
        for state,action,reward,done,next_state in zip(s_states[:-1],s_actions[:-1],s_rewards[:-1],s_dones[:-1],s_states[1:]):
            exp = Exps.Experience(state, action, reward, done, next_state)
            self.exps_buffer.append(exp)
        
        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")
            
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        train_version = model_dict['train_version']
        
        if self.update_version % self.version_diff == 0:
            self.target_net.load_state_dict(self.calculate_net.state_dict())
        
        self.calculate_net.zero_grad()
            
        for _ in range(self.num_repeat):
            s_states, s_actions, s_rewards, s_dones, s_next_states = self.exps_buffer.sample(self.batch_size)
            
            state_v = torch.Tensor(s_states)
            actions_v = torch.tensor(s_actions)
            rewards_v = torch.Tensor(s_rewards)
            next_state_v = torch.Tensor(s_next_states)

            q_values: torch.Tensor = self.calculate_net.get_q_values(state_v)
            
            with torch.no_grad():
                next_q_values: torch.Tensor = self.calculate_net.get_q_values(next_state_v)
                next_q_state_values: torch.Tensor = self.target_net.get_q_values(next_state_v)
                                
            q_value = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            next_q_value[s_dones] = 0.0
            expected_q_value = rewards_v + gamma * next_q_value
            
            loss = F.mse_loss(q_value,expected_q_value)/self.num_repeat
            
            loss.backward()

        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
        self.update_version = self.update_version +1

        return [grads],train_version
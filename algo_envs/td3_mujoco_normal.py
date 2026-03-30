# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.distributions.beta import Beta
import algo_envs.algo_base as AlgoBase
from types import SimpleNamespace

TRAIN_ENVS = {
    # 基础控制环境
    'Swimmer': SimpleNamespace(**{
        'ENV_NAME': "Swimmer-v3",
        'OBS_DIM': 8,       # 包含4个关节位置+4个关节速度
        'ACT_DIM': 2,       # 两对对称的推进器控制
        'HIDDEN_DIM': 32,     # 简单动力学适合小网络
        'USE_NOISE': True   # 水流扰动模拟
    }),
    
    # 高速运动控制
    'HalfCheetah': SimpleNamespace(**{
        'ENV_NAME': "HalfCheetah-v3",
        'OBS_DIM': 17,      # 8个身体部位位置+9个速度传感器
        'ACT_DIM': 6,       # 6个旋转关节扭矩控制
        'HIDDEN_DIM': 64,     # 中等复杂度运动控制
        'USE_NOISE': True   # 地面摩擦力随机化  
    }),

    # 复杂多关节控制
    'Ant': SimpleNamespace(**{
        'ENV_NAME': "Ant-v3",
        'OBS_DIM': 111,     # 13个关节位置+14个接触传感器+84个附加状态
        'ACT_DIM': 8,       # 四条腿各两个关节
        'HIDDEN_DIM': 256,    # 高维状态需深层网络
        'USE_NOISE': True   # 地形不平整模拟
    }),

    # 平衡控制基准
    'Hopper': SimpleNamespace(**{
        'ENV_NAME': "Hopper-v3",
        'OBS_DIM': 11,      # 4个关节角度+3个位置+4个速度
        'ACT_DIM': 3,       # 髋/膝/踝关节驱动
        'HIDDEN_DIM': 64,
        'USE_NOISE': True   # 着陆冲击噪声
    }),

    # 物体操作环境
    'Pusher': SimpleNamespace(**{
        'ENV_NAME': "Pusher-v2",
        'OBS_DIM': 23,      # 包含机械臂7关节+物体6D位姿
        'ACT_DIM': 7,       # 7自由度机械臂控制
        'HIDDEN_DIM': 128,    # 操作任务需要空间感知
        'USE_NOISE': True   # 物体滑动摩擦随机
    }),

    # 高自由度人体控制
    'Humanoid': SimpleNamespace(**{
        'ENV_NAME': "Humanoid-v3",
        'OBS_DIM': 376,     # 包含全身22个关节的完整动力学数据
        'ACT_DIM': 17,      # 主要关节驱动
        'HIDDEN_DIM': 512,    # 需大规模网络建模
        'USE_NOISE': True   # 肌肉力量波动模拟
    }),

    # 双足行走控制
    'Walker2d': SimpleNamespace(**{
        'ENV_NAME': "Walker2d-v3",
        'OBS_DIM': 17,      # 包含质心速度+关节角度
        'ACT_DIM': 6,       # 髋/膝/踝对称控制
        'HIDDEN_DIM': 64,
        'USE_NOISE': True   # 地面摩擦系数变化
    }),
    
    # 新增工业级环境
    'Manipulator': SimpleNamespace(**{
        'ENV_NAME': "UR5e-v0",       # 工业机械臂控制
        'OBS_DIM': 28,       # 6关节角度+末端位置+目标位姿
        'ACT_DIM': 6,        # 6个关节力矩控制
        'HIDDEN_DIM': 256,
        'USE_NOISE': True    # 负载质量不确定性
    }),
    
    'Reacher3D': SimpleNamespace(**{
        'ENV_NAME': "Reacher-v3",    # 三维空间到达任务
        'OBS_DIM': 16,       # 目标坐标+关节状态
        'ACT_DIM': 2,        # 平面旋转控制
        'HIDDEN_DIM': 128,
        'USE_NOISE': True    # 目标位置随机偏移
    }),
    
    'Quadruped': SimpleNamespace(**{
        'ENV_NAME': "AntMaze-v0",    # 复杂地形导航
        'OBS_DIM': 132,      # 激光雷达+本体感知
        'ACT_DIM': 12,       # 四足协调控制
        'HIDDEN_DIM': 512,
        'USE_NOISE': True    # 地形高度随机化
    }),
    
    # 医疗机器人环境
    'Bronchoscope': SimpleNamespace(**{
        'ENV_NAME': "Bronchoscope-v0", # 支气管介入仿真
        'OBS_DIM': 45,       # 包含3D位置+方向+支气管结构
        'ACT_DIM': 3,        # 弯曲/旋转/推进控制
        'HIDDEN_DIM': 256,
        'USE_NOISE': True    # 组织形变噪声
    }),
    
    'HumanoidStandup': SimpleNamespace(**{
        'ENV_NAME': "HumanoidStandup-v2",
        'OBS_DIM': 376,      # 同Humanoid-v3
        'ACT_DIM': 17,
        'HIDDEN_DIM': 512,
        'USE_NOISE': True    # 初始姿态随机化
    })
}

current_env_name = 'Swimmer'

# 训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAE_LAMBDA'] = 0.95 # gae lamada
TRAIN_CONFIG['GAMMA'] = 0.99 # 衰减系数
TRAIN_CONFIG['CLIP_COEF'] = 0.2 # pg loss clip
TRAIN_CONFIG['MAX_CLIP_COEF'] = 2 # pg loss max clip
TRAIN_CONFIG['ENT_COEF'] = 0.001 # 熵的权重
TRAIN_CONFIG['VLAUE_COEF'] = 1 # value loss 的权重
TRAIN_CONFIG['q_coef'] = 1 # q loss 的权重
TRAIN_CONFIG['q_mu_coef'] = 0.003 # q_pi loss 的权重
TRAIN_CONFIG['mu_probs_coef'] = 0.002# pi_probs的权重
TRAIN_CONFIG['IS_CLIP_VALUE_LOSS'] = False # 是否clip value loss
TRAIN_CONFIG['LEARNING_RATE'] = 2.5e-4 # 学习率

# 模型及环境 HalfCheetah
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32 # 环境数量 microRTS
MODEL_CONFIG['NUM_STEPS'] = 1000 # 一次采样的长度
MODEL_CONFIG['OBS_SPACE'] = (8,) # 状态空间 
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32) # 动作空间
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device
MODEL_CONFIG['MAX_ACTION'] = 1.0

class MujocoNormalQNet(AlgoBase.AlgoBaseNet):    
    
    def __init__(self):
        super(MujocoNormalQNet,self).__init__()
        
        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM
                
        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.noise_layer_out = AlgoBase.NoisyLinear(hide_dim,act_dim)
            self.noise_layer_hide = AlgoBase.NoisyLinear(hide_dim,hide_dim)
                            
            # normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.noise_layer_hide,
                    nn.ReLU(),
                    self.noise_layer_out,
                    nn.Tanh()
                )
        else:
            # normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Tanh()
                )
                
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
                        
        self.q_value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim+act_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1)),
            )
        
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )
        
    def get_distributions(self, states):
        mus = self.mu(states)
        dists = Normal(mus,torch.exp(self.log_std))
        return dists
        
    def forward(self, states):
        mus = self.mu(states)
        return mus
    
    def get_sample_data(self, states):
        dists = self.get_distributions(states)
        actions = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions,log_probs
    
    def get_check_data(self, states):
        dists = self.get_distributions(states)
        mus = self.mu(states)
        return mus,dists.entropy()
    
    def get_calculate_data(self, states,actions):
        values = self.value(states)
        dists = self.get_distributions(states)
        log_probs = dists.log_prob(actions) 
        return values,log_probs,dists.entropy()   
    
    def update_state(self,version,grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        train_optim.zero_grad()
        # 更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()
        
        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()
            
    def get_q_values(self, states,actions,next_states):
        q_input = torch.cat((states,actions),-1)
        q_values = self.q_value(q_input)
               
        next_dists = self.get_distributions(next_states)
                
        next_actions = next_dists.sample()
        q_input = torch.cat((next_states,next_actions),-1)
        ref_q_values = self.q_value(q_input)

        return q_values,ref_q_values
    
    def get_mu_q_values(self, states):
        mus = self.mu(states)
        noise = torch.randn_like(mus)
        mu_actions = mus + torch.exp(self.log_std) * noise
        
        q_input = torch.cat((states,mu_actions),-1)
        mu_q_values = self.q_value(q_input)
        return mu_q_values
                        
class MujocoNormalQAgent(AlgoBase.AlgoBaseAgent):
    
    def __init__(self,sample_net:MujocoNormalQNet,is_checker):
        super(MujocoNormalQAgent,self).__init__()
        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.device = MODEL_CONFIG['DEVICE']
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS']
        
        
        env_name = TRAIN_ENVS[current_env_name].ENV_NAME
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("MujocoNormalQ check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]
        
    def sample_multi_envs(self, model_dict):

        exps=[[] for _ in range(self.num_envs)]

        for _ in range(self.num_steps):
            
            actions,log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                # if i == 0:
                #     self.envs[i].render()
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()[0]
                    
                exps[i].append([self.states[i],actions[i],reward_n,done_n,log_probs[i], model_dict['TRAIN_VERSION']])
                self.states[i] = next_state_n
                
        return exps
    
    def check_single_env(self):
        step_record_dict = dict()
        
        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []

        while True:
            # self.envs.render()
            mu,entropy = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            
            steps += 1
            if is_done:
                break
            
            # time.sleep(100)
            # if steps >= self.num_steps:
            #    break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['sum_entropys'] = np.sum(entropys)
        step_record_dict['average_mus'] = np.mean(mus)
        
        return step_record_dict
            
    @torch.no_grad()
    def _get_sample_actions(self, states):
        states_v = torch.Tensor(np.array(states)) # use np.stack to replace!
        actions,log_probs = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def _get_single_action(self, state):
        state_v = torch.Tensor(np.array(state)) # use np.stack to replace!
        mu,entropy = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy()
        
Q_REPEAT_TIME = 1
        
class MujocoNormalQCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self,SHARE_MODEL:MujocoNormalQNet):
        super(MujocoNormalQCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG 
        self.share_model = SHARE_MODEL
        self.device = self.model_config['DEVICE']
        self.calculate_net = MujocoNormalQNet()
    
    def generate_grads(self,samples, model_dict):
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        self.calculate_net.to(self.device)
        train_version = model_dict['TRAIN_VERSION']
        
        gamma = self.train_config['GAMMA']
        gae_lambda = self.train_config['GAE_LAMBDA']
        ent_coef = self.train_config['ENT_COEF']
        vf_coef = self.train_config['VLAUE_COEF']
        q_coef = self.train_config['q_coef']
        q_mu_coef = self.train_config['q_mu_coef']
        
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])
    
        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)
        
        t_new_values,t_new_log_probs,t_entropys = self.calculate_net.get_calculate_data(t_states,t_actions)
                
        s_advantages = [0]
        s_returns = [0]
        
        
        
        
        # 计算优势值
        last_gae = 0.0
        with torch.no_grad():
            for value,next_value,reward,done in zip(reversed(t_new_values[:-1]),reversed(t_new_values[1:]),
                                                reversed(s_rewards[:-1]),reversed(s_dones[:-1])):
                if done:
                    delta = reward-value
                    last_gae = delta
                else:
                    delta = reward + gamma * next_value-value
                    last_gae = delta + gamma * gae_lambda * last_gae
                    
                s_advantages.append(last_gae)
                s_returns.append(last_gae + value)
                                
        t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)        
        t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)
                
        t_advantages = t_advantages.reshape(-1, 1)
        t_returns = t_returns.reshape(-1, 1)
                
        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # discrete ratio
        ratio1 = torch.exp(t_new_log_probs-old_log_probs)

        # prod ratio   
        ratio2 = torch.exp(t_new_log_probs.sum(1)-old_log_probs.sum(1)).reshape(-1, 1).expand_as(ratio1)
        
        # mixed ratio
        ratio3 = (ratio1+ratio2)/2

        # Policy loss
        pg_loss1 = self.get_pg_loss(ratio1,t_advantages)
        
        pg_loss2 = self.get_pg_loss(ratio2,t_advantages)
        
        pg_loss3 = self.get_pg_loss(ratio3,t_advantages)
        
        pg_loss4 = (pg_loss1+pg_loss2)/2
                
        # Policy loss
        pg_loss = -pg_loss3.mean()
        
        v_loss = F.mse_loss(t_returns, t_new_values)*vf_coef
        
        e_loss = -torch.mean(t_entropys)*ent_coef
        
        loss = pg_loss + v_loss + e_loss

        self.calculate_net.zero_grad()

        loss.backward()
        
        t_mu_q_values = self.calculate_net.get_mu_q_values(t_states)
        # 不能将Q网络的梯度进行传导
        for para in self.calculate_net.q_value.parameters():
            para.requires_grad = False

        q_mu_loss = -t_mu_q_values.mean() * q_mu_coef / (torch.exp(t_entropys.detach().sum()) + 1)
        q_mu_loss.backward()
        
        for para in self.calculate_net.q_value.parameters():
            para.requires_grad = True
                
        for _ in range(Q_REPEAT_TIME):            
            t_q_values,t_ref_q_values = self.calculate_net.get_q_values(t_states[:-1],t_actions[:-1],t_states[1:])
            
            # 计算Q-return
            s_q_returns = []

            with torch.no_grad():
                for reward,done,ref_q_value in zip(s_rewards,s_dones,t_ref_q_values):
                    if done:
                        q_return = reward
                    else:
                        q_return = reward + gamma * ref_q_value
                        
                    s_q_returns.append(q_return)
                                               
            t_q_returns = torch.Tensor(list(s_q_returns)).to(self.device)
            t_q_returns = t_q_returns.reshape(-1, 1)   
        
            q_loss = F.mse_loss(t_q_values,t_q_returns) * q_coef / Q_REPEAT_TIME
            q_loss.backward()
                
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
                
        return [grads],train_version
    
    def get_pg_loss(self,ratio,advantage):
        
        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']
        
        base_value = ratio * advantage
        clip_value = torch.clamp(ratio,1.0-clip_coef,1.0 + clip_coef) * advantage
        min_loss_policy = torch.min(base_value, clip_value)
        
        # return min_loss_policy
        
        max_loss_policy = torch.max(min_loss_policy,max_clip_coef * advantage)
        
        return torch.where(advantage>=0,min_loss_policy,max_loss_policy)
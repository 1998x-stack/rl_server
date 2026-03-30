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
from torch.nn import functional as F
from torch.distributions.beta import Beta
import numpy as np
import gymnasium as gym
from gymnasium.spaces.box import Box
from types import SimpleNamespace

import libs.exps as exps
import algo_envs.algo_base as AlgoBase


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

current_env_name = 'HalfCheetah'

# 训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['GAE_LAMBDA'] = 0.95 # gae lamada
TRAIN_CONFIG['GAMMA'] = 0.99 # 衰减系数
TRAIN_CONFIG['CLIP_COEF'] = 0.2 # pg loss clip
TRAIN_CONFIG['MAX_CLIP_COEF'] = 2 # pg loss max clip
TRAIN_CONFIG['ENT_COEF'] = 0.2# 熵的权重
TRAIN_CONFIG['VLAUE_COEF'] = 4 # value loss 的权重
TRAIN_CONFIG['IS_CLIP_VALUE_LOSS'] = False # 是否clip value loss
TRAIN_CONFIG['LEARNING_RATE'] = 1e-4 # 学习率

# 模型及环境 HalfCheetah
MODEL_CONFIG = dict()
MODEL_CONFIG['NUM_ENVS'] = 32 # 环境数量 microRTS
MODEL_CONFIG['NUM_STEPS'] = 1000 # 一次采样的长度
MODEL_CONFIG['OBS_SPACE'] = (8,) # 状态空间 
MODEL_CONFIG['ACTION_SHAPE'] = Box(-1.0, 1.0, (6,), np.float32) # 动作空间
MODEL_CONFIG['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device
MODEL_CONFIG['MAX_ACTION'] = 1.0

class MujocoBetaAlphaNet(AlgoBase.AlgoBaseNet):    
    def __init__(self):
        super(MujocoBetaAlphaNet,self).__init__()
        obs_dim = TRAIN_ENVS[current_env_name].OBS_DIM
        act_dim = TRAIN_ENVS[current_env_name].ACT_DIM
        hide_dim = TRAIN_ENVS[current_env_name].HIDDEN_DIM
                
        if TRAIN_ENVS[current_env_name].USE_NOISE:
            self.alpha_noisy_layers = [
                    AlgoBase.NoisyLinear(hide_dim, hide_dim),
                    AlgoBase.NoisyLinear(hide_dim, act_dim),
                ]
            self.beta_noisy_layers = [
                    AlgoBase.NoisyLinear(hide_dim, hide_dim),
                    AlgoBase.NoisyLinear(hide_dim, act_dim),
                ]
                            
            self.alpha = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.alpha_noisy_layers[0],
                    nn.ReLU(),
                    self.alpha_noisy_layers[1],
                    nn.Softplus()
                )
            
            self.beta = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.beta_noisy_layers[0],
                    nn.ReLU(),
                    self.beta_noisy_layers[1],
                    nn.Softplus()
                )
        else:
            self.alpha = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Softplus()
                )
            
            self.beta = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Softplus()
                )
                
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )

        log_alpha = -np.log(act_dim, dtype=np.float32) * np.float32(np.e)
        self.log_alpha = nn.Parameter(torch.as_tensor((log_alpha,)))  # trainable parameter
        self.target_entropy = np.log(act_dim) / act_dim
        self.train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['LEARNING_RATE'])
        
    def get_distributions(self, states):
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alphas = self.alpha(states) + 1.0
        betas = self.beta(states) + 1.0
        dists = Beta(alphas, betas)
        return dists
        
    def forward(self, states):
        dists = self.get_distributions(states)
        mus = (dists.mean-0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION']
        return mus
        
    def get_sample_data(self, states): # for samples to train
        dists = self.get_distributions(states)
        sample_actions = dists.sample() # for saving in Experience
        log_probs = dists.log_prob(sample_actions) # get log probs for policy loss
        actions = (sample_actions - 0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION'] # for env stepping
        return sample_actions, actions, log_probs
    
    def get_check_data(self, states): # for checking performence of agents
        dists = self.get_distributions(states)
        actions = (dists.mean-0.5) * 2.0 * MODEL_CONFIG['MAX_ACTION'] # for env stepping
        log_probs = dists.log_prob(dists.mean) # get log probs for stable visualize
        return actions, dists.entropy(), self.log_alpha, self.target_entropy, log_probs
    
    def get_calculate_data(self, states, actions): # for 
        values = self.value(states) # for value loss calculation between values and GAE_returns
        dists = self.get_distributions(states) # for entropy loss
        log_probs = dists.log_prob(actions) # for ratio calculation on policy loss
        return values, log_probs, dists.entropy()   
    
    def update_state(self, version, grads_buffer):
        self.train_optim.zero_grad()
        # 更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad) # TODO: why grad should be floattensor?
        self.train_optim.step()
        
        if TRAIN_ENVS[current_env_name].USE_NOISE:
            for noise_layer in self.alpha_noisy_layers:
                noise_layer.sample_noise()
            for noise_layer in self.beta_noisy_layers:
                noise_layer.sample_noise()
                        
class MujocoBetaAlphaAgent(AlgoBase.AlgoBaseAgent):
    
    def __init__(self,sample_net:MujocoBetaAlphaNet,is_checker):
        super(MujocoBetaAlphaAgent,self).__init__()
        self.model_config = MODEL_CONFIG
        self.device = MODEL_CONFIG['DEVICE']
        self.num_steps = MODEL_CONFIG['NUM_STEPS']
        self.num_envs = MODEL_CONFIG['NUM_ENVS']

        self.sample_net = sample_net
        env_name = TRAIN_ENVS[current_env_name].ENV_NAME
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("MujocoBetaAlpha check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]
        
    def sample_multi_envs(self, model_dict):
        exps=[[] for _ in range(self.num_envs)]
        for _ in range(self.num_steps):
            sample_acitons,actions,log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()[0]
                exps[i].append(
                    [
                        self.states[i],
                        sample_acitons[i],
                        reward_n,
                        done_n,
                        log_probs[i],
                        model_dict['TRAIN_VERSION']
                    ]
                )
                self.states[i] = next_state_n
        return exps
    
    def check_single_env(self):
        step_record_dict = dict()
        
        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []
        alphas = []
        log_alphas = []
        target_entropys = []
        log_probs = []

        while True:
            # self.envs.render()
            mu, entropy, alpha, log_alpha, target_entropy, log_prob = self._get_single_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()[0]
            self.states = next_state_n
            mus.append(mu)
            alphas.append(alpha)
            rewards.append(reward_n)
            entropys.append(entropy)
            log_probs.append(log_prob)
            log_alphas.append(log_alpha)
            target_entropys.append(target_entropy)
            
            steps += 1
            if is_done:
                break
                    
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_mus'] = np.mean(mus)
        # step_record_dict['mean_alphas'] = np.mean(alphas)
        # step_record_dict['mean_log_alphas'] = np.mean(log_alphas)
        # step_record_dict['mean_tar_entropys'] = np.mean(target_entropys)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        
        return step_record_dict
            
    @torch.no_grad()
    def _get_sample_actions(self, states):
        states_v = torch.Tensor(np.array(states)) # use np.stack to replace!
        sample_acitons,actions,log_probs = self.sample_net.get_sample_data(states_v)
        return sample_acitons.cpu().numpy(), actions.cpu().numpy(), log_probs.cpu().numpy()
    
    @torch.no_grad()
    def _get_single_action(self, state):
        state_v = torch.Tensor(np.array(state)) # use np.stack to replace!
        mu,entropy,log_alpha,target_entropy,log_prob = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(), entropy.cpu().numpy(), F.softplus(log_alpha).cpu().numpy(), log_alpha.cpu().numpy(), target_entropy, log_prob.cpu().numpy()


class MujocoBetaAlphaCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self,SHARE_MODEL:MujocoBetaAlphaNet):
        super(MujocoBetaAlphaCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG 
        self.share_model = SHARE_MODEL
        self.device = self.model_config['DEVICE']
        self.calculate_net = MujocoBetaAlphaNet()
        self.calculate_net.to(self.device)
        self.batch_size = 1
        self.max_trajectory = 32
        self.min_sum_rewards = 0.0
        self.exp_buffer = exps.TrajectoryBuffer(capacity=self.max_trajectory)
        
    def generate_grads(self, samples, model_dict):
        
        # TODO: the samples with better performance will be added???
        s_rewards = np.array([s[2] for s in samples])
        s_sum_rewards = s_rewards.sum()
        if (s_sum_rewards > self.min_sum_rewards):
            self.min_sum_rewards = s_sum_rewards
            self.exp_buffer.append(samples)
            
        grads_list = []
        grads = self.generate_one_grads(samples)
        grads_list.append(grads)
        if len(self.exp_buffer) >= self.batch_size:
            samples_list = self.exp_buffer.sample(self.batch_size)
            for samples in samples_list:
                grads = self.generate_one_grads(samples)
                grads_list.append(grads)
                
        return grads_list, model_dict['TRAIN_VERSION']
    
    def generate_one_grads(self,samples):
                
        gamma = self.train_config['GAMMA']
        gae_lambda = self.train_config['GAE_LAMBDA']
        vf_coef = self.train_config['VLAUE_COEF']

        ent_coef = self.train_config['ENT_COEF']
        alpha_coef = 0.0001 # F.softplus(self.calculate_net.log_alpha).detach()
    
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])
        
        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        t_new_values,t_new_log_probs,t_entropys = self.calculate_net.get_calculate_data(t_states,t_actions)
        
        s_advantages = [0]
        s_returns = [0]
        
        
        # 计算优势值
        last_gae = 0.0
        last_return = 0.0

        with torch.no_grad():
            for value, next_value, reward, done, log_prob in zip(
                    reversed(t_new_values[:-1]),
                    reversed(t_new_values[1:]),
                    reversed(s_rewards[:-1]),
                    reversed(s_dones[:-1]),
                    reversed(t_new_log_probs[:-1]),
                ):
                # only one done, so use if-else clause
                if done:
                    gae_delta = reward-value
                    last_gae = gae_delta
                else:
                    gae_delta = reward + gamma * next_value-value
                    last_gae = gae_delta + gamma * gae_lambda * last_gae

                s_advantages.append(last_gae)
                s_returns.append(last_gae + value)
                                
        t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)        
        t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)
                
        t_advantages = t_advantages.reshape(-1, 1)
        t_returns = t_returns.reshape(-1, 1)
        
        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        # discrete ratio
        ratio1 = torch.exp(t_new_log_probs - old_log_probs)
        # prod ratio   
        ratio2 = torch.exp(t_new_log_probs.sum(1)-old_log_probs.sum(1)).reshape(-1, 1).expand_as(ratio1)
        # mixed ratio
        ratio3 = (ratio1+ratio2)/2
        
        # Policy loss
        # pg_loss1 = self.get_pg_loss(ratio1,t_advantages)
        # pg_loss2 = self.get_pg_loss(ratio2,t_advantages)
        # pg_loss4 = (pg_loss1+pg_loss2)/2
        pg_loss3 = self.get_pg_loss(ratio3,t_advantages)
        pg_loss = -torch.mean(pg_loss3)
        # Value loss
        v_loss = F.mse_loss(t_returns, t_new_values) * vf_coef
        # entropy loss
        e_loss = -torch.mean(t_entropys) * ent_coef
        # TODO: what is a_loss and log_alpha?
        a_loss = torch.mean(self.calculate_net.log_alpha * (t_new_log_probs.mean(1)-self.calculate_net.target_entropy).detach())
        loss = pg_loss + v_loss + e_loss + a_loss
        self.calculate_net.zero_grad()
        loss.backward()
                
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
        
        return grads
    
    def get_pg_loss(self, ratio, advantage):
        clip_coef = self.train_config['CLIP_COEF']
        max_clip_coef = self.train_config['MAX_CLIP_COEF']
        clip_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        
        min_loss_policy = torch.min(ratio * advantage, clip_ratio * advantage)        
        max_loss_policy = torch.max(min_loss_policy, max_clip_coef * advantage)
        
        return torch.where(advantage >= 0, min_loss_policy, max_loss_policy)

if __name__ == "__main__":
    net = MujocoBetaAlphaNet()
    agent = MujocoBetaAlphaAgent(net,is_checker=False)
    agent_check = MujocoBetaAlphaAgent(net,is_checker=True)
    calculate = MujocoBetaAlphaCalculate(net)
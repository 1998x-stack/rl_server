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
import gym
import numpy as np
from gym.spaces.box import Box
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace
import algo_envs.algo_base as AlgoBase

TRAIN_ENVS = {
    'Swimmer':SimpleNamespace(**{'env_name': "Swimmer-v3",'obs_dim':8,'act_dim':2,'hide_dim':32,'use_noise':True}),
    'HalfCheetah':SimpleNamespace(**{'env_name': "HalfCheetah-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'use_noise':True}),
    'Ant':SimpleNamespace(**{'env_name': "Ant-v3",'obs_dim':111,'act_dim':8,'hide_dim':256,'use_noise':True}),
    'Hopper':SimpleNamespace(**{'env_name': "Hopper-v3",'obs_dim':11,'act_dim':3,'hide_dim':64,'use_noise':True}),
    'Pusher':SimpleNamespace(**{'env_name': "Pusher-v2",'obs_dim':23,'act_dim':7,'hide_dim':128,'use_noise':True}),
    'Humanoid':SimpleNamespace(**{'env_name': "Humanoid-v3",'obs_dim':376,'act_dim':17,'hide_dim':512,'use_noise':True}),
    'Walker2d':SimpleNamespace(**{'env_name': "Walker2d-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'use_noise':True}),
}

current_env_name = 'Pusher'

#训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['gae_lambda'] = 0.95 # gae lamada
TRAIN_CONFIG['gamma'] = 0.99 # 衰减系数
TRAIN_CONFIG['clip_coef'] = 0.2 # pg loss clip
TRAIN_CONFIG['max_clip_coef'] = 2 # pg loss max clip
TRAIN_CONFIG['ent_coef'] = 0.2# 熵的权重
TRAIN_CONFIG['vf_coef'] = 4 # value loss 的权重
TRAIN_CONFIG['clip_v_loss'] = False # 是否clip value loss
TRAIN_CONFIG['learning_rate'] = 2.5e-4 # 学习率

#模型及环境 HalfCheetah
MODEL_CONFIG = dict()
MODEL_CONFIG['num_envs'] = 32 # 环境数量 microRTS
MODEL_CONFIG['num_steps'] = 1000 # 一次采样的长度
MODEL_CONFIG['obs_space'] = (8,) # 状态空间 
MODEL_CONFIG['action_shape'] = Box(-1.0, 1.0, (6,), np.float32) # 动作空间
MODEL_CONFIG['device'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device


class MujocoNormalNet(AlgoBase.AlgoBaseNet):
    
    def __init__(self):
        super(MujocoNormalNet,self).__init__()
        
        obs_dim = TRAIN_ENVS[current_env_name].obs_dim
        act_dim = TRAIN_ENVS[current_env_name].act_dim
        hide_dim = TRAIN_ENVS[current_env_name].hide_dim
        
        if TRAIN_ENVS[current_env_name].use_noise:
            self.noise_layer_out = AlgoBase.NoisyLinear(hide_dim,act_dim)
            self.noise_layer_hide = AlgoBase.NoisyLinear(hide_dim,hide_dim)
                            
            #normal mu
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
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Tanh()
                )
                
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )
        
    def get_distributions(self,states):
        mus = self.mu(states)
        dists = Normal(mus,torch.exp(self.log_std))
        return dists
        
    def forward(self,states):
        mus = self.mu(states)
        return mus
    
    def get_sample_data(self,states):
        dists = self.get_distributions(states)
        actions = dists.sample()
        log_probs = dists.log_prob(actions)
        return actions,log_probs
    
    def get_check_data(self,states):
        dists = self.get_distributions(states)
        mus = self.mu(states)
        return mus,dists.entropy()
    
    def get_calculate_data(self,states,actions):
        values = self.value(states)
        dists = self.get_distributions(states)
        log_probs = dists.log_prob(actions) 
        return values,log_probs,dists.entropy()   
        
    def update_state(self,version,grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['learning_rate'])
        train_optim.zero_grad()
        #更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()
        
        if TRAIN_ENVS[current_env_name].use_noise:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()
        
class MujocoNormalAgent(AlgoBase.AlgoBaseAgent):
    
    def __init__(self,sample_net:MujocoNormalNet,is_checker):
        super(MujocoNormalAgent,self).__init__()
        self.model_config = MODEL_CONFIG
        self.sample_net = sample_net
        self.device = MODEL_CONFIG['device']
        self.num_steps = MODEL_CONFIG['num_steps']
        self.num_envs = MODEL_CONFIG['num_envs']
        
        
        env_name = TRAIN_ENVS[current_env_name].env_name
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
        else:
            print("MujocoNormal check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()[0]
        
    def sample_multi_envs(self,model_dict):

        exps=[[] for _ in range(self.num_envs)]

        for _ in range(self.num_steps):
            
            actions,log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                # if i == 0:
                #     self.envs[i].render()
                next_state_n, reward_n, done_n, truncated, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()[0]
                    
                exps[i].append([self.states[i],actions[i],reward_n,done_n,log_probs[i],model_dict['train_version']])
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
            #self.envs.render()
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
            #if steps >= self.num_steps:
            #    break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['sum_entropys'] = np.sum(entropys)
        step_record_dict['average_mus'] = np.mean(mus)
        
        return step_record_dict
            
    @torch.no_grad()
    def _get_sample_actions(self,states):
        states_v = torch.Tensor(np.array(states))
        actions,log_probs = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def _get_single_action(self,state):
        state_v = torch.Tensor(np.array(state))
        mu,entropy = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy()
        
class MujocoNormalCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self,share_model:MujocoNormalNet):
        super(MujocoNormalCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG 
        self.share_model = share_model
        self.device = self.model_config['device']
        self.calculate_net = MujocoNormalNet()
        
    
    def generate_grads(self,samples,model_dict):
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        self.calculate_net.to(self.device)
        train_version = model_dict['train_version']
        
        gamma = self.train_config['gamma']
        gae_lambda = self.train_config['gae_lambda']
        ent_coef = self.train_config['ent_coef']
        vf_coef = self.train_config['vf_coef']
    
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
        s_log_probs = np.array([s[4] for s in samples])
        #s_versions = [s[5] for s in samples]

        t_states = torch.Tensor(s_states).to(self.device)
        t_actions = torch.Tensor(s_actions).to(self.device)
        old_log_probs = torch.Tensor(s_log_probs).to(self.device)
        
        t_new_values,t_new_log_probs,t_entropys = self.calculate_net.get_calculate_data(t_states,t_actions)
        
        s_advantages = []
        s_returns = []
        
        s_advantages.append(0)
        s_returns.append(0)
        
        #计算优势值
        last_gae = 0.0
        with torch.no_grad():
            for value,next_value,reward,done in zip(reversed(t_new_values[:-1]),reversed(t_new_values[1:]),
                                                reversed(s_rewards[:-1]),reversed(s_dones[:-1])):
                if done:
                    delta = reward - value
                    last_gae = delta
                else:
                    delta = reward + gamma * next_value - value
                    last_gae = delta + gamma * gae_lambda * last_gae
                    
                s_advantages.append(last_gae)
                s_returns.append(last_gae + value )
                                
        t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)
        t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)
                
        t_advantages = t_advantages.reshape(-1,1)
                
        t_new_log_probs = t_new_log_probs.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        #discrete ratio
        ratio1 = torch.exp(t_new_log_probs-old_log_probs)

        #prod ratio   
        ratio2 = torch.exp(t_new_log_probs.sum(1) - old_log_probs.sum(1)).reshape(-1,1).expand_as(ratio1)
        
        #mixed ratio
        ratio3 = (ratio1+ratio2)/2

        # Policy loss
        pg_loss1 = self.get_pg_loss(ratio1,t_advantages)
        
        pg_loss2 = self.get_pg_loss(ratio2,t_advantages)
        
        pg_loss3 = self.get_pg_loss(ratio3,t_advantages)
        
        pg_loss4 = (pg_loss1+pg_loss2)/2
                
        # Policy loss
        pg_loss = -pg_loss3.mean()
        
        v_loss = F.mse_loss(t_returns.reshape(-1,1), t_new_values)*vf_coef
        
        e_loss = -torch.mean(t_entropys)*ent_coef
        
        loss = pg_loss + v_loss + e_loss

        self.calculate_net.zero_grad()

        loss.backward()
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
        
        return [grads],train_version
    
    def get_pg_loss(self,ratio,advantage):
        
        clip_coef = self.train_config['clip_coef']
        max_clip_coef = self.train_config['max_clip_coef']
        
        base_value = ratio * advantage
        clip_value = torch.clamp(ratio,1.0 - clip_coef,1.0 + clip_coef) * advantage
        min_loss_policy = torch.min(base_value, clip_value)
        
        #return min_loss_policy
        
        max_loss_policy = torch.max(min_loss_policy,max_clip_coef * advantage)
        
        return torch.where(advantage>=0,min_loss_policy,max_loss_policy)
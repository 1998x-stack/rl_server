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
from torch.distributions.beta import Beta
import algo_envs.algo_base as AlgoBase
from types import SimpleNamespace
import libs.exps as Exps

TRAIN_ENVS = {
    'Swimmer':SimpleNamespace(**{'env_name': "Swimmer-v3",'obs_dim':8,'act_dim':2,'hide_dim':32,'use_noise':True}),
    'HalfCheetah':SimpleNamespace(**{'env_name': "HalfCheetah-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'use_noise':True}),
    'Ant':SimpleNamespace(**{'env_name': "Ant-v3",'obs_dim':111,'act_dim':8,'hide_dim':256,'use_noise':True}),
    'Hopper':SimpleNamespace(**{'env_name': "Hopper-v3",'obs_dim':11,'act_dim':3,'hide_dim':64,'use_noise':True}),
    'Pusher':SimpleNamespace(**{'env_name': "Pusher-v2",'obs_dim':23,'act_dim':7,'hide_dim':128,'use_noise':True}),
    'Humanoid':SimpleNamespace(**{'env_name': "Humanoid-v3",'obs_dim':376,'act_dim':17,'hide_dim':512,'use_noise':True}),
    'Walker2d':SimpleNamespace(**{'env_name': "Walker2d-v3",'obs_dim':17,'act_dim':6,'hide_dim':64,'use_noise':True}),
}

current_env_name = 'Swimmer'

#训练参数
TRAIN_CONFIG = dict()
TRAIN_CONFIG['gamma'] = 0.99 # 衰减系数
TRAIN_CONFIG['batch_size'] = 256 # 批次大小
TRAIN_CONFIG['learning_rate'] = 2.5e-4 # 学习率

#模型及环境 HalfCheetah
MODEL_CONFIG = dict()
MODEL_CONFIG['num_envs'] = 32 # 环境数量 microRTS
MODEL_CONFIG['num_steps'] = 1000 # 一次采样的长度
MODEL_CONFIG['obs_space'] = (8,) # 状态空间 
MODEL_CONFIG['action_shape'] = Box(-1.0, 1.0, (6,), np.float32) # 动作空间
MODEL_CONFIG['device'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device
MODEL_CONFIG['max_action'] = 1.0

class MujocoNormalQNet(AlgoBase.AlgoBaseNet):    
    
    def __init__(self):
        super(MujocoNormalQNet,self).__init__()
        
        obs_dim = TRAIN_ENVS[current_env_name].obs_dim
        act_dim = TRAIN_ENVS[current_env_name].act_dim
        hide_dim = TRAIN_ENVS[current_env_name].hide_dim
                
        if TRAIN_ENVS[current_env_name].use_noise:
            self.mu_noisy_layers = [AlgoBase.NoisyLinear(hide_dim, act_dim),AlgoBase.NoisyLinear(hide_dim, hide_dim)]
            self.std_noisy_layers = [AlgoBase.NoisyLinear(hide_dim, act_dim),AlgoBase.NoisyLinear(hide_dim, hide_dim)]
                            
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.mu_noisy_layers[1],
                    nn.ReLU(),
                    self.mu_noisy_layers[0]
                )
            
            self.log_std = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.std_noisy_layers[1],
                    nn.ReLU(),
                    self.std_noisy_layers[0],
                )
        else:
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim))
                )
            
            self.log_std = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                )
                                        
        self.q1_value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim + act_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1)),
            )
        
        self.q2_value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim + act_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )
        
    def get_distris(self,states):
        mus = self.mu(states)
        log_stds = self.log_std(states).clamp(-20,2)
        distris = Normal(mus,torch.exp(log_stds))
        return distris
        
    def forward(self,states):
        mus = self.mu(states)
        return torch.tanh(mus)
    
    def get_sample_data(self,states):
        distris = self.get_distris(states)
        actions = distris.sample()
        return torch.tanh(actions)
    
    def get_check_data(self,states):
        distris = self.get_distris(states)
        mus = self.mu(states)
        return torch.tanh(mus),distris.entropy()
    
    def get_calculate_data(self,states,actions):
        values = self.value(states)
        distris = self.get_distris(states)
        log_probs = distris.log_prob(actions) 
        return values,log_probs,distris.entropy()   
    
    def update_state(self,version,grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=TRAIN_CONFIG['learning_rate'])
        train_optim.zero_grad()
        #更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad)
        train_optim.step()
        
        if TRAIN_ENVS[current_env_name].use_noise:
            for noise_layer in self.alpha_noisy_layers:
                noise_layer.sample_noise()
            for noise_layer in self.beta_noisy_layers:
                noise_layer.sample_noise()
            
    def get_q_values(self,states,actions):
        q_input = torch.cat((states,actions),-1)
        q1_values = self.q1_value(q_input)
        q2_values = self.q2_value(q_input)

        return q1_values,q2_values
    
    def get_mu_q_values(self,states):
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
        self.device = MODEL_CONFIG['device']
        self.num_steps = MODEL_CONFIG['num_steps']
        self.num_envs = MODEL_CONFIG['num_envs']
        
        
        env_name = TRAIN_ENVS[current_env_name].env_name
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset() for i in range(self.num_envs)]
        else:
            print("MujocoNormalQ check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states = self.envs.reset()
        
    def sample_multi_envs(self,model_dict):

        exps=[[] for _ in range(self.num_envs)]

        for _ in range(self.num_steps):
            
            actions,log_probs = self._get_sample_actions(self.states)
            for i in range(self.num_envs):
                # if i == 0:
                #     self.envs[i].render()
                next_state_n, reward_n, done_n, _ = self.envs[i].step(actions[i])                
                if done_n:
                    next_state_n = self.envs[i].reset()
                    
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
            next_state_n, reward_n, is_done, _ = self.envs.step(mu)
            if is_done:
                next_state_n = self.envs.reset()
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            
            steps += 1
            if is_done:
                break
            
            #time.sleep(100)
            #if steps >= self.num_steps:
            #    break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['sum_entropys'] = np.sum(entropys)
        step_record_dict['average_mus'] = np.mean(mus)
        
        return step_record_dict
            
    @torch.no_grad()
    def _get_sample_actions(self,states):
        states_v = torch.Tensor(np.array(states))
        actions = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy()
    
    @torch.no_grad()
    def _get_single_action(self,state):
        state_v = torch.Tensor(np.array(state))
        mu,entropy = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy()

#replaybuffer 只支持 单个 trainer
MAX_BUFFER_SIZE = 100000      
REPEAT_TIME = 1
        
class MujocoNormalQCalculate(AlgoBase.AlgoBaseCalculate):
    
    def __init__(self,share_model:MujocoNormalQNet):
        super(MujocoNormalQCalculate,self).__init__()
        self.train_config = TRAIN_CONFIG
        self.model_config = MODEL_CONFIG 
        self.share_model = share_model
        self.device = self.model_config['device']
        self.calculate_net = MujocoNormalQNet()
        
        self.batch_size = TRAIN_CONFIG['batch_size']
        self.exps_buffer = Exps.ExperienceBuffer(capacity=MAX_BUFFER_SIZE)
    
    def generate_grads(self,samples,model_dict):
        
        gamma = self.train_config['gamma']
        gae_lambda = self.train_config['gae_lambda']
        ent_coef = self.train_config['ent_coef']
        vf_coef = self.train_config['vf_coef']
        q_coef = self.train_config['q_coef']
        q_mu_coef = self.train_config['q_mu_coef']
        
        s_states = np.array([s[0] for s in samples])
        s_actions = np.array([s[1] for s in samples])
        s_rewards = np.array([s[2] for s in samples])
        s_dones = np.array([s[3] for s in samples])
    
        for state,action,reward,done,next_state in zip(s_states[:-1],s_actions[:-1],s_rewards[:-1],s_dones[:-1],s_states[1:]):
            exp = Exps.Experience(state, action, reward, done, next_state)
            self.exps_buffer.append(exp)
            
        if len(self.exps_buffer) < self.batch_size:
            raise ValueError("exps is not Enough")
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        self.calculate_net.to(self.device)
        train_version = model_dict['train_version']
        
        for _ in range(REPEAT_TIME):
            states, actions, rewards, dones, next_states = self.exps_buffer.sample(self.batch_size)
                    
            t_states = torch.Tensor(states).to(self.device)
            t_actions = torch.Tensor(actions).to(self.device)
        
            #q loss
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
                s_returns.append(last_gae + value)
                                
        t_advantages = torch.Tensor(list(reversed(s_advantages))).to(self.device)        
        t_returns = torch.Tensor(list(reversed(s_returns))).to(self.device)
                
        t_advantages = t_advantages.reshape(-1,1)
        t_returns = t_returns.reshape(-1,1)
                
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
        
        v_loss = F.mse_loss(t_returns, t_new_values)*vf_coef
        
        e_loss = -torch.mean(t_entropys)*ent_coef
        
        loss = pg_loss + v_loss + e_loss

        self.calculate_net.zero_grad()

        loss.backward()
        
        t_mu_q_values = self.calculate_net.get_mu_q_values(t_states)
        #不能将Q网络的梯度进行传导
        for para in self.calculate_net.q_value.parameters():
            para.requires_grad = False

        q_mu_loss = -t_mu_q_values.mean() * q_mu_coef / (torch.exp(t_entropys.detach().sum()) + 1)
        q_mu_loss.backward()
        
        for para in self.calculate_net.q_value.parameters():
            para.requires_grad = True
                
        for _ in range(Q_REPEAT_TIME):            
            t_q_values,t_ref_q_values = self.calculate_net.get_q_values(t_states[:-1],t_actions[:-1],t_states[1:])
            
            #计算Q-return
            s_q_returns = []

            with torch.no_grad():
                for reward,done,ref_q_value in zip(s_rewards,s_dones,t_ref_q_values):
                    if done:
                        q_return = reward
                    else:
                        q_return = reward + gamma * ref_q_value
                        
                    s_q_returns.append(q_return)
                                               
            t_q_returns = torch.Tensor(list(s_q_returns)).to(self.device)
            t_q_returns = t_q_returns.reshape(-1,1)   
        
            q_loss = F.mse_loss(t_q_values,t_q_returns) * q_coef / Q_REPEAT_TIME
            q_loss.backward()
                
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
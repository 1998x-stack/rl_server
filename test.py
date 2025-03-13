
import libs.config as config
model_env_name='MicroRTS'
train_net = config.create_net(model_env_name)

from tqdm import trange
import torch 
import torch.nn as nn
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
from torch.distributions.categorical import Categorical
import algo_envs.algo_base as AlgoBase
 
#模型及环境 MicroRTSEnv
model_config = dict()
model_config['env_name'] = "MicroRTSEnv" #其实用不到,只是为了区别
model_config['num_envs'] = 8 # 环境数量 microRTS
model_config['num_steps'] = 512 # 一次采样的长度
model_config['obs_space'] = (10, 10, 27) # 状态空间 
model_config['action_shape'] = [100, 6, 4, 4, 4, 4, 7, 49] # 动作空间
model_config['device'] = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu') # device
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], use_gpu = False):
        self.masks = masks
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    # H = sum(p(x)log(p(x)))
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)
    
    def argmax(self):
        return torch.argmax(self.logits,dim=-1)


class MicroRTSAgent:
    def __init__(self,sample_net, is_checker=False):
        self.model_config = model_config
        self.sample_net = sample_net
        self.num_envs = model_config['num_envs']
        self.num_check_envs = 16
        self.num_steps = model_config['num_steps']
        self.action_shape = model_config['action_shape']
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
            self.env = self.env = MicroRTSVecEnv(
                num_envs=self.num_check_envs,
                max_steps=5000,
                ai2s=[microrts_ai.coacAI for _ in range(self.num_check_envs)],
                map_path='maps/10x10/basesWorkers10x10.xml',
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
        self.obs = self.env.reset()
    
    def __del__(self):
        #del self.env
        pass
        
    def get_units_number(unit_type, bef_obs, ind_obs):
        return int(bef_obs[ind_obs][:, :, unit_type].sum())

    @torch.no_grad()
    def get_action(self,states, type_masks=None):
        
        split_logits, _ = self.sample_net(states)
        
        type_masks = torch.Tensor(type_masks)
        # print(logits.shape,type_masks.shape,len(split_logits),split_logits[0].shape)
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=type_masks)]
        action_components = [multi_categoricals[0].sample()]
        
        # action_masks = torch.ones((20,78))
        action_masks = np.array(self.env.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(len(action_components[0]), -1)
        split_suam = torch.split(torch.Tensor(action_masks), self.action_shape[1:], dim=1)
        multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                    zip(split_logits[1:], split_suam)]
        masks = torch.cat((type_masks, torch.Tensor(action_masks)), 1)
        action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
        action = torch.stack(action_components)
        
        prob=torch.stack([multi_categorical.log_prob(action_) for multi_categorical , action_ in zip(multi_categoricals,action)])
        
        return action.cpu().numpy(), masks.cpu().numpy(),prob.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,states, type_masks=None):
        
        split_logits, _ = self.sample_net(states)
        
        type_masks = torch.Tensor(type_masks)
        # print(logits.shape,type_masks.shape,len(split_logits),split_logits[0].shape)
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=type_masks)]
        action_components = [multi_categoricals[0].argmax()]
        
        # action_masks = torch.ones((20,78))
        action_masks = np.array(self.env.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(len(action_components[0]), -1)
        split_suam = torch.split(torch.Tensor(action_masks), self.action_shape[1:], dim=1)
        multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                    zip(split_logits[1:], split_suam)]
        action_components += [categorical.argmax() for categorical in multi_categoricals[1:]]
        action = torch.stack(action_components)
        
        return action.cpu().numpy()

    def sample_env(self,model_dict):
        exps=[[] for _ in range(self.num_envs)]
        if self.num_steps>0:
            for _ in range(0, self.num_steps):
                self.steps = self.steps + 1
                #self.env.env.render()
                unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_envs, -1)
                with torch.no_grad():
                    action,mask,prob=self.get_action(states=torch.Tensor(self.obs), type_masks=unit_mask)
                    next_obs, rs, done, _ = self.env.step(action.T)
                    for i in range(self.num_envs):
                        exps[i].append([self.obs[i],action.T[i],rs[i],mask[i],done[i],prob.T[i],model_dict['train_version']])
                self.obs=next_obs
        return exps

    def check_env(self):
        step_record_dict = dict()
        step_record_dict['outcomes'] = 0
        step_record_dict['reward'] = 0
        step_record_dict['total_reward'] = 0

        self.states = []
        self.actions = []
        for m in trange(10):
            self.tmp_states = []
            self.tmp_actions = []
            for _ in range(0, 512):
                unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_check_envs, -1)
                with torch.no_grad():
                    action = self.get_check_action(states=torch.Tensor(self.obs), type_masks=unit_mask)
                    self.tmp_states.append(self.obs)
                    self.tmp_actions.append(action.T)
                    
                    next_obs, rs, done, _ = self.env.step(action.T)
                    self.rewards.append(sum(rs) / len(rs))
                    self.total_rewards = self.total_rewards + rs[0]
                for i in range(self.num_check_envs):
                    if done[i]:
                        if MicroRTSAgent.get_units_number(11, self.obs, i) > MicroRTSAgent.get_units_number(12, self.obs, i):
                            self.outcomes.append(1)
                            self.states.append(np.concatenate([a[i:i+1] for a in self.tmp_states], axis=0))
                            self.actions.append(np.concatenate([a[i:i+1] for a in self.tmp_actions], axis=0))
                        else:
                            self.outcomes.append(0)
                        print(sum(self.outcomes)/len(self.outcomes))
                        step_record_dict['outcomes'] = sum(self.outcomes)/len(self.outcomes)
                        step_record_dict['reward'] = sum(self.rewards)/len(self.rewards)
                        if i == 0:
                            step_record_dict['total_reward'] = self.total_rewards
                            self.total_rewards = 0   
                self.obs=next_obs
            
        return step_record_dict
    
agent= MicroRTSAgent(sample_net=train_net, is_checker=True)
    
agent.check_env()
obs = np.concatenate(agent.states,axis=0)
act =  np.concatenate(agent.actions,axis=0)
np.save('obs.npy', obs)
np.save('act.npy', act)
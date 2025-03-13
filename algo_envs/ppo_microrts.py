import torch 
import torch.nn as nn
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
from torch.distributions.categorical import Categorical
import algo_envs.algo_base as AlgoBase

#训练参数
train_config = dict()
train_config['gae_lambda'] = 0.95 # gae lamada
train_config['gamma'] = 0.99 # 衰减系数
train_config['num_reuse'] = 1 # 样本重用次数
train_config['batch_size'] = 512 # 批次大小
train_config['dispatch_size'] = train_config['batch_size']*2 # 分发大小
train_config['clip_coef'] = 0.2 # pg loss clip
train_config['max_clip_coef'] = 4 # pg loss max clip
train_config['ent_coef'] = 0.01 # 熵的权重
train_config['vf_coef'] = 1 # value loss 的权重
train_config['clip_v_loss'] = False # 是否clip value loss
train_config['learning_rate'] = 2.5e-4 # 学习率
train_config['enable_target'] = False #是否允许目标V网络
train_config['update_alpha'] = 0.2

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


class MicroRTSNet(nn.Module):
    
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
        
    @staticmethod
    def get_device():
        return model_config['device']

    def __init__(self):
        super(MicroRTSNet,self).__init__()
        self.device = model_config['device']

        self.network = nn.Sequential(
            MicroRTSNet.layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            MicroRTSNet.layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            MicroRTSNet.layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor_unit = MicroRTSNet.layer_init(nn.Linear(256, 100), std=0.01)
        self.actor_type = MicroRTSNet.layer_init(nn.Linear(256, 6), std=0.01)
        self.actor_move = MicroRTSNet.layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_harvest = MicroRTSNet.layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_return = MicroRTSNet.layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_produce = MicroRTSNet.layer_init(nn.Linear(256, 4), std=0.01)
        self.actor_produce_type = MicroRTSNet.layer_init(nn.Linear(256, 7), std=0.01)
        self.actor_attack = MicroRTSNet.layer_init(nn.Linear(256, 49), std=0.01)
        self.critic = MicroRTSNet.layer_init(nn.Linear(256, 1), std=1)

    # def forward(self, x: torch.Tensor) -> Tuple(torch.Tensor, torch.Tensor):
    def forward(self, x: torch.Tensor):# -> Tuple(torch.Tensor, torch.Tensor):
        x = x.permute((0, 3, 1, 2))
        obs = self.network(x)
        return [self.actor_unit(obs), self.actor_type(obs), self.actor_move(obs), self.actor_harvest(obs), self.actor_return(obs), self.actor_produce(obs), self.actor_produce_type(obs), self.actor_attack(obs)], self.critic(obs)
 
    def update_state(self,version,grads_buffer):
        train_optim = torch.optim.Adam(params=self.parameters(), lr=train_config['learning_rate'])
        train_optim.zero_grad()
        #更新网络参数
        for param, grad in zip(self.parameters(), grads_buffer):
            param.grad = torch.FloatTensor(grad).to(model_config['device'])
        train_optim.step()
        
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


        for _ in range(0, 512):
            unit_mask = np.array(self.env.vec_client.getUnitLocationMasks()).reshape(self.num_check_envs, -1)
            with torch.no_grad():
                action = self.get_check_action(states=torch.Tensor(self.obs), type_masks=unit_mask)
                next_obs, rs, done, _ = self.env.step(action.T)
                self.rewards.append(sum(rs) / len(rs))
                self.total_rewards = self.total_rewards + rs[0]
            for i in range(self.num_check_envs):
                if done[i]:
                    if MicroRTSAgent.get_units_number(11, self.obs, i) > MicroRTSAgent.get_units_number(12, self.obs, i):
                        self.outcomes.append(1)
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
    
    
class MicroRTSCalculate:
    def __init__(self,share_model):
        self.train_config = train_config
        self.model_config = model_config
        self.num_reuse = self.train_config['num_reuse']
        self.dispatch_size = self.train_config['dispatch_size']
        self.batch_size = self.train_config['batch_size']
        self.device = self.model_config['device']
        self.share_model = share_model.to(device=self.device)
        self.states = torch.Tensor([]).to(device=self.device)
        self.actions = torch.Tensor([]).to(device=self.device)
        self.returns = torch.Tensor([]).to(device=self.device)
        self.masks = torch.Tensor([]).to(device=self.device)
        self.probs = torch.Tensor([]).to(device=self.device)
        self.advantages = torch.Tensor([]).to(device=self.device)
        
        self.calculate_net = MicroRTSNet()
        #self.calculate_net = self.share_model
        
    def generate_grads(self,samples,model_dict):
        
        self.calculate_net.load_state_dict(self.share_model.state_dict())
        train_version = model_dict['train_version']
                    
        ent_coef = self.train_config['ent_coef']
        vf_coef = self.train_config['vf_coef']
        
        
        len_samples = len(samples)
        dones = torch.zeros((len_samples,)).to(device=self.device)
        rewards = torch.zeros((len_samples,)).to(device=self.device)
        b_states = torch.zeros((len_samples,)+self.model_config['obs_space']).to(device=self.device)
        b_actions = torch.zeros((len_samples, len(self.model_config['action_shape']))).to(device=self.device)
        b_masks = torch.ones((len_samples, sum(self.model_config['action_shape']))).to(device=self.device)
        b_log_probs = torch.zeros((len_samples, len(self.model_config['action_shape']))).to(device=self.device)

        gamma = self.train_config['gamma']
        gae_lambda = self.train_config['gae_lambda']
                    
        for i in range(len_samples):
            b_states[i] = torch.Tensor(samples[i][0]).to(device=self.device)
            b_actions[i] = torch.Tensor(samples[i][1]).to(device=self.device)
            rewards[i] = torch.Tensor(np.array(samples[i][2]).reshape(-1,1)).to(device=self.device)
            b_masks[i] = torch.Tensor(samples[i][3]).to(device=self.device)
            dones[i] = torch.Tensor(np.array(samples[i][4]).reshape(-1,1)).to(device=self.device)
            b_log_probs[i] = torch.Tensor(np.array(samples[i][5])).to(device=self.device)  
        
        
        new_log_prob, entropy, new_values = self.get_prob_entropy_value(b_states, actions=b_actions.T, masks=b_masks)
                

        with torch.no_grad():
            last_gae_lam = 0
            b_advantages = torch.zeros((len_samples,))
            b_returns = torch.zeros((len_samples,))
            for t in reversed(range(len_samples - 1)):
                next_non_terminal = 1.0 - dones[t]
                next_values = new_values[t + 1]
                delta = rewards[t] + gamma * next_values * next_non_terminal - new_values[t]
                b_advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                b_returns[t] = b_advantages[t] + new_values[t]
                                
        b_advantages = b_advantages.reshape(-1,1)
                        
        new_log_prob = new_log_prob.to(self.model_config['device'])
        entropy = entropy.to(self.model_config['device'])
        
        ratio1 = (new_log_prob - b_log_probs).exp()
        ratio2 = (new_log_prob.sum(1)-b_log_probs.sum(1)).exp().reshape(-1,1).expand_as(ratio1)
        
        # ratio2 = ratio1.prod(1,keepdim=True).expand_as(ratio1)
        # ratio2 = AlgoBase.GradCoef.apply(ratio2,1.0/ratio2.shape[1])
        
        ratio3 = 0.5*ratio1+0.5*ratio2
                            
        # ratio = torch.where(b_advantages >= 0,torch.where(ratio <= 1 + clip_coef,ratio,ratio/ratio.detach()*(1 + clip_coef)),
        #                      torch.where(ratio >= 1 - clip_coef,ratio,ratio/ratio.detach()*(1 - clip_coef)))
        
        pg_loss1 = self.get_pg_loss(ratio1,b_advantages)
        
        pg_loss2 = self.get_pg_loss(ratio2,b_advantages)
        
        pg_loss3 = self.get_pg_loss(ratio3,b_advantages)
        
        pg_loss4 = (pg_loss1+pg_loss2)/2
        
        pg_loss5 = pg_loss1+pg_loss2
        
        # Policy loss
        pg_loss = -pg_loss3.mean()
        
        entropy_loss = -entropy.mean()
        


        v_loss = ((new_values - b_returns) ** 2).mean()

        loss = pg_loss + ent_coef * entropy_loss + v_loss*vf_coef

        self.calculate_net.zero_grad()

        loss.backward()
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in self.calculate_net.parameters()
        ]
                
        return [grads],train_version
                   
    def get_prob_entropy_value(self,states, actions, masks, action_space = [100, 6, 4, 4, 4, 4, 7, 49]):
        split_logits, value = self.calculate_net(states)
        split_masks = torch.split(masks, action_space, dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(actions, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return log_prob.T, entropy.T, value.reshape((-1,))
    
    def get_pg_loss(self,ratio,advantage):
        
        clip_coef = self.train_config['clip_coef']
        max_clip_coef = self.train_config['max_clip_coef']
        
        base_value = ratio * advantage
        clip_value = torch.clamp(ratio,1.0 - clip_coef,1.0 + clip_coef) * advantage
        min_loss_policy = torch.min(base_value, clip_value)
        max_loss_policy = torch.max(min_loss_policy,max_clip_coef * advantage)
        
        return torch.where(advantage>=0,min_loss_policy,max_loss_policy)
        



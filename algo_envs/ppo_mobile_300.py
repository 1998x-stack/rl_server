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




class Mobile300Net(nn.Module):
    def __init__(self):
        super(Mobile300Net,self).__init__()
        
        self_obs_dim = 10
        plaryer_heros_obs_dim = 10
        enemy_heros_obs_dim = 10
        player_creeps_obs_dim = 10
        enemy_creeps_obs_dim = 10
        player_turrets_obs_dim = 10
        enemy_turrets_obs_dim = 10
        
        near_obs_dim = 9
        mini_map_obs_dim = 9
        
        game_states_dim = 10
        
        feed_hide_dim = 10
        
        action_base_dim = feed_hide_dim * 9
        attention_base_dim = feed_hide_dim * 8
        
        self.self_base = nn.Sequential(
            nn.Linear(self_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.player_heros_base = nn.Sequential(
            nn.Linear(plaryer_heros_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.enemy_heros_base = nn.Sequential(
            nn.Linear(enemy_heros_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.player_creeps_base = nn.Sequential(
            nn.Linear(player_creeps_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.enemy_creeps_base = nn.Sequential(
            nn.Linear(enemy_creeps_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.player_turrets_base = nn.Sequential(
            nn.Linear(player_turrets_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.enemy_turrets_base = nn.Sequential(
            nn.Linear(enemy_turrets_obs_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
        self.mini_near_base = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.Flatten(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            )
        
        self.game_states_base = nn.Linear(game_states_dim,feed_hide_dim),
        
        self.action_base = nn.Sequential(
            nn.Linear(action_base_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
            )
        
        self.attention_base = nn.Sequential(
            nn.Linear(attention_base_dim,feed_hide_dim),
            nn.ReLU(),
            nn.Linear(feed_hide_dim,feed_hide_dim),
        )
        
    def forward(self,states):
        self_out = self.self_base(states)
        player_heros_out = self.player_heros_base(states)
        enemey_heros_out = self.enemy_heros_base(states)
        player_creeps_out = self.player_creeps_base(states)
        enemy_creeps_out = self.enemy_creeps_base(states)
        player_turrets_out = self.player_turrets_base(states)
        enemy_turrets_out = self.enemy_turrets_base(states)
        mini_near_out= self.mini_near_base(states)
        game_states_out = self.game_states_base(states)
        
        action_out = self.action_base(torch.cat((self_out,player_heros_out,enemey_heros_out,player_creeps_out,
                                                 enemy_creeps_out,player_turrets_out,enemy_turrets_out,
                                                 mini_near_out,game_states_out),1))
        
        attention_out = self.attention_base(torch.cat((self_out,player_heros_out,enemey_heros_out,player_creeps_out,
                                                 enemy_creeps_out,player_turrets_out,enemy_turrets_out,
                                                 mini_near_out,game_states_out),1))
        
        
        return 
    
    

if __name__ == "__main__":
    train_net = Mobile300Net()
    print(train_net)
    import numpy as np
    parameters = sum([np.prod(p.shape) for p in train_net.parameters()])
    print(train_net,parameters)
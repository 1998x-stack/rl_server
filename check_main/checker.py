# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import torch.multiprocessing as mp

import time
import libs.config as config
from tensorboardX import SummaryWriter
import libs.utils as utils
import libs.log as log

class Checker:

    def __init__(self, model_dict,share_model,env_name,log:log.Log) -> None:
    
        self.model_dict = model_dict
        self.share_model = share_model
        self.self_version = model_dict['train_version']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"
 
    def process_function(self):
        
        #设置随机种子
        utils.setup_seed()

        check_net = config.create_net(self.env_name)
        #check_net = check_net.to(check_net.get_device())

        check_agent = config.create_agent(self.env_name,check_net,is_checker=True)
        
        writer = SummaryWriter(comment=self.comment)

        while True:
            if self.model_dict['is_exit']:
                break
            
            try:   
                
                check_net.load_state_dict(self.share_model.state_dict())       
                check_net.train(False)         
                self.self_version=self.model_dict['train_version']
                
                infos = check_agent.check_single_env()
                
                if isinstance(infos,dict):
                    for (key,value) in  infos.items():
                        writer.add_scalar(key, value, self.self_version)
                                                                    
                time.sleep(0)
            except:
                self.log.log_exception(print_screen=True)
                continue
        writer.close()
        
        self.log.log_info('exit checker processid ' + str(self.process.pid),print_screen=True)
                
    def run_checker(self,comment=None):
        if comment is not None:   
            self.comment = comment
        self.process=mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start checker processid ' + str(self.process.pid),print_screen=True)

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)
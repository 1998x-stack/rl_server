# -*- coding: utf-8 -*-
"""遗留 Checker 子进程：周期性评估并记录标量指标。"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import time
import torch.nn as nn
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL
from tensorboardX import SummaryWriter

import libs.log as log
import libs.utils as utils
import libs.config as config

class Checker:

    def __init__(
                self,
                model_dict,
                SHARE_MODEL: nn.Module,
                env_name,
                log: log.Log
            ) -> None:
    
        self.model_dict = model_dict
        self.share_model = SHARE_MODEL
        self.self_version = model_dict['TRAIN_VERSION']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"
 
    def process_function(self):
        # 设置随机种子
        utils.setup_seed()
        check_net = config.create_net(self.env_name)
        check_agent = config.create_agent(self.env_name,check_net,is_checker=True)
        writer = SummaryWriter(comment=self.comment)
        while True:
            if self.model_dict['is_exit']:
                break
            try:   
                
                check_net.load_state_dict(self.share_model.state_dict())       
                check_net.train(False)         
                self.self_version=self.model_dict['TRAIN_VERSION']
                # 在环境中运行，检测此时模型效果
                infos = check_agent.check_single_env()
                if isinstance(infos,dict):
                    for (key,value) in  infos.items():
                        writer.add_scalar(key, value, self.self_version)
                time.sleep(0) # ​触发线程重新调度，让步其他线程
            except:
                self.log.log_exception(print_screen=True)
                continue
        writer.close()
        
        self.log.log_info('exit checker processid ' + str(self.process.pid), print_screen=True)
                
    def run_checker(self, comment: str=None):
        if comment is not None:   
            self.comment = comment
        self.process=mp.Process(target=self.process_function)  # 创建进程
        self.process.start() # 启动进程
        self.log.log_info('start checker processid ' + str(self.process.pid), print_screen=True)

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate() # 强制终止子进程
                self.process.join() # 等待进程结束
        except:
            self.log.log_exception(print_screen=True)
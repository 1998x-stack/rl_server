# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
"""
训练体,被 train_main_sample所用
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import time, queue
import numpy as np
from typing import Dict

import torch.nn as nn
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL

import libs.log as log
import libs.utils as utils
import libs.config as config
import libs.redis_cache as redis_cache

class TrainerRedis:
    
    """
    id: 实例标记
    model_dict: 共享模型参数
    redis_log: 直接从redis获取样本
    grads_queue: 梯度整合队列
    env_name: 环境名字
    log: 日志
    """
    def __init__(self, 
                 idx: int, 
                 model_dict: Dict, 
                 SHARE_MODEL: nn.Module, 
                 sample_queue: mp.Queue,
                 grads_queue: mp.Queue,
                 env_name: str,
                 log:log.Log
            ):
        self.log = log
        self.process = None
        self.trainer_id = idx
        self.env_name = env_name
        
        self.model_dict = model_dict
        self.share_model = SHARE_MODEL
        
        self.sample_queue = sample_queue
        self.grads_queue = grads_queue
    
    # 进程函数    
    def process_function(self):
        # 设置随机种子
        utils.setup_seed()
           
        calculate = config.create_calculate(self.env_name, self.share_model)
        exps_redis_config = config.get_current_redis_exps_config()
        exps_redis_cache = redis_cache.RedisCache(self.log, exps_redis_config)
                
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                samples,exps_version = exps_redis_cache.pop_exps()
                if samples is not None:                               
                    grads_list,train_version = calculate.generate_grads(samples,self.model_dict)
                    
                    for grads in grads_list:
                        grads_info = dict()
                        grads_info['grads'] = grads
                        grads_info['grads_version'] = train_version
                        grads_info['sample_version'] = exps_version
                        self.grads_queue.put(grads_info)
                time.sleep(0) # ​触发线程重新调度，让步其他线程
            
            except queue.Full:
                time.sleep(1)
                continue
            except Exception:
                self.log.log_exception(print_screen=True)
                continue
        # 保证退出
        try:
            del exps_redis_cache
        except:
            self.log.log_exception(print_screen=True)
        
        self.log.log_info('exit trainer processid ' + str(self.process.pid) + " trainerid " + str(self.trainer_id), print_screen=True)
        
    def run_trainer_redis(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start trainer processid ' + str(self.process.pid) + " trainerid " + str(self.trainer_id), print_screen=True)
        
    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)
        
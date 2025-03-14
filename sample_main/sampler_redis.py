# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

"""
1 初始化各种结构
2 采样
3 发送采样数据到exps queue
4 更新网络参数
"""

import time
import torch.nn as nn
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL
from typing import Dict

import libs.log as log
import libs.redis_cache as redis_cache
import libs.config as config
import libs.utils as utils

class SamplerRedis:

    """
    id: 实例标记
    model_dict: 共享模型参数
    redis_log: redis日志信息
    env_name: 环境名称
    log: 日志
    """
    def __init__(
                self,
                idx: int,
                model_dict: Dict,
                SHARE_MODEL: nn.Module,
                env_name: str,
                log: log.Log
            ) -> None:
        self.sampler_id = idx
        self.model_dict = model_dict
        self.share_model = SHARE_MODEL
        self.env_name = env_name
        self.log = log
        self.process=None
        
    def process_function(self):

        # 设置随机种子
        utils.setup_seed() # 用得着一直设置随机种子吗
        sample_agent = config.create_agent(self.env_name,self.share_model)
        exps_redis_config = config.get_current_redis_exps_config()
        exps_redis_cache = redis_cache.RedisCache(self.log,exps_redis_config)

        while True:
            if self.model_dict['is_exit']:
                break
            try:
                exps_list = sample_agent.sample_multi_envs(self.model_dict)
                if exps_list is not None:
                    for exps in exps_list:
                        push_result = exps_redis_cache.push_exps(exps, self.model_dict['TRAIN_VERSION'])
                        # 如果发送redis失败，则暂时停止采样5秒
                        if not push_result:
                            time.sleep(5)
                else:
                    self.log.log_info("sample_agent.sample_multi_envs return None", print_screen=True)
                time.sleep(0) # ​触发线程重新调度，让步其他线程
            except:
                self.log.log_exception(print_screen=True)
                continue
            
        # 保证退出
        try:
            del sample_agent
            del exps_redis_cache
        except:
            self.log.log_exception(print_screen=True)
            
        self.log.log_info('exit sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id), print_screen=True)
                
    def run_sampler_redis(self):
        self.process=mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id), print_screen=True)

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)

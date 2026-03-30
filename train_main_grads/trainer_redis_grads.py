# -*- coding: utf-8 -*-
"""梯度分离架构下的 Trainer（遗留）：将梯度推送至 Redis 供 ``grads_main`` 聚合。"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL
import time
import libs.log as log
import libs.utils as utils
import libs.redis_cache as redis_cache
import libs.config as config
import queue

class TrainerRedisGrads:
    
    """
    id: 实例标记
    model_dict: 共享模型参数
    sample_redis: 直接从redis获取样本
    grads_queue: 梯度整合队列
    env_name: 环境名字
    log: 日志
    """
    def __init__(self, idx, model_dict,SHARE_MODEL,grads_queue,env_name,log:log.Log):
        self.trainer_id = idx
        self.model_dict = model_dict
        self.share_model = SHARE_MODEL
        self.grads_queue = grads_queue
        self.env_name = env_name
        self.process = None
        self.log = log
    
    # 进程函数    
    def process_function(self):

        # 设置随机种子
        utils.setup_seed()

        calculate = config.create_calculate(self.env_name,self.share_model)
        
        exps_redis_config = config.get_current_redis_exps_config()
        exps_redis_cache = redis_cache.RedisCache(self.log,exps_redis_config)
                        
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                samples,exps_version = exps_redis_cache.pop_exps()
                if samples is not None:
                    grads_list,train_version = calculate.generate_grads(samples,self.model_dict)
                    for grads in grads_list:
                        grads_info = dict()
                        grads_info['grads_version'] = train_version
                        grads_info['sample_version'] = exps_version
                        grads_info['grads'] = grads         
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
        
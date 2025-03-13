
"""
1 初始化各种结构
2 采样
3 发送采样数据到exps queue
4 更新网络参数
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import torch.multiprocessing as mp
import torch.nn as nn

import time
import libs.log as log
import libs.config as config
import queue
import libs.utils as utils

class Sampler:

    """
    id: 实例标记
    model_dict: 共享模型参数
    sample_queue: 原始样本队列
    env_name: 环境名称
    log: 日志
    """
    def __init__(self, id,model_dict,share_model:nn.Module,sample_queue,env_name,log:log.Log) -> None:
        self.sampler_id = id
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.sample_queue = sample_queue
        self.process=None
 
    def process_function(self):

        #设置随机种子
        utils.setup_seed()
        sample_agent = config.create_agent(self.env_name,self.share_model)
                
        while True:
            if self.model_dict['is_exit']:
                break
            
            try:                                                    
                exps_list = sample_agent.sample_env(self.model_dict)
                
                if exps_list is not None:
                    #为了兼容多线程环境，这里exps_list必须是轨迹list
                    for exps in exps_list:
                        exps_info = dict()
                        exps_info['sample_version'] = self.model_dict['train_version']
                        exps_info['exps'] = exps         
                        self.sample_queue.put(exps_info)
                else:
                    self.log.log_info("sample_agent.sample_env return None",print_screen=True)
                                                        
                time.sleep(0)
            
            #如果队列满了，则需要暂停采样  
            except queue.Full:
                time.sleep(5)
                continue
            except:
                self.log.log_exception(print_screen=True)
                continue
            
        #保证环境退出
        try:
            del sample_agent
        except:
            self.log.log_exception(print_screen=True)
        
        self.log.log_info('exit sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id),print_screen=True)
                
    def run_sampler(self):
        self.process=mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info('start sampler processid ' + str(self.process.pid) + " samplerid " + str(self.sampler_id),print_screen=True)

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except:
            self.log.log_exception(print_screen=True)

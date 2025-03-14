# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
"""
0 启动logger
1 sample_queue 数据结构
2 启动redis
3 读取模型再启动Sampler
while True
    5 检测退出
        join各个进程
    6 从 sample queue 获取 样本
    7 发送采样数据到redis
    8 更新网络到sampler

9 清理退出
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))


import time
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL

import sampler_redis
from libs.log import Log
import libs.utils as utils
import libs.config as config
from libs.redis_cache import RedisCache

if __name__ == '__main__':
    
    # 设置多进程模式
    utils.setup_mp()
    # 设置随机种子
    utils.setup_seed()
    # 环境名称
    model_env_name = config.get_current_env_name() # "MicroRTSEnv"
    # 设置参数
    queue_config = config.get_current_queue_config()
    model_redis_config = config.get_current_redis_MODEL_CONFIG()
    # 启动日志
    sample_log = Log("sample_main")
    # 网络
    sample_net = config.create_net(model_env_name)
    sample_net.share_memory()

    # 当前网络参数版本
    current_train_version = 0
    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = current_train_version
        
    # 启动redis
    model_redis_cache = RedisCache(sample_log,model_redis_config)
    
    # 加载初始模型版本
    while True:
        try:
            new_version = model_redis_cache.get_train_version()
            is_model_updated = model_redis_cache.get_train_model(sample_net)
            if (new_version is not None) and is_model_updated:
                current_train_version = new_version
                model_dict['TRAIN_VERSION'] = current_train_version
                break
        except:
            sample_log.log_exception(print_screen=True)
            exit()
    
    # 容器
    samplers = []
    # 实例化Sampler
    for i in range(queue_config['num_sampler']):
        l_sampler = sampler_redis.SamplerRedis(
                                    i,
                                    model_dict=model_dict, 
                                    SHARE_MODEL=sample_net,
                                    env_name=model_env_name,
                                    log=sample_log
                            )
        samplers.append(l_sampler)
        l_sampler.run_sampler_redis()

    sample_log.log_info("start run sample_main_redis", print_screen=True)

    while True:
        try:
            # 最开始必须先进行退出检测
            exit_work = utils.exit_run()
            exit_flag = model_redis_cache.get_exit_flag()
            if exit_flag is not None:
                exit_work = exit_work or bool(exit_flag)
            if exit_work:
                sample_log.log_info("start exit sample_main_redis", print_screen=True)
                model_dict['is_exit'] = True
                # 等待进程退出
                for l_sampler in samplers:
                    l_sampler.stop()
                del model_redis_cache
                sample_log.log_info("end exit sample_main_redis", print_screen=True)
                break
            # 更新网络
            new_version = model_redis_cache.get_train_version()
            if (new_version is not None) and (new_version > current_train_version):
                is_model_updated = model_redis_cache.get_train_model(sample_net)
                if is_model_updated: 
                    current_train_version = new_version
                    model_dict['TRAIN_VERSION'] = current_train_version

            time.sleep(0) # ​触发线程重新调度，让步其他线程
        except:
            sample_log.log_exception(print_screen=True)
            
    sample_log.log_info("exit OK", print_screen=True)

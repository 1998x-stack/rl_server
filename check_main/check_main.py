# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import time
import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL

from tensorboardX import SummaryWriter

import checker
import libs.log as log
import libs.utils as utils
import libs.config as config
import libs.redis_cache as redis_cache

if __name__ == '__main__':
    
    # 设置多进程模式
    utils.setup_mp()
    #设置随机种子
    utils.setup_seed()
    #环境名称
    model_env_name = config.get_current_env_name()
    # 设置参数
    model_redis_config = config.get_current_redis_MODEL_CONFIG()
    #启动日志
    check_log = log.Log("check_main")
    #网络
    check_net = config.create_net(model_env_name)
    check_net.share_memory()
    #当前网络参数版本
    current_train_version = 0
    model_dict = mp.Manager().dict() # 进程间共享的字典
    model_dict['train_version'] = current_train_version
    model_dict['is_exit'] = False
    #启动redis
    model_redis_cache = redis_cache.RedisCache(check_log, model_redis_config)
    #加载初始模型版本
    while True:
        try:
            new_version = model_redis_cache.get_train_version()
            is_model_updated = model_redis_cache.get_train_model(check_net)
            if (new_version is not None) and is_model_updated:
                current_train_version = new_version
                model_dict['train_version'] = current_train_version
                break
        except:
            check_log.log_exception(print_screen=True)
            exit()
    #启动checker
    train_checker = checker.Checker(
                                model_dict=model_dict,
                                share_model=check_net,
                                env_name=model_env_name,
                                log=check_log
                            )
    train_checker.run_checker()
    check_log.log_info("start run check_main", print_screen=True)
    while True:
        try:
            #退出检测
            if utils.exit_run():
                check_log.log_info("start exit check_main", print_screen=True)
                del model_redis_cache
                model_dict['is_exit'] = True
                train_checker.stop()
                check_log.log_info("end exit check_main", print_screen=True)
                break
            #更新网络
            new_version = model_redis_cache.get_train_version()
            if (new_version is not None) and (new_version > current_train_version):
                is_model_updated = model_redis_cache.get_train_model(check_net)
                if is_model_updated: 
                    current_train_version = new_version
                    model_dict['train_version'] = current_train_version
            time.sleep(0) # ​触发线程重新调度，让步其他线程
        except:
            check_log.log_exception(print_screen=True)
    check_log.log_info("exit OK", print_screen=True)
# -*- coding: utf-8 -*-
"""
用于整合 grads 的服务器。
1 启动logger
2 启动redis
3 读取新的梯度信息，累计后更新网络。

:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import argparse
import libs.log as log
import libs.config as config
import libs.utils as utils
import libs.redis_cache as redis_cache


if __name__ == '__main__':
    
    # doto 获取环境参数
    parser = argparse.ArgumentParser()
    
    # 设置随机种子
    utils.setup_seed()
    
    # 启动logger
    grads_log = log.Log("grads_main")
    
    # 环境名称
    model_env_name = config.get_current_env_name() # "MicroRTSEnv"
    model_prefix= "grads_main"
    model_version = None
    
    # 运行参数
    queue_config = config.get_current_queue_config()
    grads_redis_config = config.get_current_redis_grads_config()
    model_redis_config = config.get_current_redis_MODEL_CONFIG()
    
    # 初始化网络
    grads_net = config.create_net(model_env_name)
    
    # 加载模型数据 
    current_train_version = utils.get_model_from_file(grads_net,f"{model_prefix}_{model_env_name}",model_version)
    
    if current_train_version is None:
        grads_log.log_info("has no model data and starts a new train", print_screen=True)
        current_train_version = 0
        
    # 启动redis
    grads_redis_cache = redis_cache.RedisCache(grads_log,grads_redis_config)
    grads_redis_cache.clear_data()
    
    model_redis_cache = redis_cache.RedisCache(grads_log,model_redis_config)
    model_redis_cache.clear_data()
    
    # 同步模型数据到redis 分为 train 和 sample
    model_redis_cache.set_train_version_model(current_train_version,grads_net)

    grads_log.log_info("start run grads_main", print_screen=True)
    
    grads_buffer = None
    grads_count = 0
    
    # 逻辑循环
    while True:
        try:
            # 退出检测
            if utils.exit_run():
                grads_log.log_info("start exit grads_main", print_screen=True)
                
                # 保存当前模型版本
                utils.save_model_to_file(grads_net,f"{model_prefix}_{model_env_name}",current_train_version)
                
                # 通知trainer 和 sampler服务器退出
                model_redis_cache.set_exit_flag(1)

                del model_redis_cache
                del grads_redis_cache
                                                
                grads_log.log_info("end exit grads_main", print_screen=True)
                break
            
            # 整合梯度
            if grads_count >= queue_config['batch_update_grads_server']:
                
                # 更新版本
                current_grads_version = current_grads_version + 1
                grads_net.update_state(current_train_version,grads_buffer)
                model_redis_cache.set_train_version_model(current_grads_version,grads_net)
       
                grads_buffer = None
                grads_count = 0
            else:
                grads_item,grads_version,sample_version = grads_redis_cache.pop_grads()
                
                if grads_item is not None:
                    grads_count = grads_count+1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for target_grad, grad in zip(grads_buffer, grads_item):
                            target_grad += grad
                    
        except:
            grads_log.log_exception(print_screen=True)
        
    grads_log.log_info("exit OK", print_screen=True)
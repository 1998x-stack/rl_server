# -*- coding: utf-8 -*-
"""Redis 模式训练主程序（遗留）：多 ``sample_main`` 与模型 Redis 协同，主进程聚合梯度并广播权重。"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch.multiprocessing as mp # 计算密集型，而非IO密集型，GIL
import time
import torch
import queue

import trainer_redis
import libs.log as log
import libs.utils as utils
import libs.config as config
import libs.redis_cache as redis_cache
    
if __name__ == '__main__':

    # 设置多进程模式
    utils.setup_mp()
    
    # 设置随机种子
    utils.setup_seed()
    # 启动日志
    train_log = log.Log("train_main_redis")
    model_prefix= "train_main_redis"
    model_env_name = config.get_current_env_name() # "MicroRTSEnv"
    model_version = None
    # 设置参数
    queue_config = config.get_current_queue_config()
    model_redis_config = config.get_current_redis_MODEL_CONFIG()
    # 梯度队列
    grads_queue = mp.Queue(maxsize=queue_config['len_grads_queue']) 

    # 梯度整合网络
    train_net = config.create_net(model_env_name)
    train_net.share_memory()
    
    # 加载模型数据 
    current_train_version = utils.get_model_from_file(train_net, f"{model_prefix}_{model_env_name}", model_version)
    
    if current_train_version is None:
        train_log.log_info("has no model data and starts a new train", print_screen=True)
        current_train_version = 0

    # 当前网络参数版本
    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = current_train_version

    # 启动redis
    model_redis_cache = redis_cache.RedisCache(train_log,model_redis_config)
    model_redis_cache.clear_data()
    
    # 清理经验redis
    exps_redis_config = config.get_current_redis_exps_config()
    exps_redis_cache = redis_cache.RedisCache(train_log,exps_redis_config)
    exps_redis_cache.clear_data()
    del exps_redis_cache
    
    # 各种训练容器
    trainers = []
                  
    for i in range(queue_config['num_trainer']):
        l_trainer = trainer_redis.TrainerRedis(
                                    idx=i,
                                    model_dict=model_dict,
                                    SHARE_MODEL=train_net,
                                    grads_queue=grads_queue,
                                    env_name=model_env_name,
                                    log=train_log
                                )
        trainers.append(l_trainer)
        l_trainer.run_trainer_redis()
        
    # 设置采样模型数据
    model_redis_cache.set_train_version_model(current_train_version,train_net)
        
    train_log.log_info("start run train_main_redis", print_screen=True)
    
    grads_count = 0
    grads_buffer = None

    while True:
        try:
            # 退出检测
            if utils.exit_run():
                train_log.log_info("start exit train_main_redis", print_screen=True)
                # 保存当前模型版本
                utils.save_model_to_file(train_net, f"{model_prefix}_{model_env_name}", current_train_version)
                # 通知sampler服务器退出
                model_redis_cache.set_exit_flag(1)
                model_dict['is_exit'] = True
                for l_trainer in trainers:
                    l_trainer.stop()
                del model_redis_cache
                train_log.log_info("end exit train_main_redis", print_screen=True)
                break
                        
            # 整合梯度
            if grads_count >= queue_config['num_update_grads']:
                # 更新版本
                current_train_version = current_train_version + 1
                train_net.update_state(current_train_version,grads_buffer)
                model_dict['TRAIN_VERSION'] = current_train_version
                
                # 重置梯度
                grads_count = 0           
                grads_buffer = None
            else:
                try:
                    grads_info = grads_queue.get(block=False) # 当队列为空时，立即抛出queue.Empty异常，而非阻塞等待新数据
                    grads_item = grads_info['grads']
                    grads_count += 1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for target_grad, grad in zip(grads_buffer, grads_item):
                            target_grad += grad
                except queue.Empty:
                    pass
            
            time.sleep(0) # ​触发线程重新调度，让步其他线程
        
        except:
            train_log.log_exception(print_screen=True)
    
    train_log.log_info("exit OK", print_screen=True)

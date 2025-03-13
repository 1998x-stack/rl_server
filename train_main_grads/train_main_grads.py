
"""
0 启动logger,tensorboardX
1 grads 数据结构
2 启动Worker
3 启动Master
4 启动redis 或者 sampler
while True
    5 检测退出
        join各个进程
    6 从redis 或 sample queue 获取 样本
    7 发送采样数据到master
    8 整合梯度，发送新网络到 master 和 redis

8 清理退出

该文件为多个train_main_grads, 一个 grads_main,多个 sample_main

"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import argparse
import trainer_redis_grads
import torch.multiprocessing as mp
import time
import libs.config as config
import libs.utils as utils
import libs.log as log
import libs.redis_cache as redis_cache
import queue
    
if __name__ == '__main__':

    # 设置多进程模式
    mp.set_start_method('spawn')
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # doto 获取环境参数
    parser = argparse.ArgumentParser()
    
    #设置随机种子
    utils.setup_seed()
    
    #启动日志
    train_log = log.Log("train_main_grad")
    model_prefix= "train_main_grad"
    model_env_name = config.get_current_env_name() #"MicroRTSEnv"
    
    # 设置参数
    queue_config = config.get_current_queue_config()
    model_redis_config = config.get_current_redis_model_config()
    grads_redis_config = config.get_current_redis_grads_config()
    
    #梯度队列
    grads_queue = mp.Queue(maxsize=queue_config['len_grads_queue']) 
        
    #梯度整合网络
    train_net = config.create_net(model_env_name)    
    #train_net = train_net.to(train_net.get_device())
    # print(train_net)
    # import numpy as np
    # parameters = sum([np.prod(p.shape) for p in train_net.parameters()])
    # print(train_net,parameters)
    train_net.share_memory()

    #当前网络参数版本
    current_train_version = 0
    model_dict = mp.Manager().dict()
    model_dict['train_version'] = current_train_version
    model_dict['is_exit'] = False
    
    #启动redis
    model_redis_cache = redis_cache.RedisCache(train_log,model_redis_config)
    grads_redis_cache = redis_cache.RedisCache(train_log,grads_redis_config)
    
    #清理经验redis
    exps_redis_config = config.get_current_redis_exps_config()
    exps_redis_cache = redis_cache.RedisCache(train_log,exps_redis_config)
    exps_redis_cache.clear_data()
    del exps_redis_cache
        
    #获取训练模型数据
    while True:
        try:
            new_version = model_redis_cache.get_train_version()
            has_new_model = model_redis_cache.get_train_model(train_net)
            if (new_version is not None) and has_new_model:
                current_train_version = new_version
                model_dict['train_version'] = current_train_version
                break
        except:
            train_log.log_exception(print_screen=True)
            exit()
        
    #各种训练容器
    trainers = []
       
    #启动trainer
    for i in range(queue_config['num_trainer']):
        l_trainer = trainer_redis_grads.TrainerRedisGrads(id=i,model_dict=model_dict,share_model=train_net,
                                                          grads_queue=grads_queue,env_name=model_env_name,log=train_log,redis_log=train_log)
        trainers.append(l_trainer)
        l_trainer.run_trainer_redis()
        
    train_log.log_info("start run train_main_grads",print_screen=True)
                
    grads_buffer = None
    grads_count = 0
    
    #只具有参考意义
    grads_version = 0
    sample_version = 0

    while True:
        try:
            #退出检测
            exit_work = utils.exit_run()
            exit_flag = model_redis_cache.get_exit_flag()
            if exit_flag is not None:
                exit_work = exit_work or bool(exit_flag)
            
            if exit_work:
                
                train_log.log_info("start exit train_main_grad",print_screen=True)
                        
                model_dict['is_exit'] = True
                
                for l_trainer in trainers:
                    l_trainer.stop()
                                
                del model_redis_cache
                del grads_redis_cache
                                 
                train_log.log_info("end exit train_main_grad",print_screen=True)
                break
            
            #整合梯度
            if grads_count >= queue_config['num_update_grads']:
                
                grads_redis_cache.push_grads(grads_buffer,grads_version,sample_version)
                                             
                grads_buffer = None
                grads_count = 0
            else:
                try:
                    grads_info = grads_queue.get(block=False)
        
                    grads_version = grads_info['grads_version']
                    sample_version = grads_info['sample_version']
                    grads_item = grads_info['grads']
                    grads_count = grads_count+1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for target_grad, grad in zip(grads_buffer, grads_item):
                            target_grad += grad
                                                                                                                        
                except queue.Empty:
                    pass
                                
            #更新网络
            new_version = model_redis_cache.get_train_version()
            if (new_version is not None) and (new_version > current_train_version):
                has_new_model = model_redis_cache.get_train_model(train_net)
                if has_new_model: 
                    current_train_version = new_version
                    model_dict['train_version'] = current_train_version
  
            time.sleep(0)
        
        except:
            train_log.log_exception(print_screen=True)
    
    train_log.log_info("exit OK",print_screen=True)

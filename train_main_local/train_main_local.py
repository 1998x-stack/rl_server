# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

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
    8 整合梯度,发送新网络到 master 和 redis

9 清理退出
该文件为独立train_main_local,自带sampler
"""
import sys,os

import argparse
import trainer
import torch.multiprocessing as mp
import time
import numpy as np
import libs.config as config
import libs.utils as utils
import libs.log as log
import sampler
import check_main.checker as checker
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
    train_log = log.Log("train_main_local")

    model_env_name = config.get_current_env_name() #"MicroRTSEnv"
    train_log.log_info("current train algo_env is " + model_env_name,print_screen=True)

    model_prefix= "train_main_local"    
    model_version = None
    
    # 设置参数
    queue_config = config.get_current_queue_config()

    #梯度队列
    grads_queue = mp.Queue(maxsize=queue_config['len_grads_queue']) 
    #采样队列
    sample_queue = mp.Queue(maxsize=queue_config['len_sample_queue'])

    #梯度整合网络
    train_net = config.create_net(model_env_name)    
    # train_net = train_net.to(train_net.get_device()) 这里主要是采样
    # print(train_net)
    # import numpy as np
    # parameters = sum([np.prod(p.shape) for p in train_net.parameters()])
    # print(train_net,parameters)
    train_net.share_memory()
    
    #加载模型数据 
    current_train_version = utils.get_model_from_file(train_net,model_prefix + "_" + model_env_name,model_version)
    
    if current_train_version is None:
        train_log.log_info("has no model data and starts a new train",print_screen=True)
        current_train_version = 0

    #当前网络参数版本
    model_dict = mp.Manager().dict()
    #被训练网络版本
    model_dict['train_version'] = current_train_version 
    #退出标志
    model_dict['is_exit'] = False

    #各种训练容器
    trainers = []
    samplers = []

    #启动trainer
    for i in range(queue_config['num_trainer']):
        l_trainer = trainer.Trainer(id=i, model_dict=model_dict,share_model=train_net,
                                    sample_queue=sample_queue,grads_queue=grads_queue,
                                    env_name=model_env_name,log=train_log)
        trainers.append(l_trainer)
        l_trainer.run_trainer()

    #启动sampler
    for i in range(queue_config['num_sampler']): 
        l_sampler = sampler.Sampler(id=i, model_dict=model_dict,share_model=train_net,sample_queue=sample_queue,
                                    env_name=model_env_name,log=train_log)
        samplers.append(l_sampler)
        l_sampler.run_sampler()
    
    #启动checker
    if  queue_config['enable_checker']:
        train_checker = checker.Checker(model_dict=model_dict
                                        ,share_model=train_net, env_name=model_env_name,log=train_log)
        train_checker.run_checker("_algo_env_mixed")
                
    train_log.log_info("start run train_main_local",print_screen=True)
    
    version_update_sample_model = queue_config['version_update_sample_model']
    version_update_calculate_model = queue_config['version_update_calculate_model']
    
    grads_buffer = None
    grads_count = 0
    while True:
        try:
            #退出检测
            if utils.exit_run():
                obs_array = []
                act_array = []
                while not sample_queue.empty():
                    samples_info = sample_queue.get()
                    exp_lists = samples_info['exps']
                    for exp in exp_lists:
                        observation = np.expand_dims(exp[0], axis=0)
                        action = np.expand_dims(exp[1], axis=0)
                        obs_array.append(observation)
                        act_array.append(action)
                
                obs_array = np.concatenate(obs_array, axis=0)
                act_array = np.concatenate(act_array, axis=0)
                np.save('obs.npy', obs_array)
                np.save('act.npy', act_array)
                
                train_log.log_info("start exit train_main_local",print_screen=True)
                
                #保存当前模型版本
                utils.save_model_to_file(train_net,model_prefix + "_" + model_env_name,current_train_version)

                model_dict['is_exit'] = True
                
                #等待进程退出
                for l_sampler in samplers:
                    l_sampler.stop()
                    
                for l_trainer in trainers:
                    l_trainer.stop()
                
                if  queue_config['enable_checker']:
                    train_checker.stop()
                                              
                train_log.log_info("end exit train_main_local",print_screen=True)
                break
                                  
            #整合梯度
            if grads_count >= queue_config['num_update_grads']:
                #更新版本
                current_train_version = current_train_version + 1
                train_net.update_state(current_train_version,grads_buffer)
                model_dict['train_version'] = current_train_version
                                             
                grads_buffer = None
                grads_count = 0
            else:
                try:
                    grads_info = grads_queue.get(block=False)
        
                    #grads_version = grads_info['grads_version']
                    #sample_version = grads_info['sample_version']
                    grads_item = grads_info['grads']
                    grads_count = grads_count+1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for target_grad, grad in zip(grads_buffer, grads_item):
                            target_grad += grad
                                                                                                                        
                except queue.Empty:
                    pass
                    
            time.sleep(0)
        
        except:
            train_log.log_exception(print_screen=True)
    
    train_log.log_info("exit OK",print_screen=True)
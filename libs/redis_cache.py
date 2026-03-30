# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
"""
1 写入 version
2 写入 model
3 写入 exps
4 读取 version
5 读取 model
6 读取 exps
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import redis
import zlib
import pickle
import torch.nn as nn
from typing import Dict, List

from libs.log import Log
import time as _time
import functools


def _retry(max_retries=3, base_delay=1.0):
    """Retry decorator with exponential backoff for Redis operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except redis.ConnectionError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.log.log_info(f"Redis connection error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        _time.sleep(delay)
                        try:
                            self.conn.ping()
                        except Exception:
                            self._reconnect()
            self.log.log_exception()
            raise last_exception
        return wrapper
    return decorator


class RedisCache:

    version_name = 'version'
    model_name = 'model'
    exps_name = 'exps'
    exit_flag_name = 'exit'
    
    train_version_name = 'TRAIN_VERSION'
    train_model_name = 'train_model'
    
    sample_version_name = 'sample_version'
    sample_model_name = 'sample_model'
    
    grads_name = 'grads'

    def __init__(self,log: Log,redis_config: Dict):
        self.log = log
        self.redis_config = redis_config
        self.conn = redis.Redis(host=self.redis_config['ip'], 
                                port=self.redis_config['port'],
                                db=self.redis_config['db'],
                                password=self.redis_config['pw'])
        
        if not self.conn.ping():
            self.log.log_info("redis connect fail and will exit")
            exit()
        else:
            pass
 
    def __del__(self):
        self.conn.close()

    def _reconnect(self):
        """Attempt to reconnect to Redis."""
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = redis.Redis(
            host=self.redis_config['ip'],
            port=self.redis_config['port'],
            db=self.redis_config['db'],
            password=self.redis_config['pw']
        )

    def health_check(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return self.conn.ping()
        except Exception:
            return False

    # 是否使用 发布/订阅来解决模型同步问题？ todo by soongxl
        
    # 清空redis
    def clear_data(self):
        self.conn.flushall()
        
    # 设置退出标志
    def set_exit_flag(self, exit_flag: bool):
        try:
            self.conn.set(RedisCache.exit_flag_name,int(exit_flag))
            return True
        
        except Exception:
            self.log.log_exception()
            return False
    
    # 获取退出标志
    def get_exit_flag(self):
        try:
            exit_flag =  self.conn.get(RedisCache.exit_flag_name)
            if exit_flag is not None:
                return int(exit_flag)
            else:
                return None
            
        except Exception: 
            self.log.log_exception()
            return None
        
    # 设置训练模型数据
    @_retry()
    def set_train_version_model(self, version: int, model: nn.Module):
        try:
            model_dict = pickle.dumps(model.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
            model_dict = zlib.compress(model_dict)
            self.conn.set(RedisCache.train_model_name, model_dict)
            self.conn.set(RedisCache.train_version_name, int(version))
            return True
        
        except Exception:
            self.log.log_exception()
            return False
        
    # 获取训练模型版本
    def get_train_version(self):
        try:
            version =  self.conn.get(RedisCache.train_version_name)
            if version is not None:
                return int(version)
            else:
                return None
            
        except Exception: 
            self.log.log_exception()
            return None
            
    # 获取训练模型参数
    def get_train_model(self, model: nn.Module):
        try:
            result=self.conn.get(RedisCache.train_model_name)
            if result is not None:
                model_dict = pickle.loads(zlib.decompress(result))                       
                model.load_state_dict(model_dict)
                return True
            else:
                return False

        except Exception:
            self.log.log_exception()
            return False
                        
    # 压入采样经验
    @_retry()
    def push_exps(self, exps: List, sample_version: int):
        try:
            exps_info = dict()
            exps_info['exps'] = exps
            exps_info['sample_version'] = sample_version
            exps_info = pickle.dumps(exps_info, protocol=pickle.HIGHEST_PROTOCOL)
            exps_info = zlib.compress(exps_info)
            self.conn.lpush(RedisCache.exps_name, exps_info)  
            return True
        
        except Exception: 
            self.log.log_exception()
            return False
    
    # 获取采样经验
    def pop_exps(self):
        try:
            exps_info = self.conn.brpop(RedisCache.exps_name, timeout=5)
            if exps_info is None:
                return None, None
            exps_info = zlib.decompress(exps_info[1])
            exps_info = pickle.loads(exps_info)
            return exps_info['exps'], exps_info['sample_version']
        except Exception:
            self.log.log_exception()
            return None, None
            
    # 压入梯度信息
    @_retry()
    def push_grads(self, grads: List, grads_version: int, sample_version: int):
        try:
            grads_info = dict()
            grads_info['grads'] = grads
            grads_info['grads_version'] = grads_version
            grads_info['sample_version'] = sample_version
            grads_info = pickle.dumps(grads_info,protocol=pickle.HIGHEST_PROTOCOL)
            
            grads_info = zlib.compress(grads_info)
            self.conn.lpush(RedisCache.grads_name,grads_info)
            
            return True
        
        except Exception: 
            self.log.log_exception()
            return False
    
    # 获取梯度信息
    def pop_grads(self):
        try:
            grads_info = self.conn.brpop(RedisCache.grads_name, timeout=5)
            if grads_info is None:
                return None, None, None
            grads_info = zlib.decompress(grads_info[1])
            grads_info = pickle.loads(grads_info)
            return grads_info['grads'], grads_info['grads_version'], grads_info['sample_version']
        except Exception:
            self.log.log_exception()
            return None, None, None
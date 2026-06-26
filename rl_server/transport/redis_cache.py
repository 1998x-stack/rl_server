# -*- coding: utf-8 -*-
"""Redis 封装：分布式训练中的模型版本、经验队列、梯度队列与退出标志。
"""
import redis
import zlib
import pickle
import torch.nn as nn
from typing import Dict, List

from rl_server.utils.logging import Log
import time as _time
import functools


def _retry(max_retries=3, base_delay=1.0):
    """为 Redis 写操作生成带指数退避的重试装饰器。

    Args:
        max_retries: 最大尝试次数。
        base_delay: 首次重试前等待的秒数，后续按 ``2**attempt`` 倍增。

    Returns:
        装饰器函数，作用于 ``RedisCache`` 的实例方法。
    """
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
    """训练协调用的 Redis 键空间约定与序列化存取。

    使用 pickle + zlib 存储张量与 Python 对象；关键名以类属性形式集中定义。
    """

    version_name = 'version'
    model_name = 'model'
    exps_name = 'exps'
    exit_flag_name = 'exit'

    train_version_name = 'TRAIN_VERSION'
    train_model_name = 'train_model'

    sample_version_name = 'sample_version'
    sample_model_name = 'sample_model'

    grads_name = 'grads'

    def __init__(self, log: Log, redis_config: Dict):
        """连接 Redis 连接池并执行连通性检查。

        Args:
            log: 日志封装。
            redis_config: 需含 ``ip``、``port``、``db``、``pw``，可选 ``pool_size``。

        Raises:
            ConnectionError: ``ping`` 失败时。
        """
        self.log = log
        self.redis_config = redis_config
        pool_size = int(redis_config.get('pool_size', 10))
        self.pool = redis.ConnectionPool(
            host=redis_config['ip'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config['pw'],
            max_connections=pool_size
        )
        self.conn = redis.Redis(connection_pool=self.pool)

        if not self.conn.ping():
            self.log.log_info("Redis connect fail, will exit")
            raise ConnectionError("Cannot connect to Redis")

    def __del__(self):
        """析构时关闭连接（忽略异常）。"""
        self.conn.close()

    def _reconnect(self):
        """关闭旧连接并按配置重建单连接客户端（无连接池）。"""
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
        """对当前客户端执行 ``PING``。

        Returns:
            成功返回 ``True``，任意异常返回 ``False``。
        """
        try:
            return self.conn.ping()
        except Exception:
            return False

    def clear_data(self):
        """清空当前 Redis 数据库（``FLUSHALL``），慎用。"""
        self.conn.flushall()

    def set_exit_flag(self, exit_flag: bool):
        """写入全局退出标志，供远程 worker 轮询。

        Args:
            exit_flag: ``True`` 表示请求停止。

        Returns:
            成功返回 ``True``，异常时返回 ``False``。
        """
        try:
            self.conn.set(RedisCache.exit_flag_name, int(exit_flag))
            return True
        except Exception:
            self.log.log_exception()
            return False

    def get_exit_flag(self):
        """读取退出标志。

        Returns:
            已设置时为 ``0`` 或 ``1`` 的整数；未设置时为 ``None``；异常时为 ``None``。
        """
        try:
            exit_flag = self.conn.get(RedisCache.exit_flag_name)
            if exit_flag is not None:
                return int(exit_flag)
            else:
                return None
        except Exception:
            self.log.log_exception()
            return None

    @_retry()
    def set_train_version_model(self, version: int, model: nn.Module):
        """将模型 ``state_dict`` 压缩后写入 Redis，并更新训练版本键。

        Args:
            version: 整数版本号。
            model: 要广播的 PyTorch 模块。

        Returns:
            成功返回 ``True``，否则 ``False``。
        """
        try:
            model_dict = pickle.dumps(model.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
            model_dict = zlib.compress(model_dict)
            self.conn.set(RedisCache.train_model_name, model_dict)
            self.conn.set(RedisCache.train_version_name, int(version))
            return True
        except Exception:
            self.log.log_exception()
            return False

    def get_train_version(self):
        """读取当前 Redis 中的训练版本号。

        Returns:
            整数版本或 ``None``（未设置或出错）。
        """
        try:
            version = self.conn.get(RedisCache.train_version_name)
            if version is not None:
                return int(version)
            else:
                return None
        except Exception:
            self.log.log_exception()
            return None

    def get_train_model(self, model: nn.Module):
        """从 Redis 读取压缩的 ``state_dict`` 并加载到 ``model``。

        Args:
            model: 与保存时结构一致的模块。

        Returns:
            成功加载返回 ``True``，无数据或失败返回 ``False``。
        """
        try:
            result = self.conn.get(RedisCache.train_model_name)
            if result is not None:
                model_dict = pickle.loads(zlib.decompress(result))
                model.load_state_dict(model_dict)
                return True
            else:
                return False
        except Exception:
            self.log.log_exception()
            return False

    @_retry()
    def push_exps(self, exps: List, sample_version: int):
        """将一批经验与采样时版本号左压入列表（供 Trainer 侧 ``brpop``）。

        Args:
            exps: 经验数据列表。
            sample_version: 采样时对应的模型版本。

        Returns:
            成功返回 ``True``，否则 ``False``。
        """
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

    def pop_exps(self):
        """阻塞式从经验列表右端弹出一条（超时 5 秒）。

        Returns:
            ``(exps, sample_version)``；无数据或超时/错误时为 ``(None, None)``。
        """
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

    @_retry()
    def push_grads(self, grads: List, grads_version: int, sample_version: int):
        """将梯度列表与版本信息左压入梯度队列。

        Args:
            grads: 梯度数据结构（通常为张量或嵌套列表）。
            grads_version: 训练侧版本。
            sample_version: 产生该梯度的采样版本。

        Returns:
            成功返回 ``True``，否则 ``False``。
        """
        try:
            grads_info = dict()
            grads_info['grads'] = grads
            grads_info['grads_version'] = grads_version
            grads_info['sample_version'] = sample_version
            grads_info = pickle.dumps(grads_info, protocol=pickle.HIGHEST_PROTOCOL)
            grads_info = zlib.compress(grads_info)
            self.conn.lpush(RedisCache.grads_name, grads_info)
            return True
        except Exception:
            self.log.log_exception()
            return False

    def pop_grads(self):
        """阻塞式从梯度队列右端弹出一条（超时 5 秒）。

        Returns:
            ``(grads, grads_version, sample_version)``；无数据或失败时为三元 ``None``。
        """
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

# -*- coding: utf-8 -*-
"""进程与运行环境：信号处理、随机种子、多进程启动方式及心跳文件。"""
import os
import time
import signal
import random
import tempfile
import threading
import numpy as np
import torch
import torch.multiprocessing as mp

_shutdown_event = threading.Event()


def setup_signal_handlers():
    """注册 SIGTERM/SIGINT 处理器，触发后 ``should_exit()`` 为真。"""
    def _handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def should_exit() -> bool:
    """是否已收到优雅退出请求。"""
    return _shutdown_event.is_set()


def request_shutdown():
    """主动设置退出标志（用于测试或子系统协调）。"""
    _shutdown_event.set()


def setup_seed(seed: int = 1970010101):
    """固定 Python、NumPy、PyTorch 及 CUDA 的随机种子，并启用 cudnn 确定性模式。

    Args:
        seed: 全局种子值。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_mp():
    """将多进程启动方式设为 ``spawn``，并限制 BLAS 线程数以避免过度订阅。"""
    mp.set_start_method('spawn', force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


HEARTBEAT_DIR = os.path.join(tempfile.gettempdir(), 'rl_server')


def write_heartbeat(worker_type: str, worker_id: int):
    """写入心跳文件，供外部监控进程存活（路径含进程 PID）。

    Args:
        worker_type: 工作者类型标识字符串。
        worker_id: 工作者实例编号。
    """
    os.makedirs(HEARTBEAT_DIR, exist_ok=True)
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    with open(path, 'w') as f:
        f.write(str(time.time()))


def cleanup_heartbeat(worker_type: str, worker_id: int):
    """进程退出时删除对应心跳文件。

    Args:
        worker_type: 与 ``write_heartbeat`` 一致。
        worker_id: 与 ``write_heartbeat`` 一致。
    """
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    if os.path.exists(path):
        os.remove(path)

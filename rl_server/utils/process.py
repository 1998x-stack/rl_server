# -*- coding: utf-8 -*-
"""Process management, signal handling, and seed setup."""
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
    def _handler(signum, frame):
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def should_exit() -> bool:
    return _shutdown_event.is_set()


def request_shutdown():
    _shutdown_event.set()


def setup_seed(seed: int = 1970010101):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_mp():
    mp.set_start_method('spawn', force=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


HEARTBEAT_DIR = os.path.join(tempfile.gettempdir(), 'rl_server')


def write_heartbeat(worker_type: str, worker_id: int):
    """Write heartbeat file for monitoring."""
    os.makedirs(HEARTBEAT_DIR, exist_ok=True)
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    with open(path, 'w') as f:
        f.write(str(time.time()))


def cleanup_heartbeat(worker_type: str, worker_id: int):
    """Remove heartbeat file on shutdown."""
    path = os.path.join(HEARTBEAT_DIR, f"{worker_type}_{worker_id}_{os.getpid()}")
    if os.path.exists(path):
        os.remove(path)

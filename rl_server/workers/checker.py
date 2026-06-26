# -*- coding: utf-8 -*-
"""Checker 工作者：周期性加载共享权重并在环境中评估，写入 TensorBoard。

逻辑源自 ``check_main/checker.py``。
"""
import time
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_net, create_agent


class CheckerWorker:
    """在独立进程中运行评估智能体，将标量指标记录到 SummaryWriter。"""

    def __init__(self, model_dict, share_model: nn.Module, env_name, log: Log):
        """构造检查进程封装。

        Args:
            model_dict: 共享字典，需含 ``is_exit``、``TRAIN_VERSION``。
            share_model: 主训练用的网络（用于同步权重复制到检查用网络）。
            env_name: 环境/算法名。
            log: 日志对象。
        """
        self.model_dict = model_dict
        self.share_model = share_model
        self.self_version = model_dict['TRAIN_VERSION']
        self.env_name = env_name
        self.log = log
        self.process = None
        self.comment = "checker"

    def process_function(self):
        """子进程：复制权重、评估、写 TB，直到退出。"""
        setup_seed()
        check_net = create_net(self.env_name)
        check_agent = create_agent(self.env_name, check_net, is_checker=True)
        writer = SummaryWriter(comment=self.comment)
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                check_net.load_state_dict(self.share_model.state_dict())
                check_net.train(False)
                self.self_version = self.model_dict['TRAIN_VERSION']
                infos = check_agent.check_single_env()
                if isinstance(infos, dict):
                    for (key, value) in infos.items():
                        writer.add_scalar(key, value, self.self_version)
                    writer.flush()
                time.sleep(0)
            except Exception:
                self.log.log_exception()
                continue
        writer.close()

        self.log.log_info(f'exit checker processid {self.process.pid}')

    def run(self, comment: str = None):
        """启动检查子进程。

        Args:
            comment: 可选，覆盖 TensorBoard 的 ``comment`` 后缀。
        """
        if comment is not None:
            self.comment = comment
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start checker processid {self.process.pid}')

    def stop(self):
        """终止检查子进程。"""
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()

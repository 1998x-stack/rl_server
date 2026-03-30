# -*- coding: utf-8 -*-
"""Trainer 工作者：从队列读取样本批次，计算梯度并写回梯度队列。

逻辑源自 ``train_main_local/trainer.py``，已改用 ``rl_server`` 包内导入。
"""
import time
import queue
from typing import Dict

import torch.nn as nn
import torch.multiprocessing as mp

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_calculate


class TrainerWorker:
    """在独立进程中运行：消费 ``sample_queue``，通过 ``create_calculate`` 生成梯度并放入 ``grads_queue``。"""

    def __init__(
        self,
        idx: int,
        model_dict: Dict,
        share_model: nn.Module,
        sample_queue: mp.Queue,
        grads_queue: mp.Queue,
        env_name: str,
        log: Log,
    ):
        """构造 Trainer 工作者。

        Args:
            idx: 本 Trainer 实例编号。
            model_dict: 多进程共享字典，需含 ``is_exit``、``TRAIN_VERSION`` 等。
            share_model: 与主进程共享内存的策略网络。
            sample_queue: 样本来源队列。
            grads_queue: 梯度输出队列。
            env_name: 算法环境名，用于 ``create_calculate``。
            log: 日志对象。
        """
        self.log = log
        self.process = None
        self.trainer_id = idx
        self.env_name = env_name

        self.model_dict = model_dict
        self.share_model = share_model

        self.sample_queue = sample_queue
        self.grads_queue = grads_queue

    def process_function(self):
        """子进程入口：循环直到 ``model_dict['is_exit']``，拉取样本并推送梯度。"""
        setup_seed()
        calculate = create_calculate(self.env_name, self.share_model)
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                samples_info = self.sample_queue.get()
                grads_list, train_version = calculate.generate_grads(samples_info['exps'], self.model_dict)

                for grads in grads_list:
                    grads_info = dict()
                    grads_info['grads'] = grads
                    grads_info['grads_version'] = train_version
                    grads_info['sample_version'] = samples_info['sample_version']
                    self.grads_queue.put(grads_info)

                time.sleep(0)

            except queue.Full:
                time.sleep(1)
                continue
            except queue.Empty:
                continue
            except Exception:
                self.log.log_exception()
                continue
        self.log.log_info(f'exit trainer processid {self.process.pid} trainerid {self.trainer_id}')

    def run(self):
        """启动子进程并记录 PID。"""
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start trainer processid {self.process.pid} trainerid {self.trainer_id}')

    def stop(self):
        """终止子进程并 ``join``。"""
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()

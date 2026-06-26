# -*- coding: utf-8 -*-
"""Sampler 工作者：在子进程中与环境交互并将经验写入样本队列。
"""
import time
import queue
import torch.nn as nn
import torch.multiprocessing as mp

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_agent


class SamplerWorker:
    """独立进程采集经验：调用 ``sample_multi_envs`` 并将结果放入 ``sample_queue``。"""

    def __init__(self, idx, model_dict, share_model: nn.Module, sample_queue, env_name, log: Log):
        """构造采样工作者。

        Args:
            idx: 采样器编号。
            model_dict: 共享状态，需含 ``is_exit``、``TRAIN_VERSION``。
            share_model: 与主进程共享的策略网络。
            sample_queue: 输出队列，元素为含 ``exps`` 与 ``sample_version`` 的字典。
            env_name: 算法环境名。
            log: 日志对象。
        """
        self.sampler_id = idx
        self.model_dict = model_dict
        self.share_model = share_model
        self.env_name = env_name
        self.log = log
        self.sample_queue = sample_queue
        self.process = None

    def process_function(self):
        """子进程主循环：采样直至收到退出标志。"""
        setup_seed()
        sample_agent = create_agent(self.env_name, self.share_model)

        while True:
            if self.model_dict['is_exit']:
                break

            try:
                sample_agent.sample_net.load_state_dict(self.share_model.state_dict())
                exps_list = sample_agent.sample_multi_envs(self.model_dict)

                if exps_list is not None:
                    for exps in exps_list:
                        exps_info = dict()
                        exps_info['sample_version'] = self.model_dict['TRAIN_VERSION']
                        exps_info['exps'] = exps
                        self.sample_queue.put(exps_info)
                else:
                    self.log.log_info("sample_agent.sample_multi_envs return None")

                time.sleep(0)

            except queue.Full:
                time.sleep(5)
                continue
            except Exception:
                self.log.log_exception()
                continue

        try:
            del sample_agent
        except Exception:
            self.log.log_exception()

        self.log.log_info(f'exit sampler processid {self.process.pid} samplerid {self.sampler_id}')

    def run(self):
        """启动采样子进程。"""
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        self.log.log_info(f'start sampler processid {self.process.pid} samplerid {self.sampler_id}')

    def stop(self):
        """终止采样子进程。"""
        try:
            if self.process is not None:
                self.process.terminate()
                self.process.join()
        except Exception:
            self.log.log_exception()

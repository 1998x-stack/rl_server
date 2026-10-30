# -*- coding: utf-8 -*-
"""Trainer 工作者：从队列读取样本批次，计算梯度并写回梯度队列。
"""
import os
import time
import queue
import numpy as np
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

    @staticmethod
    def _grad_norm(grads_list) -> float:
        """计算一批梯度的 L2 范数（同时暴露 NaN/Inf：返回 ``inf``/``nan``）。"""
        total = 0.0
        for grads in grads_list:
            for grad in grads:
                if grad is None:
                    continue
                array = np.asarray(grad)
                total += float(np.square(array, dtype=np.float64).sum())
        return total ** 0.5

    def _collect_metrics(self, grads_list, train_version, sample_version, calc_state=None) -> dict:
        """收集本批可观测指标：梯度范数 + 算法自己写入的 loss 等。

        指标随梯度批次一起传递（``grads_info['metrics']``），不再依赖主进程按版本号去
        ``model_dict['trainer_metrics']`` 里对号入座——那样每批只有第一个能匹配上。

        Args:
            grads_list: 本批梯度。
            train_version: 计算该批梯度时使用的训练版本。
            sample_version: 该批样本对应的版本。
            calc_state: 传给 ``generate_grads`` 的字典；算法可能在 ``trainer_metrics`` 里写指标。

        Returns:
            至少含 ``grad_norm`` / ``grads_version`` / ``sample_version`` 的字典。
        """
        grad_norm = self._grad_norm(grads_list)
        metrics = {}
        for source in (calc_state or {}, self.model_dict):
            try:
                existing = source.get('trainer_metrics')
            except Exception:
                self.log.exception(f'trainer metrics read failed trainer={self.trainer_id}')
                continue
            if isinstance(existing, dict):
                metrics.update({k: v for k, v in existing.items()
                                if isinstance(v, (int, float)) and k != 'train_version'})
        metrics['grad_norm'] = grad_norm
        metrics['grads_version'] = train_version
        metrics['sample_version'] = sample_version
        return metrics

    def _put_with_exit(self, grads_info) -> bool:
        """推入梯度；队列满时仍能观察 ``is_exit``（否则只能被强杀）。"""
        while True:
            if self.model_dict['is_exit']:
                return False
            try:
                self.grads_queue.put(grads_info, timeout=1.0)
                return True
            except queue.Full:
                self._queue_full_waits += 1
                self.log.info_throttled(
                    f'trainer_queue_full_{self.trainer_id}',
                    f'grads queue full trainer={self.trainer_id} waits={self._queue_full_waits}',
                    interval=30,
                )

    def process_function(self):
        """子进程入口：循环直到 ``model_dict['is_exit']``，拉取样本并推送梯度。"""
        setup_seed()
        self._queue_full_waits = 0
        heartbeat_key = f'trainer_heartbeat_{self.trainer_id}'
        self.model_dict[heartbeat_key] = time.time()
        calculate = create_calculate(self.env_name, self.share_model)
        started = time.time()
        batches = 0
        self.log.log_info(f'trainer started pid={os.getpid()} trainer={self.trainer_id}')
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                # 带超时的 get：否则 trainer 会卡在空队列上，只能被 SIGTERM 强杀，
                # 退出前的清理与统计日志就全丢了。
                samples_info = self.sample_queue.get(timeout=0.5)
                train_version = self.model_dict['TRAIN_VERSION']
                # 算法把 loss 等指标写进传进去的 dict（PPO 就是这么做），所以这个 dict
                # 必须显式带出来，否则指标随临时对象一起被丢掉。
                calc_state = {'TRAIN_VERSION': train_version}
                grads_list, _ = calculate.generate_grads(samples_info['exps'], calc_state)
                metrics = self._collect_metrics(grads_list, train_version,
                                                samples_info['sample_version'], calc_state)
                grad_norm = metrics['grad_norm']

                for grads in grads_list:
                    grads_info = dict()
                    grads_info['grads'] = grads
                    grads_info['grads_version'] = train_version
                    grads_info['sample_version'] = samples_info['sample_version']
                    grads_info['metrics'] = metrics
                    self._put_with_exit(grads_info)
                self.model_dict[heartbeat_key] = time.time()
                batches += 1

                elapsed = max(time.time() - started, 1e-6)
                self.log.info_throttled(
                    f'trainer_progress_{self.trainer_id}',
                    f'trainer progress trainer={self.trainer_id} batches={batches} '
                    f'batches_per_sec={batches / elapsed:.2f} grads_version={train_version} '
                    f'sample_version={samples_info["sample_version"]} '
                    f'grad_norm={grad_norm:.4g} queue_full_waits={self._queue_full_waits}',
                    interval=30,
                )

                time.sleep(0)

            except queue.Empty:
                continue
            except Exception:
                # 写清阶段，否则合并后的 traceback 无法判断是取样本还是算梯度出错。
                self.log.exception(f'trainer loop trainer={self.trainer_id}')
                continue
                self.log.log_info(
            f'trainer exit pid={os.getpid()} trainer={self.trainer_id} batches={batches} '
            f'queue_full_waits={self._queue_full_waits}'
        )
        try:
            self.grads_queue.cancel_join_thread()
            self.sample_queue.cancel_join_thread()
        except Exception:
            self.log.exception(f'trainer queue cleanup trainer={self.trainer_id}')

    def run(self):
        """启动子进程并记录 PID。"""
        self.process = mp.Process(target=self.process_function, name=f'trainer-{self.trainer_id}')
        self.process.start()
        self.log.log_info(f'start trainer processid {self.process.pid} trainerid {self.trainer_id}')

    def stop(self, timeout: float = 15.0, grace: float = 5.0):
        """终止子进程并 ``join``；超时则 ``kill``，避免关闭阶段被卡死。"""
        if self.process is None:
            return
        try:
            # is_exit 已经在主进程设置好：先给子进程自愿退出的时间，
            # 这样它能写完退出日志、关闭 env（MicroRTS 的 JVM 尤其重要）。
            graceful = False
            if grace > 0:
                self.process.join(grace)
                graceful = not self.process.is_alive()
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout)
            if self.process.is_alive():
                self.log.warning(
                    f'trainer did not exit gracefully pid={self.process.pid} timeout={timeout}s, killing'
                )
                self.process.kill()
                self.process.join(5)
            self.log.log_info(
                f'stopped trainer processid {self.process.pid} trainerid {self.trainer_id} '
                f'graceful={graceful} exitcode={self.process.exitcode}'
            )
        except Exception:
            self.log.exception(f'stopping trainer={self.trainer_id} failed')

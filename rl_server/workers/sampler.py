# -*- coding: utf-8 -*-
"""Sampler 工作者：在子进程中与环境交互并将经验写入样本队列。
"""
import os
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

    def _put_with_exit(self, exps) -> bool:
        """把经验放入队列，队列满时仍能响应 ``is_exit``。

        ``mp.Queue.put`` 无超时时会永久阻塞，pool 满时 worker 无法观察退出标志，
        只能被 SIGTERM 强杀（环境/日志都来不及清理）。这里改为带超时的重试。

        Args:
            exps: 单批经验。

        Returns:
            ``True`` 表示已入队；``False`` 表示收到退出请求后放弃。
        """
        exps_info = dict()
        exps_info['sample_version'] = self.model_dict['TRAIN_VERSION']
        exps_info['exps'] = exps
        while True:
            if self.model_dict['is_exit']:
                return False
            try:
                self.sample_queue.put(exps_info, timeout=1.0)
                return True
            except queue.Full:
                self._queue_full_waits += 1
                self.log.info_throttled(
                    f'sampler_queue_full_{self.sampler_id}',
                    f'sample queue full sampler={self.sampler_id} '
                    f'waits={self._queue_full_waits}',
                    interval=30,
                )

    def process_function(self):
        """子进程主循环：采样直至收到退出标志。"""
        setup_seed()
        self._last_sync_version = -1
        self._queue_full_waits = 0
        sample_agent = create_agent(self.env_name, self.share_model)
        started = time.time()
        batches = 0
        versions = 0
        self.log.log_info(f'sampler started pid={os.getpid()} sampler={self.sampler_id}')

        while True:
            if self.model_dict['is_exit']:
                break

            try:
                current_version = self.model_dict.get('TRAIN_VERSION', 0)
                if current_version != self._last_sync_version:
                    sample_agent.sample_net.load_state_dict(self.share_model.state_dict())
                    self._last_sync_version = current_version
                    versions += 1
                exps_list = sample_agent.sample_multi_envs(self.model_dict)

                if exps_list is not None:
                    for exps in exps_list:
                        if not self._put_with_exit(exps):
                            break
                    batches += 1
                else:
                    self.log.info_throttled(
                        f'sampler_none_{self.sampler_id}',
                        f'sample_multi_envs returned None sampler={self.sampler_id}',
                        interval=30,
                    )

                elapsed = max(time.time() - started, 1e-6)
                self.log.info_throttled(
                    f'sampler_progress_{self.sampler_id}',
                    f'sampler progress sampler={self.sampler_id} batches={batches} '
                    f'batches_per_sec={batches / elapsed:.2f} weight_syncs={versions} '
                    f'queue_full_waits={self._queue_full_waits} version={current_version}',
                    interval=30,
                )

                time.sleep(0)

            except Exception:
                self.log.exception(f'sampler loop sampler={self.sampler_id}')
                continue

        try:
            del sample_agent
        except Exception:
            self.log.exception(f'sampler cleanup sampler={self.sampler_id}')

        # 关闭时队列往往仍是满的、消费端已退出：不取消 feeder 的 join，子进程会卡在
        # 解释器退出阶段刷不出数据，父进程只能等满 grace 再 terminate（实测每次 5s）。
        try:
            self.sample_queue.cancel_join_thread()
        except Exception:
            self.log.exception(f'sampler queue cleanup sampler={self.sampler_id}')

        self.log.log_info(
            f'sampler exit pid={os.getpid()} sampler={self.sampler_id} batches={batches} '
            f'queue_full_waits={self._queue_full_waits}'
        )

    def run(self):
        """启动采样子进程。"""
        self.process = mp.Process(target=self.process_function, name=f'sampler-{self.sampler_id}')
        self.process.start()
        self.log.log_info(f'start sampler processid {self.process.pid} samplerid {self.sampler_id}')

    def stop(self, timeout: float = 15.0, grace: float = 5.0):
        """终止采样子进程；超时则 ``kill``，避免关闭阶段被卡死。"""
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
                    f'sampler did not exit gracefully pid={self.process.pid} timeout={timeout}s, killing'
                )
                self.process.kill()
                self.process.join(5)
            self.log.log_info(
                f'stopped sampler processid {self.process.pid} samplerid {self.sampler_id} '
                f'graceful={graceful} exitcode={self.process.exitcode}'
            )
        except Exception:
            self.log.exception(f'stopping sampler={self.sampler_id} failed')

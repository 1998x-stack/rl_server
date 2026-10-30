# -*- coding: utf-8 -*-
"""Checker 工作者：周期性加载共享权重并在环境中评估，写入 TensorBoard。
"""
import os
import time
import torch.nn as nn
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed
from rl_server.algorithms import create_net, create_agent


def _avg_seconds(started: float, count: int) -> float:
    """平均每次评估耗时；尚未完成第一次评估时返回已耗时。"""
    elapsed = time.time() - started
    return elapsed / count if count else elapsed


def _format_metrics(metrics) -> str:
    """把指标字典格式化为 ``k=v`` 单行文本（浮点保留 4 位有效数字）。"""
    return ' '.join(
        f'{key}={value:.4g}' if isinstance(value, (int, float)) else f'{key}={value}'
        for key, value in metrics.items()
    )


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
        # 让长评估在收到退出请求时提前返回 None，而不是被 SIGTERM 强杀。
        check_agent.should_stop = lambda: bool(self.model_dict['is_exit'])
        writer = SummaryWriter(comment=self.comment)
        evaluations = 0
        started = time.time()
        last_metrics = {}
        last_log = time.time()
        log_every = max(int(self.model_dict.get('check_log_interval', 100) or 100), 1)
        self.log.log_info(f'checker started pid={os.getpid()} log_every={log_every}')
        while True:
            if self.model_dict['is_exit']:
                break
            try:
                check_net.load_state_dict(self.share_model.state_dict())
                check_net.train(False)
                self.self_version = self.model_dict['TRAIN_VERSION']
                # 评估一次可能几十秒（MuJoCo/MicroRTS），没这行日志就只能干等。
                self.log.info_throttled(
                    'check_evaluating',
                    f'check evaluating version={self.self_version} completed={evaluations} '
                    f'elapsed={time.time() - started:.0f}s '
                    f'avg_sec={_avg_seconds(started, evaluations):.1f}',
                    interval=30,
                )
                infos = check_agent.check_single_env()
                if infos is None:
                    self.log.info_throttled(
                        'check_interrupted',
                        f'check interrupted version={self.self_version} '
                        f'completed={evaluations} (partial episode discarded)',
                        interval=10,
                    )
                    continue
                if isinstance(infos, dict):
                    for (key, value) in infos.items():
                        writer.add_scalar(key, value, self.self_version)
                    writer.flush()
                    evaluations += 1
                    last_metrics = dict(infos)
                    # 以前每个指标一行且只在 TB 里；现在"首评 + 每 N 次 + 每 60s"各出一条，
                    # 短跑能看到起始点，长跑不会被评估频率刷爆。
                    now = time.time()
                    if evaluations == 1 or evaluations % log_every == 0 or now - last_log >= 60:
                        self.log.info(
                            f'check progress version={self.self_version} evaluations={evaluations} '
                            + _format_metrics(last_metrics)
                        )
                        last_log = now
                time.sleep(0)
            except Exception:
                self.log.exception('checker loop')
                continue
        writer.close()

        self.log.log_info(
            f'checker exit pid={os.getpid()} evaluations={evaluations} '
            f'last_version={self.self_version} {_format_metrics(last_metrics)}'.strip()
        )

    def run(self, comment: str = None):
        """启动检查子进程。

        Args:
            comment: 可选，覆盖 TensorBoard 的 ``comment`` 后缀。
        """
        if comment is not None:
            self.comment = comment
        self.process = mp.Process(target=self.process_function, name='checker')
        self.process.start()
        self.log.log_info(f'start checker processid {self.process.pid}')

    def stop(self, timeout: float = 15.0, grace: float = 1.0):
        """终止检查子进程；超时则 ``kill``。"""
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
                    f'checker did not exit gracefully pid={self.process.pid} timeout={timeout}s, killing'
                )
                self.process.kill()
                self.process.join(5)
            self.log.log_info(
                f'stopped checker processid {self.process.pid} '
                f'graceful={graceful} exitcode={self.process.exitcode}'
            )
        except Exception:
            self.log.exception('stopping checker failed')

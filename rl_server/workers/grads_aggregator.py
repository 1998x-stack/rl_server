# -*- coding: utf-8 -*-
"""梯度聚合服务：从 Redis 拉取梯度、累加并更新模型，再广播到模型 Redis。
"""
import collections

import numpy as np
import time

from rl_server.core.optim import trainable_parameters
from rl_server.core.gradient_contract import (
    gradient_batch_rejection,
    resolve_max_gradient_lag,
)
from rl_server.utils.logging import Log
from rl_server.utils.process import should_exit
from rl_server.utils.checkpoint import save_model, load_model
from rl_server.transport.redis_cache import RedisCache
from rl_server.algorithms import create_net


class GradsAggregator:
    """维护一份聚合网络，从 ``grads`` Redis 消费梯度，达到批次数后执行 ``update_state`` 并同步到 ``model`` Redis。"""

    def __init__(
        self,
        env_name: str,
        model_prefix: str,
        grads_redis_config: dict,
        model_redis_config: dict,
        batch_update_count: int = 10,
        max_gradient_lag=0,
        log: Log = None,
    ):
        """初始化聚合器与双 Redis 连接。

        Args:
            env_name: 用于 ``create_net`` 的环境名。
            model_prefix: 保存检查点时的文件名前缀组成部分。
            grads_redis_config: 梯度队列所在 Redis 配置。
            model_redis_config: 模型广播所在 Redis 配置。
            batch_update_count: 累计多少条梯度后执行一次参数更新。
            log: 可选日志；默认新建 ``Log("grads_aggregator")``。
        """
        self.env_name = env_name
        self.model_prefix = model_prefix
        self.batch_update_count = batch_update_count
        if batch_update_count < 1:
            raise ValueError("Invalid gradient aggregation limits")
        self.configured_max_gradient_lag = max_gradient_lag
        # auto：按一个聚合窗口（batch_update_count 条梯度）推导可接受的滞后跨度。
        self.max_gradient_lag = resolve_max_gradient_lag(
            max_gradient_lag, window=batch_update_count
        )
        if self.max_gradient_lag < 0:
            raise ValueError("Invalid gradient aggregation limits")
        self.rejected_gradients = 0
        self.rejected_reasons: collections.Counter = collections.Counter()
        self.log = log or Log("grads_aggregator")

        self.grads_net = create_net(env_name)

        self.current_train_version = load_model(
            self.grads_net, f"{model_prefix}_{env_name}"
        )
        if self.current_train_version is None:
            self.log.log_info("No existing model data, starting fresh training")
            self.current_train_version = 0

        self.grads_redis = RedisCache(self.log, grads_redis_config)
        # Never flush shared Redis storage when a worker starts.

        self.model_redis = RedisCache(self.log, model_redis_config)
        # Run isolation is controlled by the configured Redis namespace.

        self.model_redis.set_exit_flag(False)
        if not self.model_redis.set_train_version_model(self.current_train_version, self.grads_net):
            raise RuntimeError("Unable to publish initial model snapshot")

    def _accept_gradients(self, grads, grads_version) -> bool:
        """Reject stale, malformed and non-finite gradients before applying."""
        return self._rejection_reason(grads, grads_version) is None

    def _rejection_reason(self, grads, grads_version) -> str | None:
        """返回拒绝原因（``None`` 表示可接受），便于分项统计与定位。"""
        return gradient_batch_rejection(
            grads, grads_version, trainable_parameters(self.grads_net),
            self.current_train_version, self.max_gradient_lag,
        )

    def _report_progress(self, stats, reasons, started):
        """周期输出聚合进度与拒绝原因分项（限流，避免刷屏）。"""
        elapsed = max(time.time() - started, 1e-6)
        rejected = stats['rejected']
        total = rejected + stats['accepted']
        rate = (100.0 * rejected / total) if total else 0.0
        detail = ",".join(f"{name}={count}" for name, count in reasons.most_common(3))
        message = (
            f"grads progress version={self.current_train_version} updates={stats['updates']} "
            f"accepted={stats['accepted']} rejected={rejected} reject_pct={rate:.1f}% "
            f"reasons=[{detail}] rate={stats['updates'] / elapsed:.2f}upd/s"
        )
        self.log.info_throttled("grads_progress", message, interval=30)
        if rate >= 50.0 and total >= 20:
            self.log.warning_throttled(
                "grads_reject_high",
                f"over half of gradient batches rejected ({rejected}/{total}); "
                f"check max_gradient_lag={self.max_gradient_lag} against worker concurrency",
                interval=60,
            )

    def _shutdown(self, saved_version, stats):
        """一次性关闭：每一步单独兜底，任何失败都不再重入关闭流程。"""
        steps = (
            ("save_model", lambda: save_model(
                self.grads_net, f"{self.model_prefix}_{self.env_name}", str(saved_version))),
            ("set_exit_flag", lambda: self.model_redis.set_exit_flag(1)),
            ("close_redis", self._close_redis),
        )
        for name, action in steps:
            try:
                action()
            except Exception:
                self.log.exception(f"shutdown step={name} failed")
        self.log.summary({
            'version': self.current_train_version,
            'updates': stats['updates'],
            'accepted': stats['accepted'],
            'rejected': stats['rejected'],
        })

    def _close_redis(self):
        """释放两个 Redis 连接（析构登录 INFO，避免关闭阶段刷屏）。"""
        del self.model_redis
        del self.grads_redis

    def run(self):
        """主循环：聚合梯度、更新模型、响应 ``should_exit()`` 时保存并设置退出标志。"""
        self.log.log_info(
            f"Starting gradient aggregation server env={self.env_name} "
            f"batch_update_count={self.batch_update_count} "
            f"max_gradient_lag={getattr(self, 'configured_max_gradient_lag', self.max_gradient_lag)}"
            f"(effective={self.max_gradient_lag}) "
            f"start_version={self.current_train_version}"
        )

        grads_buffer = None
        grads_count = 0
        pending_version = None
        stats = {'accepted': 0, 'updates': 0, 'rejected': 0}
        reasons = collections.Counter()
        started = time.time()
        shutdown_started = False

        while True:
            try:
                if should_exit() and not shutdown_started:
                    shutdown_started = True
                    self.log.log_info(
                        f"shutdown requested version={self.current_train_version} "
                        f"pending={pending_version} accepted={stats['accepted']} rejected={stats['rejected']}"
                    )
                    self._shutdown(pending_version or self.current_train_version, stats)
                    break

                if pending_version is not None:
                    try:
                        published = self.model_redis.set_train_version_model(
                            pending_version, self.grads_net
                        )
                    except Exception:
                        self.log.exception(f"publish snapshot failed version={pending_version}")
                        published = False
                    if published:
                        self.current_train_version = pending_version
                        pending_version = None
                        grads_buffer = None
                        grads_count = 0
                        stats['updates'] += 1
                        self._report_progress(stats, reasons, started)
                    else:
                        self.log.info_throttled(
                            "publish_retry",
                            f"retrying model snapshot publication version={pending_version}",
                            interval=5,
                        )
                        time.sleep(1)
                elif grads_count >= self.batch_update_count:
                    pending_version = self.current_train_version + 1
                    self.grads_net.update_state(pending_version, grads_buffer)
                else:
                    grads_item, grads_version, sample_version = self.grads_redis.pop_grads()

                    if grads_item is not None:
                        reason = self._rejection_reason(grads_item, grads_version)
                        if reason is not None:
                            stats['rejected'] += 1
                            self.rejected_gradients = stats['rejected']
                            self.rejected_reasons[reason] += 1
                            reasons[reason] += 1
                            self._report_progress(stats, reasons, started)
                            continue
                        stats['accepted'] += 1
                        grads_count += 1
                        if grads_buffer is None:
                            grads_buffer = grads_item
                        else:
                            for target_grad, grad in zip(grads_buffer, grads_item):
                                target_grad += grad

            except Exception:
                self.log.exception("gradient aggregation loop")

        self.log.log_info("Gradient aggregation server exited")

# -*- coding: utf-8 -*-
"""梯度聚合服务：从 Redis 拉取梯度、累加并更新模型，再广播到模型 Redis。
"""
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
        self.log = log or Log("grads_aggregator")

        self.grads_net = create_net(env_name)

        self.current_train_version = load_model(
            self.grads_net, f"{model_prefix}_{env_name}"
        )
        if self.current_train_version is None:
            self.log.log_info("No existing model data, starting fresh training")
            self.current_train_version = 0

        self.grads_redis = RedisCache(self.log, grads_redis_config)
        self.grads_redis.clear_data()

        self.model_redis = RedisCache(self.log, model_redis_config)
        self.model_redis.clear_data()

        self.model_redis.set_train_version_model(self.current_train_version, self.grads_net)

    def run(self):
        """主循环：聚合梯度、更新模型、响应 ``should_exit()`` 时保存并设置退出标志。"""
        self.log.log_info("Starting gradient aggregation server")

        grads_buffer = None
        grads_count = 0

        while True:
            try:
                if should_exit():
                    self.log.log_info("Shutting down gradient aggregation server")

                    save_model(self.grads_net, f"{self.model_prefix}_{self.env_name}",
                               str(self.current_train_version))

                    self.model_redis.set_exit_flag(1)

                    del self.model_redis
                    del self.grads_redis

                    self.log.log_info("Gradient aggregation server shutdown complete")
                    break

                if grads_count >= self.batch_update_count:
                    self.current_train_version += 1
                    self.grads_net.update_state(self.current_train_version, grads_buffer)
                    self.model_redis.set_train_version_model(self.current_train_version, self.grads_net)

                    grads_buffer = None
                    grads_count = 0
                else:
                    grads_item, grads_version, sample_version = self.grads_redis.pop_grads()

                    if grads_item is not None:
                        grads_count += 1
                        if grads_buffer is None:
                            grads_buffer = grads_item
                        else:
                            for target_grad, grad in zip(grads_buffer, grads_item):
                                target_grad += grad

            except Exception:
                self.log.log_exception()

        self.log.log_info("Gradient aggregation server exited")

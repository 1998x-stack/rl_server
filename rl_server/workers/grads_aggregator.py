# -*- coding: utf-8 -*-
"""
Gradient aggregation server.
Based on grads_main/grads_main.py with updated imports.
Aggregates gradients from multiple trainers and updates the shared model via Redis.
"""
from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, should_exit
from rl_server.utils.checkpoint import save_model, load_model
from rl_server.transport.redis_cache import RedisCache
from rl_server.algorithms import create_net


class GradsAggregator:
    """
    Gradient aggregation server that collects gradients from Redis,
    accumulates them, and updates the model.
    """

    def __init__(self, env_name: str, model_prefix: str, grads_redis_config: dict,
                 model_redis_config: dict, batch_update_count: int = 10, log: Log = None):
        self.env_name = env_name
        self.model_prefix = model_prefix
        self.batch_update_count = batch_update_count
        self.log = log or Log("grads_aggregator")

        # Initialize network
        self.grads_net = create_net(env_name)

        # Load existing model if available
        self.current_train_version = load_model(
            self.grads_net, f"{model_prefix}_{env_name}"
        )
        if self.current_train_version is None:
            self.log.log_info("No existing model data, starting fresh training")
            self.current_train_version = 0

        # Initialize Redis connections
        self.grads_redis = RedisCache(self.log, grads_redis_config)
        self.grads_redis.clear_data()

        self.model_redis = RedisCache(self.log, model_redis_config)
        self.model_redis.clear_data()

        # Sync model to Redis
        self.model_redis.set_train_version_model(self.current_train_version, self.grads_net)

    def run(self):
        """Main aggregation loop."""
        self.log.log_info("Starting gradient aggregation server")

        grads_buffer = None
        grads_count = 0

        while True:
            try:
                if should_exit():
                    self.log.log_info("Shutting down gradient aggregation server")

                    # Save current model
                    save_model(self.grads_net, f"{self.model_prefix}_{self.env_name}",
                               str(self.current_train_version))

                    # Notify workers to exit
                    self.model_redis.set_exit_flag(1)

                    del self.model_redis
                    del self.grads_redis

                    self.log.log_info("Gradient aggregation server shutdown complete")
                    break

                # Aggregate gradients
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

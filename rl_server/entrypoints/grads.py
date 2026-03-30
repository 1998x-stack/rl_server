# -*- coding: utf-8 -*-
"""梯度聚合服务入口：读取配置并启动 ``GradsAggregator``。"""
import os
import argparse

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_signal_handlers
from rl_server.config.loader import load_config
from rl_server.workers.grads_aggregator import GradsAggregator


def parse_args():
    """解析命令行参数。

    Returns:
        含 ``config``、``override``、``env_name``、``prefix``。
    """
    parser = argparse.ArgumentParser(description='RL Server Gradient Aggregation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config YAML file')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment name from config')
    parser.add_argument('--prefix', type=str, default='grads_main',
                        help='Model file prefix')
    return parser.parse_args()


def main():
    """根据 YAML 构造双 Redis 配置并运行聚合主循环。"""
    args = parse_args()

    setup_seed()
    setup_signal_handlers()

    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)

    grads_log = Log("grads_main")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    grads_log.log_info(f"Gradient server starting for env: {env_name}")

    redis_cfg = config.get('redis', {})
    grads_redis_config = {
        'ip': redis_cfg.get('grads', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('grads', {}).get('port', 6379)),
        'db': str(redis_cfg.get('grads', {}).get('db', 2)),
        'pw': redis_cfg.get('grads', {}).get('password', ''),
    }
    model_redis_config = {
        'ip': redis_cfg.get('model', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('model', {}).get('port', 6379)),
        'db': str(redis_cfg.get('model', {}).get('db', 0)),
        'pw': redis_cfg.get('model', {}).get('password', ''),
    }

    batch_count = config.get('training', {}).get('batch_update_grads_server', 10)

    aggregator = GradsAggregator(
        env_name=env_name,
        model_prefix=args.prefix,
        grads_redis_config=grads_redis_config,
        model_redis_config=model_redis_config,
        batch_update_count=batch_count,
        log=grads_log
    )

    aggregator.run()


if __name__ == '__main__':
    main()

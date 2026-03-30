# -*- coding: utf-8 -*-
"""
Standalone sampler entrypoint for distributed training via Redis.
"""
import os
import argparse
import time

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_signal_handlers, should_exit
from rl_server.config.loader import load_config
from rl_server.algorithms import create_net, create_agent
from rl_server.transport.redis_cache import RedisCache


def parse_args():
    parser = argparse.ArgumentParser(description='RL Server Sampler')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config YAML file')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment name from config')
    return parser.parse_args()


def main():
    args = parse_args()

    setup_seed()
    setup_signal_handlers()

    # Load config
    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)

    sample_log = Log("sample_main")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    sample_log.log_info(f"Sampler starting for env: {env_name}")

    # Redis config
    redis_cfg = config.get('redis', {})
    model_redis_config = {
        'ip': redis_cfg.get('model', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('model', {}).get('port', 6379)),
        'db': str(redis_cfg.get('model', {}).get('db', 0)),
        'pw': redis_cfg.get('model', {}).get('password', ''),
    }
    exps_redis_config = {
        'ip': redis_cfg.get('exps', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('exps', {}).get('port', 6379)),
        'db': str(redis_cfg.get('exps', {}).get('db', 1)),
        'pw': redis_cfg.get('exps', {}).get('password', ''),
    }

    # Initialize network and agent
    sample_net = create_net(env_name)
    sample_agent = create_agent(env_name, sample_net)

    # Connect to Redis
    model_redis = RedisCache(sample_log, model_redis_config)
    exps_redis = RedisCache(sample_log, exps_redis_config)

    model_dict = {'TRAIN_VERSION': 0}

    sample_log.log_info("Sampler running")

    while True:
        try:
            if should_exit():
                sample_log.log_info("Sampler shutting down")
                break

            # Check for exit flag from Redis
            exit_flag = model_redis.get_exit_flag()
            if exit_flag and int(exit_flag) == 1:
                sample_log.log_info("Exit flag received from Redis")
                break

            # Sync model from Redis
            version = model_redis.get_train_version()
            if version is not None and version != model_dict['TRAIN_VERSION']:
                model_redis.get_train_model(sample_net)
                model_dict['TRAIN_VERSION'] = version

            # Sample
            exps_list = sample_agent.sample_multi_envs(model_dict)
            if exps_list is not None:
                for exps in exps_list:
                    exps_redis.push_exps(exps, model_dict['TRAIN_VERSION'])

            time.sleep(0)

        except Exception:
            sample_log.log_exception()

    sample_log.log_info("Sampler exited")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""模型评估入口：加载检查点（可选）并在环境中循环评估，写入 TensorBoard。"""
import os
import argparse
import time
from tensorboardX import SummaryWriter

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_signal_handlers, should_exit
from rl_server.utils.checkpoint import load_model
from rl_server.config.loader import load_config
from rl_server.algorithms import create_net, create_agent


def parse_args():
    """解析命令行参数。

    Returns:
        含配置路径、环境名、模型前缀、版本与评估回合数等。
    """
    parser = argparse.ArgumentParser(description='RL Server Model Checker')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config YAML file')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment name from config')
    parser.add_argument('--prefix', type=str, default='checker',
                        help='Model file prefix')
    parser.add_argument('--version', type=str, default=None,
                        help='Model version to evaluate')
    parser.add_argument('--episodes', type=int, default=0,
                        help='Number of episodes to evaluate (0 = continuous)')
    return parser.parse_args()


def main():
    """创建检查智能体，可选加载权重，按回合评估直至信号或达到 ``episodes``。"""
    args = parse_args()

    setup_seed()
    setup_signal_handlers()

    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)

    check_log = Log("checker")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    check_log.log_info(f"Checker starting for env: {env_name}")

    check_net = create_net(env_name)
    check_agent = create_agent(env_name, check_net, is_checker=True)

    if args.version:
        loaded_version = load_model(check_net, f"{args.prefix}_{env_name}", args.version)
        if loaded_version is None:
            check_log.log_info(f"Could not load model version {args.version}")
        else:
            check_log.log_info(f"Loaded model version {loaded_version}")

    check_net.train(False)
    writer = SummaryWriter(comment="checker")

    episode = 0
    check_log.log_info("Checker running")

    while True:
        try:
            if should_exit():
                check_log.log_info("Checker shutting down")
                break

            infos = check_agent.check_single_env()
            if isinstance(infos, dict):
                for key, value in infos.items():
                    writer.add_scalar(key, value, episode)
                    check_log.log_info(f"Episode {episode}: {key}={value}")

            episode += 1

            if args.episodes > 0 and episode >= args.episodes:
                break

            time.sleep(0)

        except Exception:
            check_log.log_exception()

    writer.close()
    check_log.log_info("Checker exited")


if __name__ == '__main__':
    main()

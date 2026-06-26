# -*- coding: utf-8 -*-
"""本地多进程训练入口：Trainer / Sampler / Checker 与主进程梯度聚合。

基于 ``train_main_local/train_main_local.py``，统一使用 ``rl_server`` 包内模块。
"""
import os
import argparse
import time
import queue
import torch.multiprocessing as mp

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_mp, setup_seed, should_exit, setup_signal_handlers
from rl_server.utils.checkpoint import save_model, load_model
from rl_server.config.loader import load_config
from rl_server.algorithms import create_net, set_device
from rl_server.workers.sampler import SamplerWorker
from rl_server.workers.trainer import TrainerWorker
from rl_server.workers.checker import CheckerWorker


def parse_args():
    """解析命令行参数。

    Returns:
        ``argparse.Namespace``：含 ``config``、``override``、``env_name``、``prefix``、``version``。
    """
    parser = argparse.ArgumentParser(description='RL Server Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config YAML file')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment name from config')
    parser.add_argument('--prefix', type=str, default='train_main_local',
                        help='Model file prefix')
    parser.add_argument('--version', type=str, default=None,
                        help='Model version to load')
    return parser.parse_args()


def main():
    """加载配置、启动子进程、在主循环中聚合梯度并处理优雅退出。"""
    args = parse_args()

    setup_mp()
    setup_seed()
    setup_signal_handlers()

    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml'
    )
    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), '..', '..', 'config', 'default.yaml'
        )
    config = load_config(config_path, args.override)

    train_log = Log("train_main_local")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    model_prefix = args.prefix
    model_version = args.version

    train_log.log_info(f"Current training algo_env is {env_name}")

    training_cfg = config.get('training', {})
    queues_cfg = config.get('queues', {})
    num_trainers = training_cfg.get('num_trainers', 1)
    num_samplers = training_cfg.get('num_samplers', 2)
    num_update_grads = training_cfg.get('num_update_grads', 1)
    enable_checker = training_cfg.get('enable_checker', True)

    grads_queue = mp.Queue(maxsize=queues_cfg.get('len_grads_queue', 1000))
    sample_queue = mp.Queue(maxsize=queues_cfg.get('len_sample_queue', 1000))

    device_str = training_cfg.get('device', 'cpu')
    set_device(env_name, device_str)
    train_log.log_info(f"Using device: {device_str}")

    train_net = create_net(env_name)
    if next(train_net.parameters()).device.type == 'cpu':
        train_net.share_memory()
    else:
        train_log.log_info("GPU mode: skipping share_memory (workers sync via load_state_dict)")

    current_train_version = load_model(train_net, f"{model_prefix}_{env_name}", model_version)
    if current_train_version is None:
        train_log.log_info("No existing model data, starting fresh training")
        current_train_version = 0
    else:
        current_train_version = int(current_train_version)
        train_log.log_info(f"Resumed from version {current_train_version}")

    model_dict = mp.Manager().dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = current_train_version

    trainers = []
    samplers = []

    for i in range(num_trainers):
        t = TrainerWorker(
            idx=i,
            model_dict=model_dict,
            share_model=train_net,
            sample_queue=sample_queue,
            grads_queue=grads_queue,
            env_name=env_name,
            log=train_log
        )
        trainers.append(t)
        t.run()

    for i in range(num_samplers):
        s = SamplerWorker(
            idx=i,
            model_dict=model_dict,
            share_model=train_net,
            sample_queue=sample_queue,
            env_name=env_name,
            log=train_log
        )
        samplers.append(s)
        s.run()

    train_checker = None
    if enable_checker:
        train_checker = CheckerWorker(
            model_dict=model_dict,
            share_model=train_net,
            env_name=env_name,
            log=train_log
        )
        train_checker.run("_algo_env_mixed")

    train_log.log_info("Started train_main_local")

    grads_buffer = None
    grads_count = 0

    while True:
        try:
            if should_exit():
                train_log.log_info("Shutting down train_main_local")

                save_model(train_net, f"{model_prefix}_{env_name}", str(current_train_version))
                model_dict['is_exit'] = True

                for s in samplers:
                    s.stop()
                for t in trainers:
                    t.stop()
                if train_checker:
                    train_checker.stop()

                train_log.log_info("Shutdown complete")
                break

            if grads_count >= num_update_grads:
                current_train_version += 1
                train_net.update_state(current_train_version, grads_buffer)
                model_dict['TRAIN_VERSION'] = current_train_version

                grads_count = 0
                grads_buffer = None
            else:
                try:
                    grads_info = grads_queue.get(block=False)
                    grads_item = grads_info['grads']
                    grads_count += 1
                    if grads_buffer is None:
                        grads_buffer = grads_item
                    else:
                        for target_grad, grad in zip(grads_buffer, grads_item):
                            target_grad += grad
                except queue.Empty:
                    pass
            time.sleep(0)
        except Exception:
            train_log.log_exception()

    train_log.log_info("Exit OK")


if __name__ == '__main__':
    main()

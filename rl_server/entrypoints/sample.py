# -*- coding: utf-8 -*-
"""独立采样进程入口：通过 Redis 同步模型并上推经验批次。"""
import os
import argparse
import time

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_seed, setup_signal_handlers, should_exit
from rl_server.config.loader import load_config
from rl_server.algorithms import create_net, create_agent
from rl_server.transport.redis_cache import RedisCache

EXIT_SIGNAL = "signal"
EXIT_REDIS = "redis_exit_flag"


def parse_args():
    """解析命令行参数。

    Returns:
        含 ``config``、``override``、``env_name`` 的命名空间。
    """
    parser = argparse.ArgumentParser(description='RL Server Sampler')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config YAML file')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment name from config')
    return parser.parse_args()


def _push_exps(exps_list, exps_redis, model_dict, max_queue_size, log, stats):
    """把一批经验推入 Redis；队列满时阻塞等待。

    Args:
        exps_list: ``sample_multi_envs`` 返回的经验列表。
        exps_redis: 经验队列的 ``RedisCache``。
        model_dict: 含 ``TRAIN_VERSION`` 的字典。
        max_queue_size: 队列上限，``None`` 表示不限制。
        log: 日志对象。
        stats: 计数用字典，原地更新。

    Returns:
        退出原因字符串（``None`` 表示已全部推完）。
    """
    for exps in exps_list:
        while not should_exit():
            if exps_redis.get_exit_flag() == 1:
                return EXIT_REDIS
            if exps_redis.push_exps(
                exps, model_dict['TRAIN_VERSION'], max_queue_size=max_queue_size
            ):
                stats['batches'] += 1
                break
            stats['queue_full_waits'] += 1
            log.info_throttled(
                "sample_queue_full",
                f"sample queue at capacity waits={stats['queue_full_waits']} "
                f"limit={max_queue_size}",
                interval=30,
            )
            time.sleep(0.1)
        else:
            return EXIT_SIGNAL
    return None


def main():
    """连接模型 Redis 与经验 Redis，循环拉取最新权重并推送经验。"""
    args = parse_args()

    setup_seed()
    setup_signal_handlers()

    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)

    sample_log = Log.from_config(config, "sample_main")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')

    redis_cfg = config.get('redis', {})
    model_redis_config = {
        'ip': redis_cfg.get('model', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('model', {}).get('port', 6379)),
        'db': str(redis_cfg.get('model', {}).get('db', 0)),
        'pw': redis_cfg.get('model', {}).get('password', ''),
        'namespace': redis_cfg.get('model', {}).get('namespace', ''),
    }
    exps_redis_config = {
        'ip': redis_cfg.get('exps', {}).get('host', 'localhost'),
        'port': str(redis_cfg.get('exps', {}).get('port', 6379)),
        'db': str(redis_cfg.get('exps', {}).get('db', 1)),
        'pw': redis_cfg.get('exps', {}).get('password', ''),
        'namespace': redis_cfg.get('exps', {}).get('namespace', ''),
    }

    max_queue_size = config.get('queues', {}).get('len_sample_queue', 1000)
    if max_queue_size <= 0:
        max_queue_size = None

    # 启动即写清连到哪、队列上限多少，避免排查时只能翻代码。
    sample_log.log_info(
        f"start env={env_name} pid={os.getpid()} "
        f"model_redis={model_redis_config['ip']}:{model_redis_config['port']}/db{model_redis_config['db']} "
        f"exps_redis={exps_redis_config['ip']}:{exps_redis_config['port']}/db{exps_redis_config['db']} "
        f"namespace={model_redis_config['namespace']} sample_queue_limit={max_queue_size}"
    )

    sample_net = create_net(env_name)
    sample_agent = create_agent(env_name, sample_net)

    model_redis = RedisCache(sample_log, model_redis_config)
    exps_redis = RedisCache(sample_log, exps_redis_config)

    model_dict = {'TRAIN_VERSION': None}
    stats = {'batches': 0, 'queue_full_waits': 0, 'weight_syncs': 0}
    started = time.time()
    exit_reason = None

    sample_log.log_info("Sampler running")

    while exit_reason is None:
        try:
            if should_exit():
                exit_reason = EXIT_SIGNAL
                break

            if model_redis.get_exit_flag() == 1:
                exit_reason = EXIT_REDIS
                break

            version = model_redis.get_train_version()
            if version is None:
                sample_log.info_throttled(
                    "waiting_for_model",
                    "no train version in redis yet, waiting",
                    interval=30,
                )
                time.sleep(0.1)
                continue
            if version != model_dict['TRAIN_VERSION']:
                loaded_version = model_redis.get_train_snapshot(sample_net)
                if loaded_version is None:
                    sample_log.info_throttled(
                        "waiting_for_snapshot",
                        f"model snapshot not readable yet version={version}",
                        interval=30,
                    )
                    time.sleep(0.1)
                    continue
                model_dict['TRAIN_VERSION'] = loaded_version
                stats['weight_syncs'] += 1

            exps_list = sample_agent.sample_multi_envs(model_dict)
            if exps_list is not None:
                exit_reason = _push_exps(
                    exps_list, exps_redis, model_dict, max_queue_size, sample_log, stats
                )

            elapsed = max(time.time() - started, 1e-6)
            sample_log.info_throttled(
                "sampler_progress",
                f"sampler progress version={model_dict['TRAIN_VERSION']} "
                f"batches={stats['batches']} batches_per_sec={stats['batches'] / elapsed:.2f} "
                f"weight_syncs={stats['weight_syncs']} queue_full_waits={stats['queue_full_waits']}",
                interval=30,
            )

            time.sleep(0)

        except Exception:
            # 带上阶段信息，否则只剩一个无法定位的 traceback。
            sample_log.exception("sampler loop")

    elapsed = max(time.time() - started, 1e-6)
    sample_log.log_info(
        f"Sampler exited reason={exit_reason} batches={stats['batches']} "
        f"batches_per_sec={stats['batches'] / elapsed:.2f} "
        f"weight_syncs={stats['weight_syncs']} queue_full_waits={stats['queue_full_waits']} "
        f"elapsed={elapsed:.0f}s"
    )
    sample_log.summary({
        'version': model_dict['TRAIN_VERSION'],
        'batches': stats['batches'],
    })


if __name__ == '__main__':
    main()

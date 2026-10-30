# -*- coding: utf-8 -*-
"""本地多进程训练入口：Trainer / Sampler / Checker 与主进程梯度聚合。
"""
import collections
import os
import argparse
import time
import queue
import torch.multiprocessing as mp
import numpy as np

from rl_server.utils.logging import Log
from rl_server.utils.process import setup_mp, setup_seed, should_exit, setup_signal_handlers
from rl_server.utils.checkpoint import save_model, load_model
from rl_server.core.gradient_contract import (
    DEFAULT_LAG_CAP,
    REASON_FUTURE,
    REASON_STALE,
    gradient_batch_rejection,
    is_auto_lag,
    resolve_max_gradient_lag,
)
from rl_server.config.loader import load_config
from rl_server.algorithms import create_net, set_device
from rl_server.core.optim import trainable_parameters
from rl_server.workers.sampler import SamplerWorker
from rl_server.workers.trainer import TrainerWorker
from rl_server.workers.checker import CheckerWorker
from tensorboardX import SummaryWriter


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


def _stop_workers(workers, log, timeout: float = 15.0):
    """逐个关闭 worker，并保证不无限等待。

    Args:
        workers: 拥有 ``stop()`` 的工作者列表（元素可为 ``None``）。
        log: 日志对象。
        timeout: 单个 worker 的等待秒数。
    """
    for worker in workers:
        if worker is None:
            continue
        try:
            worker.stop(timeout=timeout)
        except Exception:
            log.exception(f"stopping worker={type(worker).__name__} failed")


def _run_shutdown(steps, log) -> None:
    """一次性执行关闭步骤：每一步单独兜底，任何失败都不再重入。

    每步耗时写入 DEBUG，超过 5s 的步骤告警——用于定位“退出卡很久”的场景。

    Args:
        steps: ``(名字, 无参可调用)`` 序列。
        log: 日志对象。
    """
    for name, action in steps:
        started = time.time()
        try:
            action()
        except Exception:
            log.exception(f"shutdown step={name} failed")
            continue
        elapsed = time.time() - started
        log.debug(f"shutdown step={name} took={elapsed:.3f}s")
        if elapsed > 5.0:
            log.warning(f"shutdown step={name} slow took={elapsed:.1f}s")


def main():
    """加载配置、启动子进程、在主循环中聚合梯度并处理优雅退出。"""
    args = parse_args()

    setup_mp()
    setup_seed()
    setup_signal_handlers()

    config_path = args.config or os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', 'config', 'default.yaml'
    )
    config = load_config(config_path, args.override)

    train_log = Log.from_config(config, "train_main_local")

    env_name = args.env_name or config.get('training', {}).get('env_name', 'DQNGymClassic')
    model_prefix = args.prefix
    model_version = args.version

    training_cfg = config.get('training', {})
    queues_cfg = config.get('queues', {})
    num_trainers = training_cfg.get('num_trainers', 1)
    num_samplers = training_cfg.get('num_samplers', 2)
    num_update_grads = training_cfg.get('num_update_grads', 1)
    checkpoint_interval = training_cfg.get('checkpoint_interval', 100)
    max_versions = training_cfg.get('max_versions', 10000)
    enable_checker = training_cfg.get('enable_checker', True)
    configured_max_lag = training_cfg.get('max_gradient_lag', 'auto')
    lag_cap = int(training_cfg.get('max_gradient_lag_cap', DEFAULT_LAG_CAP))
    progress_interval = training_cfg.get('version_log_interval', 10)
    cold_start_versions = int(training_cfg.get('cold_start_versions', 20))
    stall_warn_secs = training_cfg.get('stall_warn_secs', 60)
    stall_fatal_secs = training_cfg.get('stall_fatal_secs', 300)
    grads_queue_size = queues_cfg.get('len_grads_queue', 1000)
    sample_queue_size = queues_cfg.get('len_sample_queue', 1000)

    device_str = training_cfg.get('device', 'cpu')
    try:
        set_device(env_name, device_str)
    except Exception:
        train_log.exception(f"algorithm load failed env={env_name} device={device_str}")
        raise SystemExit(2)
    # 启动配置一次性写全，避免“要看代码才知道跑了什么”。
    train_log.log_info(
        f"start env={env_name} device={device_str} samplers={num_samplers} trainers={num_trainers} "
        f"num_update_grads={num_update_grads} max_gradient_lag={configured_max_lag} "
        f"lag_cap={lag_cap} "
        f"checkpoint_interval={checkpoint_interval} max_versions={max_versions} "
        f"stall_warn={stall_warn_secs}s stall_fatal={stall_fatal_secs}s "
        f"checker={enable_checker} queues(grads={grads_queue_size},samples={sample_queue_size}) "
        f"model_prefix={model_prefix}"
    )

    grads_queue = mp.Queue(maxsize=grads_queue_size)
    sample_queue = mp.Queue(maxsize=sample_queue_size)

    if (not is_auto_lag(configured_max_lag) and configured_max_lag == 0
            and (num_trainers + num_samplers) > 1):
        # 严格同步下，任何在途批次都会因版本号过期而被丢弃（实测约 50%）。
        train_log.warning(
            "max_gradient_lag=0 with concurrent workers: in-flight gradient batches will be "
            "rejected as stale, wasting trainer compute; set training.max_gradient_lag >= 1"
        )

    train_net = create_net(env_name)
    if next(train_net.parameters()).device.type == 'cpu':
        train_net.share_memory()
    else:
        train_log.log_info("GPU mode: skipping share_memory (workers sync via load_state_dict)")

    current_train_version = load_model(train_net, f"{model_prefix}_{env_name}", model_version)
    missing_keys = getattr(train_net, '_checkpoint_missing_keys', []) or []
    if missing_keys and hasattr(train_net, 'sync_targets'):
        # 旧检查点不含目标 critic（例如 SAC 结构升级前）：用在线权重重同步后再训练。
        train_net.sync_targets()
        train_log.warning(
            f"target networks re-synced from online weights missing_keys={missing_keys[:4]}"
        )
    if current_train_version is None:
        train_log.log_info("No existing model data, starting fresh training")
        current_train_version = 0
    else:
        current_train_version = int(current_train_version)
        train_log.log_info(f"Resumed from version {current_train_version}")

    manager = mp.Manager()
    model_dict = manager.dict()
    model_dict['is_exit'] = False
    model_dict['TRAIN_VERSION'] = current_train_version
    model_dict['check_log_interval'] = training_cfg.get('check_log_interval', 100)

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

    # Training metrics writer for TensorBoard
    tb_override = os.environ.get('RL_SERVER_TENSORBOARD_DIR')
    train_writer = SummaryWriter(logdir=tb_override) if tb_override else SummaryWriter(comment="train")
    train_log.log_info(
        f"tensorboard logdir={os.path.abspath(train_writer.logdir)} "
        f"models_dir={os.path.abspath('models')}"
    )

    all_workers = samplers + trainers + ([train_checker] if train_checker else [])
    dead_workers = {}
    # auto 模式在观测到批次规模前先保守取 1，避免未定义引用。
    effective_max_lag = resolve_max_gradient_lag(configured_max_lag, None, lag_cap)

    grads_buffer = None
    grads_count = 0
    rejected_gradients = 0
    cold_start_rejected = 0  # 启动瞬态：worker 首批梯度必然来自 version=0
    accepted_gradients = 0
    reject_reasons = collections.Counter()
    # 每条梯度"到达时的滞后"（current_version - grads_version）。它直接反映流水线延迟：
    # lag 设得比它小，就等于把正常在途的梯度当垃圾扔掉。
    arrival_lags = collections.deque(maxlen=256)
    parameters = trainable_parameters(train_net)
    training_started = time.time()
    last_progress_ts = training_started
    shutdown_started = False
    latest_metrics = None

    def _shutdown(reason):
        """一次性关闭：写入模型、通知 worker、逐个收工；任何步骤失败都只记录不重试。"""
        train_log.log_info(
            f"shutdown requested reason={reason} version={current_train_version} "
            f"accepted={accepted_gradients} rejected={rejected_gradients}"
        )
        _run_shutdown((
            ("save_model", lambda: save_model(
                train_net, f"{model_prefix}_{env_name}", str(current_train_version))),
            ("signal_workers", lambda: model_dict.__setitem__('is_exit', True)),
            ("stop_samplers", lambda: _stop_workers(samplers, train_log)),
            ("stop_trainers", lambda: _stop_workers(trainers, train_log)),
            ("stop_checker", lambda: _stop_workers([train_checker], train_log)),
            ("close_queues", lambda: _close_queues([grads_queue, sample_queue])),
            ("close_writer", train_writer.close),
            ("shutdown_manager", manager.shutdown),
        ), train_log)
        elapsed = max(time.time() - training_started, 1e-6)
        detail = ",".join(f"{name}={count}" for name, count in reject_reasons.most_common())
        train_log.log_info(
            f"shutdown complete version={current_train_version} elapsed={elapsed:.0f}s "
            f"versions_per_sec={current_train_version / elapsed:.2f} "
            f"accepted={accepted_gradients} rejected={rejected_gradients} "
            f"cold_start_rejected={cold_start_rejected} reasons=[{detail}]"
        )
        train_log.summary({
            'version': current_train_version,
            'accepted': accepted_gradients,
            'rejected': rejected_gradients,
        })

    while True:
        try:
            if should_exit() and not shutdown_started:
                # 关闭只跑一次：旧实现把关闭放在热循环里，一旦 manager 已失效，
                # 写 is_exit 抛 BrokenPipeError 后会被 except 捕获并无限重入（实测 6.5 万次）。
                shutdown_started = True
                _shutdown("signal")
                break

            if grads_count >= num_update_grads:
                current_train_version += 1
                train_net.update_state(current_train_version, grads_buffer)
                model_dict['TRAIN_VERSION'] = current_train_version

                # 指标随梯度批次而来：每个版本都能拿到自己的 loss/norm（原先靠版本对号，只有 1/N 命中）。
                metric_text = ""
                if isinstance(latest_metrics, dict):
                    for key, value in latest_metrics.items():
                        if key in ('grads_version', 'sample_version'):
                            continue
                        if isinstance(value, (int, float)):
                            train_writer.add_scalar(f"train/{key}", value, current_train_version)
                            metric_text += f" {key}={value:.4g}"
                    train_writer.flush()

                _check_backlog(_queue_depth_int(grads_queue), grads_queue_size,
                               train_log, num_trainers)
                elapsed = max(time.time() - training_started, 1e-6)
                sample_lag = ""
                if isinstance(latest_metrics, dict) and isinstance(
                        latest_metrics.get('sample_version'), int):
                    sample_lag = f" sample_lag={current_train_version - latest_metrics['sample_version']}"
                if arrival_lags:
                    sample_lag += (f" arrival_lag_p95={_arrival_lag_p95(arrival_lags)}"
                                   f" max_lag={effective_max_lag}")
                _total = accepted_gradients + rejected_gradients
                train_log.info_every(
                    "version_progress", progress_interval,
                    f"version={current_train_version} rate={current_train_version / elapsed:.2f}ver/s "
                    f"accepted={accepted_gradients} rejected={rejected_gradients} "
                    f"cold_start_rejected={cold_start_rejected} reject_pct={100.0 * rejected_gradients / _total if _total else 0.0:.1f} "
                    f"queue_grads={_queue_depth(grads_queue)}{sample_lag}{metric_text}",
                )

                if (checkpoint_interval > 0
                        and current_train_version % checkpoint_interval == 0):
                    saved_path = save_model(train_net, f"{model_prefix}_{env_name}",
                                            str(current_train_version), max_versions=max_versions)
                    if saved_path:
                        train_log.log_info(
                            f"checkpoint saved version={current_train_version} path={saved_path}"
                        )
                    else:
                        train_log.error(
                            f"checkpoint save failed version={current_train_version} "
                            f"prefix={model_prefix}_{env_name}"
                        )

                grads_count = 0
                grads_buffer = None
                continue

            try:
                grads_info = grads_queue.get(timeout=0.05)
            except queue.Empty:
                # 没有梯度时睡在带超时的 get 上，避免空转烧 CPU（旧实现是 time.sleep(0) 死循环）。
                _check_trainer_heartbeats(model_dict, num_trainers, train_log,
                                          started_at=training_started)
                idle = time.time() - last_progress_ts
                if idle > stall_warn_secs:
                    train_log.warning_throttled(
                        "no_progress",
                        f"no training progress idle={idle:.0f}s accepted={accepted_gradients} "
                        f"rejected={rejected_gradients} dead_workers={len(dead_workers)}/{len(all_workers)} "
                        f"queue_grads={_queue_depth(grads_queue)}",
                        interval=max(stall_warn_secs, 30),
                    )
                if stall_fatal_secs > 0 and idle > stall_fatal_secs:
                    shutdown_started = True
                    _shutdown("stalled")
                    train_log.error(
                        f"aborting: no gradient applied for {idle:.0f}s "
                        f"accepted={accepted_gradients} rejected={rejected_gradients} "
                        f"dead_workers={len(dead_workers)}/{len(all_workers)}"
                    )
                    raise SystemExit(1)

                dead = _report_dead_workers(all_workers, train_log, dead_workers)
                if dead >= len(all_workers):
                    # 所有 worker 都死了：再跑下去只是空转，直接以失败退出。
                    shutdown_started = True
                    _shutdown("all_workers_dead")
                    train_log.error(
                        f"aborting: every worker exited version={current_train_version} "
                        f"accepted={accepted_gradients}"
                    )
                    raise SystemExit(1)
                continue

            grads_item = grads_info.get('grads')
            grads_version = grads_info.get('grads_version')
            # auto 模式：用实测的"每批梯度条数"推导能覆盖整批的滞后上限。
            if type(grads_version) is int:
                arrival_lags.append(max(0, current_train_version - grads_version))
            if is_auto_lag(configured_max_lag):
                effective_max_lag = _effective_lag(configured_max_lag, arrival_lags, lag_cap)
            else:
                effective_max_lag = configured_max_lag
            reason = gradient_batch_rejection(
                grads_item, grads_version, parameters, current_train_version, effective_max_lag,
            )
            if reason is not None:
                # 旧日志只说“stale or invalid”，既不能看出原因也不分项统计。
                # 启动瞬态单独归类：worker 首批梯度必然来自 version=0，混进拒绝率会虚高。
                if current_train_version <= cold_start_versions:
                    cold_start_rejected += 1
                    continue
                rejected_gradients += 1
                reject_reasons[reason] += 1
                # lag_behind = 当前版本 - 梯度版本；非负表示"落后几版"（负数只会出现在
                # 未来版本上，属于 bug），写成正数语义可以避免被误读。
                lag_behind = (current_train_version - grads_version
                              if type(grads_version) is int else None)
                train_log.warning_throttled(
                    f"grad_reject_{reason}",
                    f"gradient batch rejected reason={reason} grads_version={grads_version} "
                    f"current_version={current_train_version} lag_behind={lag_behind} "
                    f"total={rejected_gradients}",
                    interval=15,
                )
                total_batches = accepted_gradients + rejected_gradients
                if (rejected_gradients >= 20 and rejected_gradients / total_batches >= 0.2
                        and len(arrival_lags) >= 8):
                    # 只有"版本过期"类原因才值得讨论 lag；数据/结构类问题要给不同提示。
                    if reason in (REASON_STALE, REASON_FUTURE):
                        hint = (f"effective_lag={effective_max_lag} "
                                f"arrival_lag_p95={_arrival_lag_p95(arrival_lags)} "
                                f"hint=use max_gradient_lag: auto")
                    else:
                        hint = f"hint={_reason_hint(reason)}"
                    train_log.warning_throttled(
                        "grad_reject_rate",
                        f"gradient rejection rate high "
                        f"reject_pct={100.0 * rejected_gradients / total_batches:.1f} "
                        f"rejected={rejected_gradients}/{total_batches} "
                        f"{hint} reason={reason} max_gradient_lag={configured_max_lag} "
                        f"effective_lag={effective_max_lag} "
                        f"arrival_lag_p95={_arrival_lag_p95(arrival_lags)}",
                        interval=60,
                    )
                continue

            latest_metrics = grads_info.get('metrics')
            accepted_gradients += 1
            last_progress_ts = time.time()
            grads_count += 1
            if grads_buffer is None:
                grads_buffer = grads_item
            else:
                for target_grad, grad in zip(grads_buffer, grads_item):
                    np.add(target_grad, grad, out=target_grad)
        except Exception:
            train_log.exception("train main loop")

    train_log.log_info("Exit OK")


def _close_queues(queues) -> None:
    """关闭 mp.Queue 并取消 feeder 的 join。

    不关闭会在解释器退出时留下 \"leaked semaphore objects\" 警告（实测 3 个），
    并可能让退出卡住。
    """
    for q in queues:
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass


def _arrival_lag_p95(arrival_lags) -> str:
    """梯度到达时滞后的 p95（``current_version - grads_version``）。

    这是"正常在途"的滞后规模：``max_gradient_lag`` 小于它时，健康流水线产出的梯度
    也会被判过期。展示它比给一个静态建议值更有用。
    """
    if not arrival_lags:
        return "?"
    ordered = sorted(arrival_lags)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return str(ordered[index])


def _effective_lag(configured, arrival_lags, cap: int) -> int:
    """解析本轮的 ``max_gradient_lag``：显式整数原样使用，``auto`` 按到达滞后自适应。"""
    if not is_auto_lag(configured):
        return configured
    if not arrival_lags:
        return 1
    ordered = sorted(arrival_lags)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return min(int(cap), max(1, ordered[index]))


def _suggested_lag(arrival_lags) -> str:
    """给出建议的固定 lag（取 p95 到达滞后）。"""
    return _arrival_lag_p95(arrival_lags)


def _reason_hint(reason: str) -> str:
    """把拒绝原因翻译成可执行的排查提示。"""
    return {
        'none_gradient': 'algorithm produced params without grads (loss does not cover them)',
        'shape_mismatch': 'gradient shape != parameter shape (network changed mid-run?)',
        'non_finite': 'gradient contains NaN/Inf (loss diverged, lower lr or clip)',
        'length_mismatch': 'gradient count != parameter count (incompatible checkpoint)',
        'version_type': 'grads_version missing or not int (producer bug)',
        'non_float_dtype': 'gradient dtype is not floating point',
        'not_sequence': 'gradient payload is not a list/tuple',
    }.get(reason, 'inspect the producing calculator')


def _suggested_lag(grads_per_batch) -> str:
    """给出能覆盖整批梯度的 ``max_gradient_lag`` 建议值。

    用**观察到的最大批次**而不是均值：均值里混着"还在生产中"的半批，会低估。
    """
    if not grads_per_batch:
        return "?"
    return str(max(1, max(grads_per_batch.values()) - 1))


def _check_backlog(depth, capacity, log, num_trainers: int) -> None:
    """梯度队列积压告警：生产快于消费时队列会一直涨，梯度也就一直陈旧。"""
    if not capacity or capacity <= 0:
        return
    if depth < max(10, capacity * 0.25):
        return
    log.warning_throttled(
        "grads_backlog",
        f"grads queue backlog depth={depth}/{capacity} ({100.0 * depth / capacity:.0f}%): "
        f"trainers produce faster than the main loop applies; consider raising "
        f"training.num_trainers (now {num_trainers}) or lowering num_update_grads "
        f"— otherwise stale batches accumulate",
        interval=60,
    )


def _queue_depth_int(q) -> int:
    """安全读取队列深度为整数（失败时返回 0）。"""
    try:
        return int(q.qsize())
    except (NotImplementedError, OSError, ValueError):
        return 0


def _queue_depth(q) -> str:
    """安全读取队列深度（Unix 上可用，失败时返回 ``?``）。"""
    try:
        return str(q.qsize())
    except (NotImplementedError, OSError):
        return "?"


def _check_trainer_heartbeats(model_dict, num_trainers, log, startup_grace: float = 60.0,
                              started_at: float = 0.0):
    """心跳检测：只在真正空闲时检查，且每个 trainer 限流 30s 一条。"""
    now = time.time()
    for i in range(num_trainers):
        hb = model_dict.get(f'trainer_heartbeat_{i}', 0)
        if not hb:
            # 从未写过心跳：超过启动宽限期就是异常（例如 worker 直接崩在初始化里）。
            if started_at and now - started_at > startup_grace:
                log.warning_throttled(
                    f"heartbeat_missing_{i}",
                    f"trainer never reported a heartbeat trainer={i} uptime={now - started_at:.0f}s",
                    interval=60,
                )
            continue
        if now - hb > 30:
            log.warning_throttled(
                f"heartbeat_trainer_{i}",
                f"trainer heartbeat stale trainer={i} idle={now - hb:.0f}s",
                interval=30,
            )


def _report_dead_workers(workers, log, reported) -> int:
    """记录新死的 worker（只报一次）；返回当前已死的数量。"""
    for worker in workers:
        if worker is None:
            continue
        process = getattr(worker, 'process', None)
        if process is None:
            continue
        key = id(worker)
        if key in reported:
            continue
        if not process.is_alive():
            reported[key] = True
            log.error(
                f"worker died worker={type(worker).__name__} pid={process.pid} "
                f"exitcode={process.exitcode}"
            )
    return len(reported)


if __name__ == '__main__':
    main()

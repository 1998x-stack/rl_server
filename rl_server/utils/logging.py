# -*- coding: utf-8 -*-
"""结构化日志：按进程写入旋转文件，控制台可选输出，内置限流与异常折叠。

本模块针对线上排查时暴露的几类故障做过加固，改动请连同 ``tests/unit/test_logging.py`` 一起维护：

1. ``spawn`` 子进程不继承父进程的 handler。若 ``Log`` 被 pickled 传给 worker 而不重建
   handler，子进程里的 ``info``/``error`` 会全部丢失（只有 WARNING 以上经
   ``logging.lastResort`` 打到 stderr，且没有时间戳与进程信息）。``Log.__getstate__`` /
   ``__setstate__`` 因此在子进程中重新建 handler。
2. ``logger.setLevel(level)`` 会连带过滤掉文件 handler 想收的 DEBUG 记录。现在 logger
   固定 DEBUG，只有控制台 handler 受 ``level`` 约束，"文件始终 DEBUG" 才成立。
3. 多进程同时轮转同一个文件会损坏日志。主进程写 ``<dir_name>.log``，其它进程写
   ``<dir_name>.<pid>.log``，互不干扰。
4. 异常去重原先以 ``type:str(exc_value)`` 为 key：带 id/计数的异常永远去重不掉，不同
   调用点相同的消息又会被合并。现在以「异常类型 + 最深栈帧位置 + 归一化消息」为 key，
   重复只输出一行摘要（默认 60s 一次），计数由 :meth:`Log.summary` 在退出时汇总。
5. 普通循环消息（如关闭流程、队列超时）原先没有任何限流手段，一次异常退出可写出
   6 万行同一条 INFO。现在提供 :meth:`Log.info_throttled` / :meth:`Log.info_every` /
   :meth:`Log.once`。
"""
import os
import re
import sys
import time
import logging
import logging.handlers
import traceback
from typing import Any, Dict, Optional

_DEFAULT_THROTTLE_SECS = 60.0
_EXC_REPORT_SECS = 60.0
_DIGITS = re.compile(r'\d+')
_MAX_VALUE_CHARS = 160


def _repo_root() -> str:
    """日志根目录：``RL_SERVER_LOG_ROOT`` 优先，否则仓库根。

    该值用于解析相对的 ``logging.dir``；环境变量既方便容器化部署，也让隔离测试把
    日志写到临时目录，不污染工作区。
    """
    override = os.environ.get('RL_SERVER_LOG_ROOT')
    if override:
        return os.path.abspath(override)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _normalize(value: Any, limit: int = _MAX_VALUE_CHARS) -> str:
    """把消息中的数字折叠为 ``#``，让 "timeout 1234" 与 "timeout 5678" 视为同类。"""
    text = _DIGITS.sub('#', str(value))
    return text[:limit]


def _exc_origin(tb) -> str:
    """取最深栈帧作为异常来源标识；缺栈帧时退化为 ``?``。"""
    if tb is None:
        return '?'
    frames = traceback.extract_tb(tb)
    if not frames:
        return '?'
    last = frames[-1]
    return f"{os.path.basename(last.filename)}:{last.lineno}:{last.name}"


def _resolve_log_path(base_dir: Optional[str], dir_name: str, pid_suffix: str) -> str:
    """解析并创建日志文件路径。

    Args:
        base_dir: 日志根目录；相对路径按仓库根解析，``None`` 时使用 ``<repo>/logs``。
        dir_name: 日志子目录（通常与记录器同名）。
        pid_suffix: 进程后缀，主进程为空串，子进程为 ``.<pid>``。

    Returns:
        日志文件的绝对路径。
    """
    if base_dir is None:
        root = os.path.join(_repo_root(), 'logs')
    elif os.path.isabs(str(base_dir)):
        root = str(base_dir)
    else:
        root = os.path.join(_repo_root(), str(base_dir))
    log_dir = os.path.join(root, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'{dir_name}{pid_suffix}.log')


def setup_logging(
    dir_name: str,
    level: str = 'INFO',
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 3,
    base_dir: Optional[str] = None,
    pid_suffix: str = '',
) -> logging.Logger:
    """创建（或复用）带旋转文件与控制台的 ``Logger``。

    Args:
        dir_name: 日志目录与记录器名称（位于 ``<repo>/logs/dir_name`` 下）。
        level: 控制台输出的最低级别；文件始终记录 DEBUG。
        max_bytes: 单个日志文件最大字节数，超出后自动轮转。
        backup_count: 保留的历史日志文件数量。
        base_dir: 日志根目录；``None`` 表示 ``<repo>/logs``。
        pid_suffix: 文件名进程后缀，``Log`` 在子进程里传 ``.<pid>`` 避免共享轮转。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    path = _resolve_log_path(base_dir, dir_name, pid_suffix)
    console_level = getattr(logging, str(level).upper(), logging.INFO)

    logger = logging.getLogger(dir_name)
    # 文件后缀会随进程变化，签名不一致时必须重建 handler，否则子进程会写父进程的文件。
    signature = (path, console_level, os.getpid())
    if getattr(logger, '_rl_signature', None) != signature:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - 关闭旧 handler 失败不应影响日志初始化
                pass

        fmt = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s:%(processName)s:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(console_level)
        logger.addHandler(console_handler)

        # 让 logger 自身放行全部记录，级别过滤交给各 handler。
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        # 私有标记用 setattr 设置，避免 mypy 认为 Logger 没有该属性。
        setattr(logger, '_rl_signature', signature)

    return logger


class Log:
    """结构化日志封装：限流、异常折叠与退出汇总。

    - 五级日志：``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``。
    - ``exception`` 按「类型 + 栈帧位置 + 归一化消息」折叠：首次完整 traceback，
      之后每 ``exc_report_secs`` 秒最多一行摘要，其余计数进入 :meth:`summary`。
    - ``info_throttled`` / ``info_every`` / ``once`` 用于循环内的重复消息。
    - 保留 ``log_info`` / ``log_exception`` 兼容旧代码。
    """

    def __init__(
        self,
        dir_name: str,
        level: str = 'INFO',
        base_dir: Optional[str] = None,
        throttle_secs: Optional[float] = None,
        exc_report_secs: float = _EXC_REPORT_SECS,
    ):
        """初始化日志包装器。

        Args:
            dir_name: 日志目录与记录器名称。
            level: 控制台输出的最低级别（文件始终 DEBUG）。
            base_dir: 日志根目录；``None`` 表示 ``<repo>/logs``。
            throttle_secs: ``info_throttled`` 的默认最小间隔秒数。
            exc_report_secs: 重复异常摘要的最小输出间隔秒数。
        """
        self.dir_name = dir_name
        self.level = level
        self.base_dir = base_dir
        self.throttle_secs = _DEFAULT_THROTTLE_SECS if throttle_secs is None else float(throttle_secs)
        self.exc_report_secs = float(exc_report_secs)
        self._owner_pid = os.getpid()

        self.logger = setup_logging(dir_name, level, base_dir=base_dir)
        self._start_ts = time.time()
        # 异常折叠状态: key = (type, origin, normalized value) -> dict
        self._exc_state: Dict[tuple, Dict[str, float]] = {}
        # 普通消息限流状态: key -> dict(first, last_emit, total, suppressed)
        self._throttle_state: Dict[str, Dict[str, float]] = {}

    # -- pickle 支持：worker 进程重建 handler -------------------------------

    def __getstate__(self):
        """``spawn`` 子进程只带配置过河，不传文件句柄。"""
        return {
            'dir_name': self.dir_name,
            'level': self.level,
            'base_dir': self.base_dir,
            'throttle_secs': self.throttle_secs,
            'exc_report_secs': self.exc_report_secs,
            'owner_pid': self._owner_pid,
        }

    def __setstate__(self, state):
        """在子进程中重建 handler：写独立的 ``<name>.<pid>.log``，避免共享轮转。"""
        owner_pid = state.get('owner_pid')
        self.__init__(
            state['dir_name'],
            state.get('level', 'INFO'),
            base_dir=state.get('base_dir'),
            throttle_secs=state.get('throttle_secs'),
            exc_report_secs=state.get('exc_report_secs', _EXC_REPORT_SECS),
        )
        if owner_pid != os.getpid():
            self.logger = setup_logging(
                state['dir_name'], state.get('level', 'INFO'),
                base_dir=state.get('base_dir'), pid_suffix=f'.{os.getpid()}',
            )

    @classmethod
    def from_config(cls, config: Optional[Dict], dir_name: str) -> 'Log':
        """按 YAML ``logging`` 段构造日志（``level`` / ``dir`` / ``throttle_secs``）。

        Args:
            config: 完整配置字典，可为 ``None``。
            dir_name: 日志目录与记录器名称。

        Returns:
            ``Log`` 实例。
        """
        section = (config or {}).get('logging', {}) or {}
        return cls(
            dir_name,
            level=section.get('level', 'INFO'),
            base_dir=section.get('dir'),
            throttle_secs=section.get('throttle_secs'),
        )

    # -- 分级日志 ---------------------------------------------------------

    def debug(self, message: str):
        """记录 DEBUG 级别消息（仅文件，控制台默认不输出）。"""
        self.logger.debug(message)

    def info(self, message: str):
        """记录 INFO 级别消息。"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录 WARNING 级别消息。"""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """记录 ERROR 级别消息，可选附带当前异常栈。"""
        self.logger.error(message, exc_info=exc_info)

    # -- 限流接口 ---------------------------------------------------------

    def info_throttled(self, key: str, message: str, interval: Optional[float] = None) -> bool:
        """同一 ``key`` 在 ``interval`` 秒内只输出一次，其余计入被抑制计数。

        Args:
            key: 限流键（通常是消息类别，如 ``"pop_exps_timeout"``）。
            message: 日志正文；输出时会追加 ``suppressed=N``。
            interval: 最小间隔秒数，``None`` 用构造时的 ``throttle_secs``。

        Returns:
            是否真正输出了这一行。
        """
        return self._emit_throttled(logging.INFO, key, message, interval)

    def warning_throttled(self, key: str, message: str, interval: Optional[float] = None) -> bool:
        """``info_throttled`` 的 WARNING 版本。"""
        return self._emit_throttled(logging.WARNING, key, message, interval)

    def info_every(self, key: str, count: int, message: str) -> bool:
        """每累计 ``count`` 次调用输出一次（用于 "每 N 个版本" 类进度日志）。"""
        state = self._throttle_state.setdefault(
            key, {'first': time.time(), 'last_emit': 0.0, 'total': 0.0, 'suppressed': 0.0}
        )
        state['total'] += 1
        if count <= 0 or state['total'] % count != 0:
            state['suppressed'] += 1
            return False
        state['last_emit'] = time.time()
        state['suppressed'] = 0
        self.logger.info(message)
        return True

    def once(self, key: str, message: str) -> bool:
        """同一进程内同一 ``key`` 只输出一次。"""
        if key in self._throttle_state:
            self._throttle_state[key]['total'] += 1
            self._throttle_state[key]['suppressed'] += 1
            return False
        self._throttle_state[key] = {
            'first': time.time(), 'last_emit': time.time(), 'total': 1.0, 'suppressed': 0.0,
        }
        self.logger.info(message)
        return True

    def _emit_throttled(self, level: int, key: str, message: str, interval: Optional[float]) -> bool:
        """限流实现：首次立即输出，之后按间隔输出并带上被抑制条数。"""
        window = self.throttle_secs if interval is None else float(interval)
        now = time.time()
        state = self._throttle_state.get(key)
        if state is None:
            self._throttle_state[key] = {
                'first': now, 'last_emit': now, 'total': 1.0, 'suppressed': 0.0,
            }
            self.logger.log(level, message)
            return True

        state['total'] += 1
        if now - state['last_emit'] < window:
            state['suppressed'] += 1
            return False
        suppressed = int(state['suppressed'])
        state['last_emit'] = now
        state['suppressed'] = 0
        suffix = f" suppressed={suppressed}" if suppressed else ""
        self.logger.log(level, f"{message}{suffix}")
        return True

    # -- 异常 -------------------------------------------------------------

    def exception(self, message: Optional[str] = None):
        """记录异常并折叠重复 traceback。

        首次出现：完整 traceback。重复出现：最多每 ``exc_report_secs`` 秒一行摘要。
        累计次数通过 :meth:`summary` 汇总，避免"折叠即丢失"。

        Args:
            message: 可选上下文描述，建议写明出错阶段（如 ``sampler loop``）。
        """
        exc_type, exc_value, tb = sys.exc_info()
        msg = message or "exception"
        if exc_type is None:
            self.logger.error(f"{msg} (no active exception)")
            return

        origin = _exc_origin(tb)
        key = (exc_type.__name__, origin, _normalize(exc_value))
        now = time.time()
        state = self._exc_state.get(key)

        if state is None:
            self._exc_state[key] = {
                'count': 1.0, 'first': now, 'last': now,
                'reported': now, 'suppressed': 0.0,
            }
            self.logger.error(
                f"{msg} type={exc_type.__name__} origin={origin} "
                f"value={_normalize(exc_value)}\n{traceback.format_exc()}"
            )
            return

        state['count'] += 1
        state['last'] = now
        state['suppressed'] += 1
        if now - state['reported'] < self.exc_report_secs:
            return

        suppressed = int(state['suppressed'])
        state['reported'] = now
        state['suppressed'] = 0
        self.logger.error(
            f"{msg} type={exc_type.__name__} origin={origin} "
            f"value={_normalize(exc_value)} repeated={int(state['count'])} "
            f"suppressed={suppressed} first={_hms(state['first'])} last={_hms(state['last'])}"
        )

    # -- 汇总 -------------------------------------------------------------

    def summary(self, extra: Optional[Dict[str, Any]] = None):
        """输出折叠统计（异常次数 + 被抑制的限流消息数），供进程退出前调用。

        Args:
            extra: 额外需要写进汇总行的键值对（如版本数、耗时）。
        """
        duration = time.time() - self._start_ts
        parts = [f"duration={duration:.0f}s"]
        for name, value in (extra or {}).items():
            parts.append(f"{name}={value}")
        if self._exc_state:
            total = sum(int(state['count']) for state in self._exc_state.values())
            parts.append(f"exceptions={total} kinds={len(self._exc_state)}")
        if self._throttle_state:
            suppressed = sum(int(state['suppressed']) for state in self._throttle_state.values())
            parts.append(f"throttled_keys={len(self._throttle_state)} suppressed={suppressed}")
        self.logger.info("log summary " + " ".join(p for p in parts if p))

        for (exc_name, origin, value), state in self._exc_state.items():
            self.logger.info(
                f"log summary exception type={exc_name} origin={origin} value={value} "
                f"count={int(state['count'])} first={_hms(state['first'])} last={_hms(state['last'])}"
            )

    def close(self):
        """关闭本进程创建的 handler（子进程退出前调用，避免句柄泄漏）。"""
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响退出
                pass
        if getattr(self.logger, '_rl_signature', None) is not None:
            self.logger._rl_signature = None

    # -- 兼容旧接口 -------------------------------------------------------

    def log_info(self, message: str, **_kwargs):
        """旧版兼容：同 ``info``。"""
        self.info(message)

    def log_exception(self, message: Optional[str] = None, **_kwargs):
        """旧版兼容：同 ``exception``，自动捕获并折叠 traceback。"""
        self.exception(message)


def _hms(timestamp: float) -> str:
    """把时间戳格式化为 ``HH:MM:SS``。"""
    return time.strftime('%H:%M:%S', time.localtime(timestamp))

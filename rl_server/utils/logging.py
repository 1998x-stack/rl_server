# -*- coding: utf-8 -*-
"""结构化日志：按进程名写入旋转文件并可选输出到控制台。

接口兼容旧版 ``libs/log.py`` 的 ``log_info`` / ``log_exception``，
同时提供标准 ``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``。

重复异常自动折叠，避免循环中的异常刷爆日志文件。
"""
import os
import sys
import time
import logging
import logging.handlers
from typing import Optional


def setup_logging(
    dir_name: str,
    level: str = 'INFO',
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 3,
) -> logging.Logger:
    """创建带旋转文件与控制台的 ``Logger``。

    Args:
        dir_name: 日志子目录名（位于项目 ``logs/dir_name`` 下）。
        level: 日志级别名称，如 ``INFO``、``DEBUG``。
        max_bytes: 单个日志文件最大字节数，超出后自动轮转。
        backup_count: 保留的历史日志文件数量。

    Returns:
        配置好的 ``logging.Logger`` 实例。
    """
    log_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', '..', 'logs', dir_name
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(dir_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f'{dir_name}.log'),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(ch)

    return logger


class Log:
    """结构化日志封装，内置异常去重。

    - 五级日志：``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``。
    - ``exception`` 自动捕获异常栈并去重：相同 traceback 首次完整输出，
      之后折叠为 ``(repeated N times, last at HH:MM:SS)``，每 60 次
      或每 5 分钟输出一次完整 traceback。
    - 文件自动轮转（默认 50 MB，保留 3 个历史文件）。
    - 保留 ``log_info`` / ``log_exception`` 兼容旧代码。
    """

    _DEDUP_THROTTLE_COUNT = 60   # 每 N 次重复输出一次完整 traceback
    _DEDUP_THROTTLE_SECS = 300   # 每 M 秒至少输出一次完整 traceback

    def __init__(self, dir_name: str, level: str = 'INFO'):
        """初始化日志包装器。

        Args:
            dir_name: 日志目录与记录器名称。
            level: 控制台输出的最低级别（文件始终 DEBUG）。
        """
        self.logger = setup_logging(dir_name, level)
        # 异常去重状态: key = hash(type+first_line), value = (count, first_timestamp, last_timestamp)
        self._exc_state: dict[str, tuple[int, float, float]] = {}

    # -- 新接口 -----------------------------------------------------------

    def debug(self, message: str):
        """记录 DEBUG 级别消息（仅文件，默认不输出到控制台）。"""
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

    def exception(self, message: Optional[str] = None):
        """记录异常并自动折叠重复 traceback。

        首次出现：完整 traceback + 上下文。
        重复出现：折叠为单行计数，每 ``_DEDUP_THROTTLE_COUNT`` 次或
        每 ``_DEDUP_THROTTLE_SECS`` 秒展开一次完整 traceback。

        Args:
            message: 可选上下文描述。
        """
        import traceback as _tb

        msg = message or "Exception occurred"
        exc_type, exc_value, _ = sys.exc_info()
        tb_lines = _tb.format_exc()

        if exc_type is None:
            self.logger.error(f"{msg}\n{tb_lines}")
            return

        # 用异常类型 + 第一帧位置作为去重 key
        key = f"{exc_type.__name__}:{exc_value}"
        now = time.time()

        if key in self._exc_state:
            count, first_ts, last_ts = self._exc_state[key]
            count += 1
            self._exc_state[key] = (count, first_ts, now)

            throttled = (count % self._DEDUP_THROTTLE_COUNT != 0
                         and (now - last_ts) < self._DEDUP_THROTTLE_SECS)
            if throttled:
                return  # 静默跳过，不写日志

            # 每 N 次或超时时展开一次
            elapsed = now - first_ts
            self.logger.error(
                f"{msg} (repeated {count} times over {elapsed:.0f}s, "
                f"last at {time.strftime('%H:%M:%S', time.localtime(now))})\n{tb_lines}"
            )
        else:
            self._exc_state[key] = (1, now, now)
            self.logger.error(f"{msg}\n{tb_lines}")

    # -- 兼容旧接口 -------------------------------------------------------

    def log_info(self, message: str, **_kwargs):
        """旧版兼容：同 ``info``。"""
        self.info(message)

    def log_exception(self, **_kwargs):
        """旧版兼容：同 ``exception``，自动捕获并去重 traceback。"""
        self.exception()

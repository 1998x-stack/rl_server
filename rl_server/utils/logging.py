# -*- coding: utf-8 -*-
"""结构化日志：按进程名写入文件并可选输出到控制台。

接口兼容旧版 ``libs/log.py`` 的 ``log_info`` / ``log_exception``，
同时提供标准 ``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``。
"""
import os
import logging
import sys
from typing import Optional


def setup_logging(dir_name: str, level: str = 'INFO') -> logging.Logger:
    """创建带文件与控制台的 ``Logger``。

    Args:
        dir_name: 日志子目录名（位于项目 ``logs/dir_name`` 下）。
        level: 日志级别名称，如 ``INFO``、``DEBUG``。

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
        fh = logging.FileHandler(
            os.path.join(log_dir, f'{dir_name}.log'), encoding='utf-8'
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)  # file always gets DEBUG; console obeys configured level
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(ch)

    return logger


class Log:
    """结构化日志封装。

    提供五级日志：``debug`` / ``info`` / ``warning`` / ``error`` / ``exception``。
    ``exception`` 自动捕获当前异常栈（仅 ``except`` 块内有效）。
    文件始终写 DEBUG 级别，控制台级别由初始化时的 ``level`` 控制。

    保留 ``log_info`` / ``log_exception`` 方法以兼容旧代码。
    """

    def __init__(self, dir_name: str, level: str = 'INFO'):
        """初始化日志包装器。

        Args:
            dir_name: 日志目录与记录器名称。
            level: 控制台输出的最低级别（文件始终 DEBUG）。
        """
        self.logger = setup_logging(dir_name, level)

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
        """记录 ERROR 级别消息，可选附带当前异常栈。

        Args:
            message: 错误描述。
            exc_info: 若为 True，附加 ``sys.exc_info()`` 的完整 traceback。
        """
        self.logger.error(message, exc_info=exc_info)

    def exception(self, message: Optional[str] = None):
        """记录 ERROR 级别消息并自动附带完整 traceback。

        必须在 ``except`` 块内调用，否则 traceback 为 ``NoneType: None``。

        Args:
            message: 可选上下文描述；为 None 时仅输出 traceback。
        """
        self.logger.exception(message or "Exception occurred")

    # -- 兼容旧接口 -------------------------------------------------------

    def log_info(self, message: str, **_kwargs):
        """旧版兼容：同 ``info``。"""
        self.info(message)

    def log_exception(self, **_kwargs):
        """旧版兼容：同 ``exception``，自动捕获 traceback。"""
        self.exception()

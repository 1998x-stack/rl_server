# -*- coding: utf-8 -*-
"""结构化日志：按目录名落盘并输出到控制台，兼容旧版 ``Log`` 接口。"""
import os
import logging
import sys


def setup_logging(dir_name: str, level: str = 'INFO') -> logging.Logger:
    """创建同时写入文件与标准输出的日志记录器。

    Args:
        dir_name: 日志子目录名，日志文件位于 ``logs/{dir_name}/{dir_name}.log``。
        level: 日志级别名称，如 ``INFO``、``DEBUG``。

    Returns:
        配置完成的 ``logging.Logger``。
    """
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs', dir_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(dir_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh = logging.FileHandler(
            os.path.join(log_dir, f'{dir_name}.log'),
            encoding='utf-8'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


class Log:
    """对 ``setup_logging`` 的薄封装，提供 ``log_info`` / ``log_exception``。"""

    def __init__(self, dir_name: str):
        """初始化日志包装器。

        Args:
            dir_name: 日志目录名。
        """
        self.logger = setup_logging(dir_name)

    def log_info(self, message: str, print_screen: bool = False):
        """写入 INFO 级别消息（``print_screen`` 仅保留兼容，行为与文件一致）。"""
        self.logger.info(message)

    def log_exception(self, print_screen: bool = False):
        """记录当前异常栈。"""
        self.logger.exception("Exception occurred")

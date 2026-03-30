# -*- coding: utf-8 -*-
"""
Structured logging with worker identity.
Backwards-compatible: Log class interface preserved.
"""
import os
import logging
import sys


def setup_logging(dir_name: str, level: str = 'INFO') -> logging.Logger:
    """Create a logger with both file and console handlers."""
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs', dir_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(dir_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # File handler
        fh = logging.FileHandler(
            os.path.join(log_dir, f'{dir_name}.log'),
            encoding='utf-8'
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


class Log:
    """Backwards-compatible wrapper around Python logging."""

    def __init__(self, dir_name: str):
        self.logger = setup_logging(dir_name)

    def log_info(self, message: str, print_screen: bool = False):
        self.logger.info(message)

    def log_exception(self, print_screen: bool = False):
        self.logger.exception("Exception occurred")

# -*- coding: utf-8 -*-
"""YAML 配置加载器：支持覆盖文件合并与环境变量插值。
"""
import os
import re
import copy
import yaml
from typing import Any, Dict, Optional


_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def interpolate_env_vars(value: str) -> str:
    """将字符串中的 ``${VAR}`` 或 ``${VAR:default}`` 替换为环境变量值。

    Args:
        value: 待处理的字符串；若非 ``str`` 类型则原样返回。

    Returns:
        替换后的字符串；若 ``value`` 非字符串则返回 ``value`` 本身。
    """
    if not isinstance(value, str):
        return value

    def _replace(match):
        expr = match.group(1)
        if ':' in expr:
            var_name, default = expr.split(':', 1)
        else:
            var_name, default = expr, ''
        return os.environ.get(var_name, default)

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _interpolate_recursive(obj: Any) -> Any:
    """在嵌套的字典与列表结构中递归执行环境变量插值。

    Args:
        obj: 任意 YAML 解析后的 Python 对象。

    Returns:
        所有字符串叶子节点已插值后的深拷贝结构。
    """
    if isinstance(obj, str):
        return interpolate_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_recursive(item) for item in obj]
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典：``override`` 中的值覆盖 ``base``。

    当两处的值均为字典时，递归合并子字典；否则以 ``override`` 为准。

    Args:
        base: 基础配置。
        override: 覆盖配置。

    Returns:
        新的合并后字典（不修改传入的 ``base``）。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str, override_path: Optional[str] = None) -> Dict:
    """加载 YAML 配置文件，可选合并覆盖文件，并对结果做环境变量插值。

    Args:
        base_path: 主配置文件路径。
        override_path: 可选的覆盖配置文件路径；若存在则与主配置深度合并。

    Returns:
        合并且插值后的配置字典。

    Raises:
        FileNotFoundError: 当 ``base_path`` 不存在时。
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Config file not found: {base_path}")

    with open(base_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    if override_path and os.path.exists(override_path):
        with open(override_path, 'r') as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    return _interpolate_recursive(config)

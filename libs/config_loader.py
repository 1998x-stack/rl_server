# -*- coding: utf-8 -*-
"""YAML 配置加载：支持覆盖文件深度合并与环境变量 ``${VAR:default}`` 插值。"""
import os
import re
import copy
import yaml
from typing import Any, Dict, Optional


_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def interpolate_env_vars(value: str) -> str:
    """将字符串中的 ``${VAR}`` / ``${VAR:default}`` 替换为环境变量值。

    Args:
        value: 输入字符串；非字符串类型原样返回。

    Returns:
        替换后的字符串或原 ``value``。
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
    """递归处理字典与列表中的字符串叶子节点。"""
    if isinstance(obj, str):
        return interpolate_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_recursive(item) for item in obj]
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并：``override`` 覆盖 ``base``，子字典递归合并。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str, override_path: Optional[str] = None) -> Dict:
    """加载主 YAML，可选合并覆盖文件，并对结果做环境变量插值。

    Args:
        base_path: 主配置文件路径。
        override_path: 可选覆盖文件路径。

    Returns:
        最终配置字典。

    Raises:
        FileNotFoundError: 主文件不存在时。
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

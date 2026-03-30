# -*- coding: utf-8 -*-
"""
YAML configuration loader with environment variable interpolation.
Copied from libs/config_loader.py.
"""
import os
import re
import copy
import yaml
from typing import Any, Dict, Optional


_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def interpolate_env_vars(value: str) -> str:
    """Replace ${VAR:default} patterns with environment variable values."""
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
    """Recursively interpolate env vars in a nested dict/list structure."""
    if isinstance(obj, str):
        return interpolate_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_recursive(item) for item in obj]
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(base_path: str, override_path: Optional[str] = None) -> Dict:
    """Load YAML config with optional override file and env var interpolation.

    Args:
        base_path: Path to the base YAML config file.
        override_path: Optional path to an override YAML file.

    Returns:
        Merged and interpolated configuration dictionary.

    Raises:
        FileNotFoundError: If base_path does not exist.
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

# -*- coding: utf-8 -*-
"""配置结构校验：确保运行所需的关键节与键存在。"""
from typing import Dict

# 各配置节必须包含的键集合（用于启动前快速失败）。
REQUIRED_KEYS = {
    'redis': {'model', 'exps'},
    'training': {'env_name', 'num_samplers', 'num_trainers'},
}


def validate_config(config: Dict) -> None:
    """校验配置字典是否包含必需的节与键。

    Args:
        config: 已加载并合并后的配置字典。

    Raises:
        ValueError: 当缺少某个必需节，或节内缺少某个必需键时。
    """
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: {section}.{key}")

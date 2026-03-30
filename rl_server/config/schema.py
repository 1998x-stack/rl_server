# -*- coding: utf-8 -*-
"""Configuration validation."""
from typing import Dict

REQUIRED_KEYS = {
    'redis': {'model', 'exps'},
    'training': {'env_name', 'num_samplers', 'num_trainers'},
}


def validate_config(config: Dict) -> None:
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing config key: {section}.{key}")

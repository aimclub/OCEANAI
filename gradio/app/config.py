"""
File: config.py
Author: Elena Ryumina and Dmitry Ryumin
Description: Configuration module for handling settings.
License: MIT License
"""

import toml
from typing import Callable, Dict
from types import SimpleNamespace

CONFIG_NAME = "config.toml"


def flatten_dict(prefix: str, d: Dict) -> Dict:
    result = {}

    for k, v in d.items():
        result.update(flatten_dict(f"{prefix}{k}_", v) if isinstance(v, dict) else {f"{prefix}{k}": v})

    return result


def load_tab_creators(file_path: str, available_functions: Callable) -> Dict:
    config = toml.load(file_path)
    tab_creators_data = config.get("TabCreators", {})

    return {key: available_functions[value] for key, value in tab_creators_data.items()}


def load_config(file_path: str) -> SimpleNamespace:
    config = toml.load(file_path)
    config_data = flatten_dict("", config)

    return SimpleNamespace(**config_data)


config_data = load_config(CONFIG_NAME)

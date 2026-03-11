from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_SECRETS_PATH = PROJECT_ROOT / "configs" / "secrets.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def deep_merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Optional[str] = None,
    secrets_path: Optional[str] = None,
) -> Dict[str, Any]:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    secrets_file = Path(secrets_path) if secrets_path else DEFAULT_SECRETS_PATH

    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file

    if not secrets_file.is_absolute():
        secrets_file = PROJECT_ROOT / secrets_file

    config = load_yaml(config_file)

    if secrets_file.exists():
        secrets = load_yaml(secrets_file)
        config = deep_merge_dict(config, secrets)

    return config
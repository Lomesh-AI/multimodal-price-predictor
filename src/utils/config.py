"""YAML config loading and validation.

Every script in scripts/ loads its run configuration through load_config(),
never by hardcoding values, so every run is reproducible from its config file.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.exceptions import ConfigError
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_TOP_LEVEL_KEYS = ["seed", "data", "encoders", "fusion", "head", "training"]


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Raises:
        ConfigError: if the file is missing, malformed, or missing required keys.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config {config_path}: {e}") from e

    if config is None:
        raise ConfigError(f"Config file {config_path} is empty")

    if not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} must define a mapping at the top level")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ConfigError(
            f"Config {config_path} is missing required top-level keys: {missing}"
        )

    _validate_encoders(config["encoders"], config_path)
    _validate_fusion(config["fusion"], config_path)
    _validate_training(config["training"], config_path)

    logger.info("Loaded config from %s", config_path)
    return config


def _validate_encoders(encoders_cfg: Dict[str, Any], config_path: Path) -> None:
    for modality in ("text", "image"):
        if modality not in encoders_cfg:
            raise ConfigError(f"Config {config_path} missing encoders.{modality}")
        block = encoders_cfg[modality]
        for key in ("name", "output_dim"):
            if key not in block:
                raise ConfigError(f"Config {config_path} missing encoders.{modality}.{key}")
        if not isinstance(block["output_dim"], int) or block["output_dim"] <= 0:
            raise ConfigError(
                f"encoders.{modality}.output_dim must be a positive int, got {block['output_dim']}"
            )


def _validate_fusion(fusion_cfg: Dict[str, Any], config_path: Path) -> None:
    if "type" not in fusion_cfg:
        raise ConfigError(f"Config {config_path} missing fusion.type")
    valid_types = {"cross_attention", "concat", "gated"}
    if fusion_cfg["type"] not in valid_types:
        raise ConfigError(
            f"fusion.type must be one of {valid_types}, got {fusion_cfg['type']!r}"
        )


def _validate_training(training_cfg: Dict[str, Any], config_path: Path) -> None:
    required = ["loss", "lr", "batch_size", "epochs"]
    missing = [k for k in required if k not in training_cfg]
    if missing:
        raise ConfigError(f"Config {config_path} missing training keys: {missing}")
    if training_cfg["batch_size"] <= 0:
        raise ConfigError("training.batch_size must be positive")
    if training_cfg["epochs"] <= 0:
        raise ConfigError("training.epochs must be positive")

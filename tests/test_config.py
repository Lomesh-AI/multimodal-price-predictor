import textwrap

import pytest

from src.utils.config import load_config
from src.utils.exceptions import ConfigError

VALID_CONFIG = """
seed: 42
data:
  train_csv: data/raw/train.csv
  test_csv: data/raw/test.csv
encoders:
  text:
    name: embeddinggemma
    output_dim: 768
  image:
    name: siglip2-giant
    output_dim: 1792
fusion:
  type: cross_attention
  proj_dim: 512
head:
  hidden_dims: [512, 256, 128]
training:
  loss: smape
  lr: 0.0001
  batch_size: 64
  epochs: 25
"""


def test_load_valid_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_CONFIG)
    config = load_config(str(config_path))
    assert config["seed"] == 42
    assert config["fusion"]["type"] == "cross_attention"


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("does/not/exist.yaml")


def test_missing_required_key_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n")
    with pytest.raises(ConfigError):
        load_config(str(config_path))


def test_invalid_fusion_type_raises(tmp_path):
    bad_config = VALID_CONFIG.replace("cross_attention", "not_a_fusion")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(bad_config)
    with pytest.raises(ConfigError):
        load_config(str(config_path))


def test_empty_file_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    with pytest.raises(ConfigError):
        load_config(str(config_path))

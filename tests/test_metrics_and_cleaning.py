import numpy as np
import pytest

from src.data.text_cleaning import clean_text
from src.evaluation.metrics import mae, smape
from src.utils.exceptions import ConfigError, DataError


def test_smape_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert smape(y, y) == pytest.approx(0.0, abs=1e-9)


def test_smape_shape_mismatch_raises():
    with pytest.raises(DataError):
        smape(np.array([1.0, 2.0]), np.array([1.0]))


def test_mae_basic():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_clean_text_strips_html_entities():
    assert clean_text("Wireless &amp; Bluetooth") == "Wireless & Bluetooth"


def test_clean_text_strips_prefixes():
    assert clean_text("Bullet Point 1: Long battery life") == "Long battery life"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_handles_nan():
    assert clean_text(float("nan")) == ""

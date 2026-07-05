"""Evaluation metrics, computed with numpy so they don't require torch."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.exceptions import DataError


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise DataError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    numerator = 2.0 * np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(numerator / denominator))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise DataError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))


def per_category_smape(
    df: pd.DataFrame,
    true_col: str = "price",
    pred_col: str = "predicted_price",
    category_col: Optional[str] = None,
) -> Dict[str, float]:
    """Breaks SMAPE down by category if a category column is available.
    Falls back to a single overall score if not — never raises just because
    the dataset lacks category labels."""
    if true_col not in df.columns or pred_col not in df.columns:
        raise DataError(f"DataFrame missing '{true_col}' or '{pred_col}' column")

    if category_col is None or category_col not in df.columns:
        return {"overall": smape(df[true_col].values, df[pred_col].values)}

    results = {"overall": smape(df[true_col].values, df[pred_col].values)}
    for cat, group in df.groupby(category_col):
        if len(group) > 0:
            results[str(cat)] = smape(group[true_col].values, group[pred_col].values)
    return results

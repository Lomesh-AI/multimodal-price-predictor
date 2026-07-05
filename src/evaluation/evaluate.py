"""Runs a trained PriceModel over a dataloader and reports metrics."""

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import mae, smape
from src.utils.exceptions import CheckpointError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.to(device)
    model.eval()

    all_true, all_pred = [], []
    for batch in loader:
        text_emb, image_emb, price = [t.to(device) for t in batch]
        pred = model(text_emb, image_emb)
        all_true.append(price.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    if not all_true:
        raise CheckpointError("Evaluation dataloader produced zero batches")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }
    logger.info("Evaluation: SMAPE=%.4f MAE=%.4f n=%d", metrics["smape"], metrics["mae"], metrics["n_samples"])
    return metrics

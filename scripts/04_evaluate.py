#!/usr/bin/env python
"""Stage 04 — evaluate.

Loads the best checkpoint from stage 03 and reports SMAPE/MAE on a held-out
split of the cached training embeddings. Writes reports/<run>/eval_report.json.

Usage:
    python scripts/04_evaluate.py --config configs/base.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import EmbeddingDataset
from src.evaluation.evaluate import evaluate_model
from src.models.price_model import PriceModel
from src.utils.config import load_config
from src.utils.exceptions import CheckpointError, PricePredictorError
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run(config_path: str) -> dict:
    config = load_config(config_path)
    set_seed(config["seed"])

    embeddings_dir = Path(config["data"]["embeddings_dir"])
    prices = np.load(embeddings_dir / "train_price.npy")
    full_dataset = EmbeddingDataset(
        text_embeddings_path=str(embeddings_dir / "train_text.npy"),
        image_embeddings_path=str(embeddings_dir / "train_image.npy"),
        prices=prices,
    )

    val_split = config["training"].get("val_split", 0.15)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(config["seed"])
    _, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = PriceModel.from_config(config)
    checkpoint_path = Path(config["checkpoint_dir"]) / "best.pt"
    if not checkpoint_path.exists():
        raise CheckpointError(f"No checkpoint found at {checkpoint_path} — run scripts/03_train.py first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate_model(model, val_loader, device)

    report_dir = Path("reports") / Path(config["checkpoint_dir"]).name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "eval_report.json"
    with report_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote evaluation report to %s", report_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 04: evaluate a trained checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        run(args.config)
    except PricePredictorError as e:
        logger.error("Evaluation failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during evaluation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

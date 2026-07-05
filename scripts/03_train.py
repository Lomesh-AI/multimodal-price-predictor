#!/usr/bin/env python
"""Stage 03 — train.

Trains only the projections + fusion + regression head on cached embeddings
(stage 02 output). Backbones are never loaded here.

Usage:
    python scripts/03_train.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import EmbeddingDataset
from src.models.price_model import PriceModel
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.exceptions import PricePredictorError
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    data_cfg = config["data"]
    embeddings_dir = Path(data_cfg["embeddings_dir"])
    price_path = embeddings_dir / "train_price.npy"

    if not price_path.exists():
        raise PricePredictorError(
            f"{price_path} not found — run scripts/02_extract_embeddings.py first"
        )

    prices = np.load(price_path)
    full_dataset = EmbeddingDataset(
        text_embeddings_path=str(embeddings_dir / "train_text.npy"),
        image_embeddings_path=str(embeddings_dir / "train_image.npy"),
        prices=prices,
    )

    val_split = config["training"].get("val_split", 0.15)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    if n_train <= 0:
        raise PricePredictorError(
            f"val_split={val_split} leaves zero training rows for dataset of size {len(full_dataset)}"
        )

    generator = torch.Generator().manual_seed(config["seed"])
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PriceModel.from_config(config)
    trainer = Trainer(model, config["training"], checkpoint_dir=config["checkpoint_dir"])
    history = trainer.fit(train_loader, val_loader)

    logger.info("Training complete. Best val_smape=%.4f", trainer.best_val_loss)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 03: train fusion + head")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        run(args.config)
    except PricePredictorError as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during training: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

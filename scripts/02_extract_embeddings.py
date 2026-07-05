#!/usr/bin/env python
"""Stage 02 — extract embeddings.

The one GPU-heavy stage: runs the frozen EmbeddingGemma + SigLIP2 encoders
once over every row and caches the results as .npy files, so training
(stage 03) never has to re-run the encoders.

Usage:
    python scripts/02_extract_embeddings.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoders.registry import build_image_encoder, build_text_encoder
from src.utils.config import load_config
from src.utils.exceptions import PricePredictorError
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    embeddings_dir = Path(data_cfg["embeddings_dir"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    text_encoder = build_text_encoder(config["encoders"]["text"])
    image_encoder = build_image_encoder(config["encoders"]["image"])

    for split in ("train", "test"):
        parquet_path = processed_dir / f"{split}_clean.parquet"
        if not parquet_path.exists():
            raise PricePredictorError(
                f"{parquet_path} not found — run scripts/01_preprocess.py first"
            )
        df = pd.read_parquet(parquet_path)
        logger.info("=== Extracting embeddings for split: %s (%d rows) ===", split, len(df))

        text_vectors = []
        for batch in _batched(df["catalog_content"].tolist(), config["encoders"]["text"]["batch_size"]):
            text_vectors.append(text_encoder.encode(batch).numpy())
        text_matrix = np.concatenate(text_vectors, axis=0)

        image_vectors = []
        for batch_paths in _batched(df["image_path"].tolist(), config["encoders"]["image"]["batch_size"]):
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            image_vectors.append(image_encoder.encode(images).numpy())
        image_matrix = np.concatenate(image_vectors, axis=0)

        if text_matrix.shape[0] != len(df) or image_matrix.shape[0] != len(df):
            raise PricePredictorError(
                f"Embedding count mismatch for split={split}: "
                f"text={text_matrix.shape[0]} image={image_matrix.shape[0]} rows={len(df)}"
            )

        np.save(embeddings_dir / f"{split}_text.npy", text_matrix)
        np.save(embeddings_dir / f"{split}_image.npy", image_matrix)
        if split == "train":
            np.save(embeddings_dir / "train_price.npy", df["price"].to_numpy(dtype=np.float32))
        df[["sample_id"]].to_csv(embeddings_dir / f"{split}_ids.csv", index=False)

        logger.info(
            "Saved embeddings for %s: text %s, image %s",
            split, text_matrix.shape, image_matrix.shape,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 02: extract and cache embeddings")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        run(args.config)
    except PricePredictorError as e:
        logger.error("Embedding extraction failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during embedding extraction: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

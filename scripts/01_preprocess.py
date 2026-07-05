#!/usr/bin/env python
"""Stage 01 — preprocess.

Reads data/raw/{train,test}.csv, cleans text, downloads + resizes images.
Writes data/processed/{train,test}_clean.parquet and data/processed/images/.

Usage:
    python scripts/01_preprocess.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.image_pipeline import process_batch
from src.data.loader import load_test, load_train
from src.data.text_cleaning import clean_dataframe
from src.utils.config import load_config
from src.utils.exceptions import PricePredictorError
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split, loader_fn, csv_key in [("train", load_train, "train_csv"), ("test", load_test, "test_csv")]:
        logger.info("=== Preprocessing split: %s ===", split)
        df = loader_fn(data_cfg[csv_key])
        df = clean_dataframe(df, column="catalog_content")

        image_dir = processed_dir / "images" / split
        results = process_batch(
            urls=df["image_link"].tolist(),
            output_dir=str(image_dir),
            size=data_cfg["image_size"],
            max_workers=data_cfg.get("image_download_workers", 16),
        )
        df["image_path"] = [r["path"] for r in results]
        df["image_is_placeholder"] = [r["placeholder"] for r in results]

        out_path = processed_dir / f"{split}_clean.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 01: preprocess raw data")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        run(args.config)
    except PricePredictorError as e:
        logger.error("Preprocessing failed: %s", e)
        sys.exit(1)
    except Exception as e:  # unexpected — still fail loudly with context
        logger.exception("Unexpected error during preprocessing: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

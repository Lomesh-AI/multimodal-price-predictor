#!/usr/bin/env python
"""Stage 05 — predict.

Runs the full end-to-end chain (encoders + fusion + head) over the test
split's raw text/image inputs and writes a submission CSV.

Usage:
    python scripts/05_predict.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import Predictor
from src.utils.config import load_config
from src.utils.exceptions import PricePredictorError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run(config_path: str, output_path: str) -> None:
    config = load_config(config_path)

    processed_dir = Path(config["data"]["processed_dir"])
    test_path = processed_dir / "test_clean.parquet"
    if not test_path.exists():
        raise PricePredictorError(f"{test_path} not found — run scripts/01_preprocess.py first")

    df = pd.read_parquet(test_path)
    checkpoint_path = str(Path(config["checkpoint_dir"]) / "best.pt")
    predictor = Predictor(config, checkpoint_path)

    batch_size = 32
    all_prices = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        prices = predictor.predict_batch(
            chunk["catalog_content"].tolist(), chunk["image_path"].tolist()
        )
        all_prices.extend(prices)
        logger.info("Predicted %d/%d rows", min(start + batch_size, len(df)), len(df))

    out_df = pd.DataFrame({"sample_id": df["sample_id"], "price": all_prices})
    out_df.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s (%d rows)", output_path, len(out_df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05: predict on the test set")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="data/raw/test_out.csv")
    args = parser.parse_args()

    try:
        run(args.config, args.output)
    except PricePredictorError as e:
        logger.error("Prediction failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during prediction: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

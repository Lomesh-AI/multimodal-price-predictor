"""Load and schema-validate the raw competition CSVs."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.exceptions import DataError
from src.utils.logging import get_logger

logger = get_logger(__name__)

TRAIN_REQUIRED_COLUMNS = ["sample_id", "catalog_content", "image_link", "price"]
TEST_REQUIRED_COLUMNS = ["sample_id", "catalog_content", "image_link"]


def load_raw_csv(path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a CSV and validate it has the expected columns.

    Raises:
        DataError: if the file is missing, unreadable, empty, or missing columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise DataError(f"CSV is empty: {csv_path}") from e
    except pd.errors.ParserError as e:
        raise DataError(f"CSV could not be parsed: {csv_path}: {e}") from e
    except UnicodeDecodeError as e:
        raise DataError(
            f"CSV has an encoding issue, try re-saving as UTF-8: {csv_path}: {e}"
        ) from e

    if df.empty:
        raise DataError(f"CSV loaded but contains zero rows: {csv_path}")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataError(
                f"CSV {csv_path} is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    duplicate_ids = df["sample_id"].duplicated().sum() if "sample_id" in df.columns else 0
    if duplicate_ids > 0:
        logger.warning("%d duplicate sample_id values found in %s", duplicate_ids, csv_path)

    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def load_train(path: str) -> pd.DataFrame:
    df = load_raw_csv(path, required_columns=TRAIN_REQUIRED_COLUMNS)
    n_bad_price = df["price"].isna().sum() + (df["price"] <= 0).sum()
    if n_bad_price > 0:
        raise DataError(
            f"{n_bad_price} rows in {path} have missing or non-positive price values"
        )
    return df


def load_test(path: str) -> pd.DataFrame:
    return load_raw_csv(path, required_columns=TEST_REQUIRED_COLUMNS)

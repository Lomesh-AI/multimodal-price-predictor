"""Text cleaning for the catalog_content field.

Product descriptions are semi-structured (HTML entities, "Bullet Point N:"
prefixes, encoding artifacts). Numeric patterns like "16 GB" or "2.5 kg" are
strong price signals and are deliberately preserved, not stripped.
"""

import html
import re
from typing import Optional

import pandas as pd

from src.utils.exceptions import DataError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PREFIX_PATTERN = re.compile(
    r"(Bullet Point \d+:|Item Name:|Value:|Unit:|Product Description:)",
    flags=re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MOJIBAKE_PATTERN = re.compile(r"[â€™â€“â€œâ€]")


def clean_text(text: Optional[str]) -> str:
    """Clean a single catalog_content string. Never raises — missing/bad
    input becomes an empty string so downstream batching never breaks on
    one malformed row.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = _MOJIBAKE_PATTERN.sub("", text)
    text = _PREFIX_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame, column: str = "catalog_content") -> pd.DataFrame:
    """Apply clean_text to a column, returning a new DataFrame.

    Raises:
        DataError: if the column does not exist.
    """
    if column not in df.columns:
        raise DataError(f"Column '{column}' not found in DataFrame. Columns: {list(df.columns)}")

    out = df.copy()
    out[column] = out[column].apply(clean_text)

    n_empty = (out[column] == "").sum()
    if n_empty > 0:
        logger.warning("%d rows have empty text after cleaning", n_empty)

    return out

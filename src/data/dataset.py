"""Dataset classes.

EmbeddingDataset wraps precomputed .npy embeddings for fast training (stage 03
reads these, never re-runs the encoders). RawInferenceDataset wraps raw
text/image inputs for the live predictor, which does need the encoders.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.exceptions import DataError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingDataset(Dataset):
    """Wraps cached text/image embeddings plus optional price labels."""

    def __init__(
        self,
        text_embeddings_path: str,
        image_embeddings_path: str,
        prices: Optional[np.ndarray] = None,
    ):
        self.text_embeddings = self._load_npy(text_embeddings_path)
        self.image_embeddings = self._load_npy(image_embeddings_path)

        if self.text_embeddings.shape[0] != self.image_embeddings.shape[0]:
            raise DataError(
                "Text and image embedding counts do not match: "
                f"{self.text_embeddings.shape[0]} vs {self.image_embeddings.shape[0]}"
            )

        if prices is not None:
            prices = np.asarray(prices, dtype=np.float32)
            if len(prices) != self.text_embeddings.shape[0]:
                raise DataError(
                    f"Price count ({len(prices)}) does not match embedding count "
                    f"({self.text_embeddings.shape[0]})"
                )
        self.prices = prices

        logger.info(
            "Loaded EmbeddingDataset: %d rows, text_dim=%d, image_dim=%d, labeled=%s",
            self.text_embeddings.shape[0],
            self.text_embeddings.shape[1],
            self.image_embeddings.shape[1],
            self.prices is not None,
        )

    @staticmethod
    def _load_npy(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise DataError(f"Embedding file not found: {p}")
        try:
            arr = np.load(p)
        except (ValueError, OSError) as e:
            raise DataError(f"Failed to load embedding file {p}: {e}") from e
        if arr.ndim != 2:
            raise DataError(f"Expected a 2D embedding array in {p}, got shape {arr.shape}")
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return self.text_embeddings.shape[0]

    def __getitem__(self, idx: int):
        text_vec = torch.from_numpy(self.text_embeddings[idx])
        image_vec = torch.from_numpy(self.image_embeddings[idx])
        if self.prices is not None:
            price = torch.tensor(self.prices[idx], dtype=torch.float32)
            return text_vec, image_vec, price
        return text_vec, image_vec


class RawInferenceDataset(Dataset):
    """Wraps raw (uncleaned) text + image path/url pairs for live inference,
    where embeddings have not been precomputed."""

    def __init__(self, texts: List[str], image_sources: List[str]):
        if len(texts) != len(image_sources):
            raise DataError(
                f"texts ({len(texts)}) and image_sources ({len(image_sources)}) "
                "must be the same length"
            )
        self.texts = texts
        self.image_sources = image_sources

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.image_sources[idx]

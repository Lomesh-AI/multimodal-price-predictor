"""EmbeddingGemma text encoder (frozen, 308M params, Gemma-3-based).

Loaded lazily so importing this module never requires the model weights or
a GPU to be present — only calling .encode() does. This matters for unit
tests and for scripts that only need the class registered, not instantiated.
"""

from typing import List, Optional

import torch

from src.encoders.base import TextEncoder
from src.utils.exceptions import EncoderError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_MRL_DIMS = {128, 256, 512, 768}


class EmbeddingGemmaEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: Optional[str] = None,
        mrl_truncate: Optional[int] = None,
        batch_size: int = 64,
    ):
        if mrl_truncate is not None and mrl_truncate not in _VALID_MRL_DIMS:
            raise EncoderError(
                f"mrl_truncate must be one of {_VALID_MRL_DIMS} or None, got {mrl_truncate}"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mrl_truncate = mrl_truncate
        self.batch_size = batch_size
        self.output_dim = mrl_truncate or 768
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EncoderError(
                "sentence-transformers is required for EmbeddingGemmaEncoder. "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:  # library raises various error types on download/auth failure
            raise EncoderError(
                f"Failed to load text encoder '{self.model_name}': {e}"
            ) from e

        logger.info("Loaded EmbeddingGemma on %s (mrl_truncate=%s)", self.device, self.mrl_truncate)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise EncoderError("encode() called with an empty text list")

        self._load()
        safe_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

        try:
            embeddings = self._model.encode(
                safe_texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        except RuntimeError as e:
            raise EncoderError(f"Text encoding failed (possible OOM): {e}") from e

        if self.mrl_truncate:
            embeddings = embeddings[:, : self.mrl_truncate]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.float().cpu()

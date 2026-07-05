"""SigLIP2 vision encoder (frozen). Defaults to the Giant (1B) variant but
any siglip2-* checkpoint can be passed via model_name — swap to
siglip2-so400m or siglip2-large through config alone if you're compute
constrained (see the pipeline design doc, Phase 3 note)."""

from typing import List, Optional

import torch

from src.encoders.base import ImageEncoder
from src.utils.exceptions import EncoderError
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DIM_BY_VARIANT = {
    "giant": 1792,
    "so400m": 1152,
    "large": 1024,
    "base": 768,
}


class SigLIP2Encoder(ImageEncoder):
    def __init__(
        self,
        model_name: str = "google/siglip2-giant-opt-patch16-384",
        device: Optional[str] = None,
        batch_size: int = 32,
        output_dim: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.output_dim = output_dim or self._infer_dim(model_name)
        self._model = None
        self._processor = None

    @staticmethod
    def _infer_dim(model_name: str) -> int:
        for key, dim in _DIM_BY_VARIANT.items():
            if key in model_name:
                return dim
        logger.warning(
            "Could not infer output_dim from model_name=%s, defaulting to 1152. "
            "Pass output_dim explicitly to avoid this warning.",
            model_name,
        )
        return 1152

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise EncoderError(
                "transformers is required for SigLIP2Encoder. Install with: pip install transformers"
            ) from e

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        except Exception as e:
            raise EncoderError(f"Failed to load image encoder '{self.model_name}': {e}") from e

        logger.info("Loaded SigLIP2 (%s) on %s", self.model_name, self.device)

    @torch.no_grad()
    def encode(self, images: List) -> torch.Tensor:
        if not images:
            raise EncoderError("encode() called with an empty image list")

        self._load()

        try:
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            outputs = self._model.get_image_features(**inputs)
        except RuntimeError as e:
            raise EncoderError(f"Image encoding failed (possible OOM): {e}") from e
        except (ValueError, TypeError) as e:
            raise EncoderError(f"Image preprocessing failed, check input images: {e}") from e

        return outputs.float().cpu()

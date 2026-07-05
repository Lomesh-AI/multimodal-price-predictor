"""Encoder registry — the single extension point for adding new backbones.

To add a new encoder: implement TextEncoder or ImageEncoder in a new file,
register it here, add a config block. Nothing else in the pipeline changes.
"""

from typing import Any, Dict

from src.encoders.base import ImageEncoder, TextEncoder
from src.encoders.embeddinggemma import EmbeddingGemmaEncoder
from src.encoders.siglip2 import SigLIP2Encoder
from src.utils.exceptions import EncoderError
from src.utils.logging import get_logger

logger = get_logger(__name__)

TEXT_ENCODER_REGISTRY = {
    "embeddinggemma": EmbeddingGemmaEncoder,
}

IMAGE_ENCODER_REGISTRY = {
    "siglip2-giant": SigLIP2Encoder,
    "siglip2-so400m": SigLIP2Encoder,
    "siglip2-large": SigLIP2Encoder,
}


def build_text_encoder(cfg: Dict[str, Any]) -> TextEncoder:
    name = cfg.get("name")
    if name not in TEXT_ENCODER_REGISTRY:
        raise EncoderError(
            f"Unknown text encoder '{name}'. Available: {list(TEXT_ENCODER_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "output_dim")}
    logger.info("Building text encoder: %s", name)
    return TEXT_ENCODER_REGISTRY[name](**kwargs)


def build_image_encoder(cfg: Dict[str, Any]) -> ImageEncoder:
    name = cfg.get("name")
    if name not in IMAGE_ENCODER_REGISTRY:
        raise EncoderError(
            f"Unknown image encoder '{name}'. Available: {list(IMAGE_ENCODER_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "output_dim")}
    logger.info("Building image encoder: %s", name)
    return IMAGE_ENCODER_REGISTRY[name](**kwargs)

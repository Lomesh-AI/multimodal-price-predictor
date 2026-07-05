"""Live prediction on raw (unseen, uncached) text/image inputs. This is the
one place the full chain — encoders + fusion + head — runs together, and
is what api/main.py calls."""

from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data.image_pipeline import process_image
from src.data.text_cleaning import clean_text
from src.encoders.registry import build_image_encoder, build_text_encoder
from src.models.price_model import EndToEndPriceModel, PriceModel
from src.utils.exceptions import CheckpointError, InferenceError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, config: Dict[str, Any], checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            price_model = PriceModel.from_config(config)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            price_model.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError as e:
            raise CheckpointError(f"Checkpoint not found: {checkpoint_path}") from e
        except (KeyError, RuntimeError) as e:
            raise CheckpointError(
                f"Checkpoint at {checkpoint_path} is incompatible with the current "
                f"model config: {e}"
            ) from e

        text_encoder = build_text_encoder(config["encoders"]["text"])
        image_encoder = build_image_encoder(config["encoders"]["image"])

        self.model = EndToEndPriceModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            price_model=price_model,
            device=self.device,
        )
        self.image_size = config["data"].get("image_size", 384)
        logger.info("Predictor ready on %s (checkpoint=%s)", self.device, checkpoint_path)

    def predict_one(self, text: str, image_source: str) -> float:
        return self.predict_batch([text], [image_source])[0]

    def predict_batch(self, texts: List[str], image_sources: List[str]) -> List[float]:
        if len(texts) != len(image_sources):
            raise InferenceError(
                f"texts ({len(texts)}) and image_sources ({len(image_sources)}) "
                "must be the same length"
            )
        if not texts:
            raise InferenceError("predict_batch called with empty input lists")

        try:
            cleaned_texts = [clean_text(t) for t in texts]
            images = [process_image(src, size=self.image_size)[0] for src in image_sources]
            prices = self.model.predict(cleaned_texts, images)
        except Exception as e:
            raise InferenceError(f"Prediction failed: {e}") from e

        return [round(float(p), 2) for p in prices]

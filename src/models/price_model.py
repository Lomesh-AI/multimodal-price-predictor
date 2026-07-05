"""Assembles the trainable model (projections + fusion + head) from config,
and an end-to-end wrapper that also owns the frozen encoders for inference
on raw, never-before-seen text/image pairs.

Design note: PriceModel operates on precomputed embeddings (fast, used in
training). EndToEndPriceModel wraps PriceModel with the encoders attached
(slower, used only in scripts/05_predict.py and the live API) — this mirrors
the cached-vs-live split described in the pipeline design doc.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.encoders.base import ImageEncoder, TextEncoder
from src.fusion.registry import build_fusion
from src.models.regression_head import PriceRegressionHead
from src.utils.exceptions import ModelBuildError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PriceModel(nn.Module):
    """Trainable component only: projections + fusion + regression head.
    Takes precomputed text/image embeddings, outputs a price."""

    def __init__(
        self,
        text_input_dim: int,
        image_input_dim: int,
        fusion_cfg: Dict[str, Any],
        hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        proj_dim = fusion_cfg.get("proj_dim", 512)

        try:
            self.text_proj = nn.Linear(text_input_dim, proj_dim)
            self.image_proj = nn.Linear(image_input_dim, proj_dim)
            self.fusion = build_fusion(fusion_cfg)
            self.head = PriceRegressionHead(
                input_dim=self.fusion.output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        except Exception as e:
            raise ModelBuildError(f"Failed to assemble PriceModel: {e}") from e

        logger.info(
            "Built PriceModel: text_dim=%d image_dim=%d proj_dim=%d fusion=%s head=%s",
            text_input_dim, image_input_dim, proj_dim,
            fusion_cfg.get("type"), hidden_dims,
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_p = self.text_proj(text_emb)
        image_p = self.image_proj(image_emb)
        fused = self.fusion(text_p, image_p)
        return self.head(fused)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PriceModel":
        try:
            return cls(
                text_input_dim=config["encoders"]["text"]["mrl_truncate"]
                or config["encoders"]["text"]["output_dim"],
                image_input_dim=config["encoders"]["image"]["output_dim"],
                fusion_cfg=config["fusion"],
                hidden_dims=config["head"]["hidden_dims"],
                dropout=config["head"].get("dropout", 0.1),
            )
        except KeyError as e:
            raise ModelBuildError(f"Config missing required key for PriceModel: {e}") from e


class EndToEndPriceModel:
    """Wraps frozen encoders + a trained PriceModel for prediction on raw
    text/image inputs. Not an nn.Module itself — it orchestrates two
    frozen encoders (no grad) and one trained module."""

    def __init__(
        self,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder,
        price_model: PriceModel,
        device: Optional[str] = None,
    ):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.price_model = price_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.price_model.to(self.device)
        self.price_model.eval()

    @torch.no_grad()
    def predict(self, texts: List[str], images: List) -> torch.Tensor:
        text_emb = self.text_encoder.encode(texts).to(self.device)
        image_emb = self.image_encoder.encode(images).to(self.device)
        raw_price = self.price_model(text_emb, image_emb)
        return torch.clamp(raw_price, min=0.01)

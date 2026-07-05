"""Gated fusion. A learned sigmoid gate decides, per-sample, how much to
trust text vs image before concatenating — cheaper than cross-attention,
still allows the model to weight modalities adaptively."""

import torch
import torch.nn as nn

from src.fusion.base import FusionModule
from src.utils.exceptions import FusionError


class GatedFusion(FusionModule):
    def __init__(self, proj_dim: int = 512):
        super().__init__()
        self.proj_dim = proj_dim
        self.output_dim = proj_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.shape[-1] != self.proj_dim or image_emb.shape[-1] != self.proj_dim:
            raise FusionError(
                f"Expected inputs with last dim {self.proj_dim}, got "
                f"{text_emb.shape[-1]} and {image_emb.shape[-1]}"
            )
        g = self.gate(torch.cat([text_emb, image_emb], dim=-1))
        gated_text = g * text_emb
        gated_image = (1 - g) * image_emb
        return torch.cat([gated_text, gated_image], dim=-1)

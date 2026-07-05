"""Simple concatenation fusion — the baseline every ablation compares against.
No cross-modal interaction: the two vectors stay parallel until the head."""

import torch

from src.fusion.base import FusionModule
from src.utils.exceptions import FusionError


class ConcatFusion(FusionModule):
    def __init__(self, proj_dim: int = 512):
        super().__init__()
        self.proj_dim = proj_dim
        self.output_dim = proj_dim * 2

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.shape[-1] != self.proj_dim or image_emb.shape[-1] != self.proj_dim:
            raise FusionError(
                f"Expected inputs with last dim {self.proj_dim}, got "
                f"{text_emb.shape[-1]} and {image_emb.shape[-1]}"
            )
        return torch.cat([text_emb, image_emb], dim=-1)

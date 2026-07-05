"""Dual-path cross-attention fusion.

Text queries image (Q=text, K/V=image) and image queries text (Q=image,
K/V=text) in two independent paths, each followed by a residual connection,
layer norm, and a small FFN. The two enriched vectors are concatenated.

Note (documented deliberately, not hidden): because both inputs are pooled,
single-vector embeddings rather than token sequences, each attention call
has sequence length 1 on both sides. This still functions as a learned
cross-modal gating/mixing operation via the Q/K/V projections, but it is not
performing token-level attention. Describe it that way in write-ups.
"""

import torch
import torch.nn as nn

from src.fusion.base import FusionModule
from src.utils.exceptions import FusionError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DualPathCrossAttention(FusionModule):
    def __init__(self, proj_dim: int = 512, heads: int = 8, ffn_mult: int = 2):
        super().__init__()
        if proj_dim % heads != 0:
            raise FusionError(
                f"proj_dim ({proj_dim}) must be divisible by heads ({heads})"
            )

        self.proj_dim = proj_dim
        self.output_dim = proj_dim * 2

        self.text_to_image = nn.MultiheadAttention(proj_dim, heads, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(proj_dim, heads, batch_first=True)

        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(proj_dim * ffn_mult, proj_dim),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(proj_dim * ffn_mult, proj_dim),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        if text_emb.dim() != 2 or image_emb.dim() != 2:
            raise FusionError(
                f"Expected 2D (batch, proj_dim) inputs, got shapes "
                f"{tuple(text_emb.shape)} and {tuple(image_emb.shape)}"
            )
        if text_emb.shape[-1] != self.proj_dim or image_emb.shape[-1] != self.proj_dim:
            raise FusionError(
                f"Expected inputs with last dim {self.proj_dim}, got "
                f"{text_emb.shape[-1]} and {image_emb.shape[-1]}"
            )

        text_seq = text_emb.unsqueeze(1)   # (batch, 1, proj_dim)
        image_seq = image_emb.unsqueeze(1)

        t2i, _ = self.text_to_image(query=text_seq, key=image_seq, value=image_seq)
        t2i = self.norm1(t2i.squeeze(1) + text_emb)
        t2i = self.ffn1(t2i) + t2i

        i2t, _ = self.image_to_text(query=image_seq, key=text_seq, value=text_seq)
        i2t = self.norm2(i2t.squeeze(1) + image_emb)
        i2t = self.ffn2(i2t) + i2t

        return torch.cat([t2i, i2t], dim=-1)

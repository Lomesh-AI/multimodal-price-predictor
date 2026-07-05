"""Abstract fusion interface. Every fusion strategy (cross-attention, concat,
gated) implements this so swapping strategies is a one-line config change,
and the same interface makes an apples-to-apples ablation study trivial."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FusionModule(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """text_emb, image_emb: (batch, proj_dim). Returns (batch, output_dim)."""
        raise NotImplementedError

"""Minimalist regression head. This is the literal last-layer-trained
component of the whole pipeline — everything above it (encoders) is frozen,
everything at its level (projections, fusion) feeds into it."""

from typing import List

import torch
import torch.nn as nn

from src.utils.exceptions import ModelBuildError


class PriceRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        if not hidden_dims:
            raise ModelBuildError("hidden_dims must contain at least one layer")

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ModelBuildError(f"Expected 2D input (batch, dim), got shape {tuple(x.shape)}")
        return self.net(x).squeeze(-1)

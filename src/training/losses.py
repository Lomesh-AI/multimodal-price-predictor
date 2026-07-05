"""Loss functions. SMAPE matches the actual evaluation metric, so training
directly on it aligns gradients with what will be scored."""

from typing import Dict

import torch
import torch.nn as nn

from src.utils.exceptions import TrainingError


class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = 2.0 * torch.abs(pred - target)
        denominator = torch.abs(pred) + torch.abs(target) + self.eps
        return torch.mean(numerator / denominator)


_LOSS_REGISTRY: Dict[str, type] = {
    "smape": SMAPELoss,
    "mse": nn.MSELoss,
}


def build_loss(name: str) -> nn.Module:
    if name not in _LOSS_REGISTRY:
        raise TrainingError(f"Unknown loss '{name}'. Available: {list(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name]()

"""AdamW + cosine annealing, the training configuration verified against
the original writeup as standard, sound practice."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.exceptions import TrainingError


def build_optimizer_and_scheduler(
    model: nn.Module, training_cfg: Dict[str, Any], total_steps: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if total_steps <= 0:
        raise TrainingError(f"total_steps must be positive, got {total_steps}")

    lr = training_cfg.get("lr", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 0.01)
    eta_min = training_cfg.get("eta_min", 1e-6)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise TrainingError("Model has no trainable parameters — check that the encoders "
                             "are excluded and the fusion/head modules are included")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    return optimizer, scheduler

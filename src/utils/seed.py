"""Seed every RNG the project touches, for reproducible runs."""

import os
import random

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if installed) RNGs.

    Torch is imported lazily so this module has no hard torch dependency for
    callers that only need numpy-level reproducibility (e.g. evaluation scripts).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed, skipping torch seeding")

    logger.info("Seeded RNGs with seed=%d", seed)

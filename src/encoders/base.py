"""Abstract encoder interfaces.

Any new backbone (a different text model, a different vision model, or later
a hosted API-based encoder) is a drop-in replacement as long as it implements
encode() and declares output_dim. Nothing downstream needs to change except
the config file and the projection layer's input size.
"""

from abc import ABC, abstractmethod
from typing import List

import torch


class TextEncoder(ABC):
    output_dim: int

    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Returns a (batch, output_dim) float32 tensor. Must run frozen
        (no gradient tracking) internally — callers should not have to
        wrap calls in torch.no_grad() themselves."""
        raise NotImplementedError


class ImageEncoder(ABC):
    output_dim: int

    @abstractmethod
    def encode(self, images: List) -> torch.Tensor:
        """Returns a (batch, output_dim) float32 tensor. Accepts a list of
        PIL.Image objects. Must run frozen internally."""
        raise NotImplementedError

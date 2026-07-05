"""Fusion registry — maps config fusion.type to a concrete FusionModule."""

from typing import Any, Dict

from src.fusion.base import FusionModule
from src.fusion.concat import ConcatFusion
from src.fusion.cross_attention import DualPathCrossAttention
from src.fusion.gated import GatedFusion
from src.utils.exceptions import FusionError
from src.utils.logging import get_logger

logger = get_logger(__name__)

FUSION_REGISTRY = {
    "cross_attention": DualPathCrossAttention,
    "concat": ConcatFusion,
    "gated": GatedFusion,
}


def build_fusion(cfg: Dict[str, Any]) -> FusionModule:
    fusion_type = cfg.get("type")
    if fusion_type not in FUSION_REGISTRY:
        raise FusionError(f"Unknown fusion.type '{fusion_type}'. Available: {list(FUSION_REGISTRY)}")

    cls = FUSION_REGISTRY[fusion_type]
    kwargs = {"proj_dim": cfg.get("proj_dim", 512)}
    if fusion_type == "cross_attention":
        kwargs["heads"] = cfg.get("heads", 8)

    logger.info("Building fusion module: %s", fusion_type)
    return cls(**kwargs)

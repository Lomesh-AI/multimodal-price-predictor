import torch
import pytest

from src.fusion.concat import ConcatFusion
from src.fusion.cross_attention import DualPathCrossAttention
from src.fusion.gated import GatedFusion
from src.fusion.registry import build_fusion
from src.utils.exceptions import FusionError


@pytest.mark.parametrize("fusion_cls", [ConcatFusion, GatedFusion])
def test_fusion_output_shape(fusion_cls):
    fusion = fusion_cls(proj_dim=64)
    text_emb = torch.randn(4, 64)
    image_emb = torch.randn(4, 64)
    out = fusion(text_emb, image_emb)
    assert out.shape == (4, fusion.output_dim)


def test_cross_attention_output_shape():
    fusion = DualPathCrossAttention(proj_dim=64, heads=4)
    text_emb = torch.randn(4, 64)
    image_emb = torch.randn(4, 64)
    out = fusion(text_emb, image_emb)
    assert out.shape == (4, 128)


def test_cross_attention_rejects_bad_heads():
    with pytest.raises(FusionError):
        DualPathCrossAttention(proj_dim=65, heads=8)


def test_fusion_rejects_dim_mismatch():
    fusion = ConcatFusion(proj_dim=64)
    with pytest.raises(FusionError):
        fusion(torch.randn(4, 64), torch.randn(4, 32))


def test_build_fusion_unknown_type_raises():
    with pytest.raises(FusionError):
        build_fusion({"type": "not_a_real_fusion"})


def test_build_fusion_cross_attention_from_config():
    fusion = build_fusion({"type": "cross_attention", "proj_dim": 64, "heads": 4})
    assert isinstance(fusion, DualPathCrossAttention)

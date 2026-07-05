import torch
import pytest

from src.models.price_model import PriceModel
from src.models.regression_head import PriceRegressionHead
from src.utils.exceptions import ModelBuildError


def test_regression_head_output_shape():
    head = PriceRegressionHead(input_dim=32, hidden_dims=[16, 8])
    x = torch.randn(4, 32)
    out = head(x)
    assert out.shape == (4,)


def test_regression_head_rejects_empty_hidden_dims():
    with pytest.raises(ModelBuildError):
        PriceRegressionHead(input_dim=32, hidden_dims=[])


def test_price_model_forward_shape():
    model = PriceModel(
        text_input_dim=32,
        image_input_dim=48,
        fusion_cfg={"type": "concat", "proj_dim": 16},
        hidden_dims=[16, 8],
    )
    text_emb = torch.randn(4, 32)
    image_emb = torch.randn(4, 48)
    out = model(text_emb, image_emb)
    assert out.shape == (4,)


def test_price_model_from_config():
    config = {
        "encoders": {
            "text": {"output_dim": 32, "mrl_truncate": None},
            "image": {"output_dim": 48},
        },
        "fusion": {"type": "gated", "proj_dim": 16},
        "head": {"hidden_dims": [16, 8], "dropout": 0.1},
    }
    model = PriceModel.from_config(config)
    out = model(torch.randn(2, 32), torch.randn(2, 48))
    assert out.shape == (2,)

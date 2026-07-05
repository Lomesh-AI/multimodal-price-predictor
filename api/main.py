"""FastAPI app for live price prediction. Loads the predictor once at
startup (expensive: loads two model backbones) and reuses it per request."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predictor import Predictor
from src.utils.config import load_config
from src.utils.exceptions import CheckpointError, ConfigError, InferenceError
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Multimodal Price Predictor")

_predictor: Optional[Predictor] = None


class PredictRequest(BaseModel):
    text: str
    image_url: str


class PredictResponse(BaseModel):
    predicted_price: float


@app.on_event("startup")
def load_predictor() -> None:
    global _predictor
    try:
        config = load_config("configs/base.yaml")
        checkpoint_path = f"{config['checkpoint_dir']}/best.pt"
        _predictor = Predictor(config, checkpoint_path)
        logger.info("Predictor loaded at startup")
    except (ConfigError, CheckpointError) as e:
        logger.error("Failed to load predictor at startup: %s", e)
        _predictor = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _predictor is not None else "model_not_loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server startup logs")

    try:
        price = _predictor.predict_one(req.text, req.image_url)
    except InferenceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return PredictResponse(predicted_price=price)

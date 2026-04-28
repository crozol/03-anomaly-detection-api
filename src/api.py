"""FastAPI service: POST /predict returns an anomaly score for one window.

Endpoints
---------

* ``GET  /health``    — liveness probe.
* ``GET  /info``      — model metadata: window length, n_features,
                        threshold, sensor list, normalisation stats.
* ``POST /predict``   — evaluate a single window. Body: ``{"values":
                        [[float] * n_features] * seq_len}``. Returns
                        ``{"score": float, "threshold": float,
                          "is_anomaly": bool}``.

The model and the normalisation statistics are loaded once at startup
from ``MODEL_PATH`` (env var, defaults to ``checkpoints/autoencoder.pt``).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .autoencoder import LSTMAutoencoder
from .train import load_checkpoint


MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/autoencoder.pt")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "data/eval_report.json")


# --------------------------------------------------------------------------- #
#  schemas
# --------------------------------------------------------------------------- #

class WindowRequest(BaseModel):
    values: list[list[float]] = Field(
        ...,
        description="Window as a (seq_len, n_features) matrix of raw sensor "
                    "readings. The service normalises with the stats stored "
                    "in the checkpoint before scoring.",
    )


class Prediction(BaseModel):
    score: float = Field(..., description="Reconstruction MSE on the window.")
    threshold: float = Field(..., description="Calibrated decision threshold.")
    is_anomaly: bool = Field(..., description="score > threshold")


class InfoResponse(BaseModel):
    seq_len: int
    n_features: int
    sensor_columns: list[str]
    threshold: float
    healthy_cutoff: int
    anomaly_cutoff: int
    n_params: int
    normalisation: dict[str, list[float]]


# --------------------------------------------------------------------------- #
#  application state
# --------------------------------------------------------------------------- #

class _State:
    model: LSTMAutoencoder | None = None
    payload: dict | None = None
    threshold: float = float("nan")


state = _State()


def _load_threshold(report_path: str | os.PathLike) -> float:
    """Read the calibrated threshold from the eval report on disk.

    If the report is missing, fall back to NaN — ``/predict`` will then
    raise 503 until the operator runs ``main.py`` and produces it.
    """
    p = Path(report_path)
    if not p.exists():
        return float("nan")
    import json
    with open(p) as fh:
        return float(json.load(fh)["threshold"])


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state.model, state.payload = load_checkpoint(MODEL_PATH, device="cpu")
    state.threshold = _load_threshold(THRESHOLD_PATH)
    yield


app = FastAPI(
    title="CMAPSS Anomaly Detection API",
    version="1.0.0",
    description=(
        "LSTM autoencoder trained on healthy CMAPSS FD001 windows. "
        "POST /predict accepts a (seq_len, n_features) window of raw sensor "
        "readings and returns the reconstruction error and the binary flag."
    ),
    lifespan=_lifespan,
)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _ensure_loaded() -> None:
    if state.model is None or state.payload is None:
        raise HTTPException(status_code=503, detail="model not loaded")


def _validate_window(values: list[list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"window must be a 2-D matrix, got ndim={arr.ndim}",
        )
    seq_len = state.payload["seq_len"]
    n_feat = len(state.payload["sensor_cols"])
    if arr.shape != (seq_len, n_feat):
        raise HTTPException(
            status_code=422,
            detail=(
                f"window shape mismatch: expected ({seq_len}, {n_feat}), "
                f"got {tuple(arr.shape)}"
            ),
        )
    return arr


def _score(arr: np.ndarray) -> float:
    mean = state.payload["mean"].astype(np.float32)
    std = state.payload["std"].astype(np.float32)
    norm = (arr - mean) / std
    x = torch.from_numpy(norm).unsqueeze(0)
    with torch.no_grad():
        return float(state.model.reconstruction_error(x).item())


# --------------------------------------------------------------------------- #
#  routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    _ensure_loaded()
    p = state.payload
    n_params = sum(par.numel() for par in state.model.parameters())
    return InfoResponse(
        seq_len=int(p["seq_len"]),
        n_features=len(p["sensor_cols"]),
        sensor_columns=list(p["sensor_cols"]),
        threshold=float(state.threshold),
        healthy_cutoff=int(p.get("healthy_cutoff", 100)),
        anomaly_cutoff=int(p.get("anomaly_cutoff", 30)),
        n_params=int(n_params),
        normalisation={
            "mean": [float(x) for x in p["mean"]],
            "std": [float(x) for x in p["std"]],
        },
    )


@app.post("/predict", response_model=Prediction)
def predict(req: WindowRequest) -> Prediction:
    _ensure_loaded()
    if not (state.threshold == state.threshold):  # NaN check
        raise HTTPException(
            status_code=503,
            detail=("threshold is not available; run main.py to generate "
                    "data/eval_report.json"),
        )
    arr = _validate_window(req.values)
    score = _score(arr)
    return Prediction(
        score=score,
        threshold=float(state.threshold),
        is_anomaly=bool(score > state.threshold),
    )

"""Smoke tests for the FastAPI service.

The tests build a fixture `TestClient` that runs the lifespan once,
loading the trained checkpoint and the calibrated threshold from disk.
Skip is gracefully triggered if those artefacts are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "autoencoder.pt"
EVAL = ROOT / "data" / "eval_report.json"


pytestmark = pytest.mark.skipif(
    not CKPT.exists() or not EVAL.exists(),
    reason="run `python main.py` first to produce checkpoint + eval report",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    from src.api import app
    with TestClient(app) as tc:
        yield tc


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_info_endpoint(client: TestClient) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    payload = r.json()
    assert payload["seq_len"] == 30
    assert payload["n_features"] == 14
    assert payload["threshold"] > 0
    assert len(payload["sensor_columns"]) == payload["n_features"]
    assert len(payload["normalisation"]["mean"]) == payload["n_features"]


def _example_window(seq_len: int, n_features: int) -> list[list[float]]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(seq_len, n_features), scale=0.5).tolist()


def test_predict_shape_validation(client: TestClient) -> None:
    bad = {"values": [[1.0, 2.0]]}    # wrong shape
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_returns_score(client: TestClient) -> None:
    info = client.get("/info").json()
    payload = {"values": _example_window(info["seq_len"], info["n_features"])}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert {"score", "threshold", "is_anomaly"} <= body.keys()
    assert isinstance(body["score"], float) and body["score"] >= 0
    assert isinstance(body["is_anomaly"], bool)


def test_predict_anomalous_window_flagged(client: TestClient) -> None:
    """A window with implausibly large values must be flagged."""
    info = client.get("/info").json()
    seq_len, n_feat = info["seq_len"], info["n_features"]
    payload = {"values": (np.full((seq_len, n_feat), 1e4)).tolist()}
    r = client.post("/predict", json=payload)
    body = r.json()
    assert body["is_anomaly"] is True
    assert body["score"] > body["threshold"]

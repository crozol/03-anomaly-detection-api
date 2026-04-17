"""FastAPI app: endpoint /predict que recibe una serie multivariada y devuelve el score de anomalía."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Anomaly Detection API", version="0.1.0")


class Series(BaseModel):
    values: list[list[float]] = Field(..., description="Matriz [seq_len x n_features] con la serie a evaluar.")


class Prediction(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(series: Series) -> Prediction:
    # TODO: cargar modelo entrenado, correr reconstrucción y comparar contra umbral
    raise NotImplementedError("Cargar modelo entrenado y calcular score de reconstrucción.")

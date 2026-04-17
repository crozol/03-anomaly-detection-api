# Anomaly Detection API — Autoencoder + FastAPI + Docker

Anomaly detection on multivariate time series (NASA Turbofan CMAPSS engine sensors) using an autoencoder trained on nominal data, served as a REST API with an interactive Streamlit demo.

## Motivation

This is the "engineering pivot" of the portfolio: it combines applied deep learning with the skills companies expect from an ML Engineer (APIs, containers, reproducible demos).

The idea: an autoencoder learns to reconstruct normal sequences. When given an anomalous sequence, reconstruction fails and reconstruction error rises → flagged as an anomaly.

## Stack

- Python 3.11+
- PyTorch 2.x
- FastAPI + Uvicorn + Pydantic
- Docker + docker-compose
- Streamlit (frontend demo)
- Dataset: [NASA Turbofan CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

## Structure

```
03-anomaly-detection-api/
├── README.md
├── requirements.txt
├── Dockerfile             # (pending)
├── src/
│   ├── autoencoder.py     # LSTM / 1D-CNN autoencoder model
│   ├── data.py            # Dataset loading and preprocessing
│   ├── train.py           # Training loop
│   ├── api.py             # FastAPI: /predict endpoint
│   └── app_streamlit.py   # Interactive demo
└── notebooks/
```

## Roadmap

- [ ] Download and preprocess CMAPSS dataset.
- [ ] Implement autoencoder and train on "healthy" sequences.
- [ ] Define anomaly threshold (99th percentile of validation error).
- [ ] FastAPI `/predict` endpoint accepting a series and returning `{is_anomaly, score}`.
- [ ] Streamlit demo to upload a CSV and visualize results.
- [ ] Dockerfile + docker-compose.
- [ ] Final README with metrics (precision / recall / ROC-AUC) and demo screenshot.

## How to run (when ready)

```bash
docker-compose up --build
# API at http://localhost:8000/docs
# Demo at http://localhost:8501
```

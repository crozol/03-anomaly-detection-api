# Anomaly Detection API — Autoencoder + FastAPI + Docker

Detección de anomalías en series temporales multivariadas (sensores de motores NASA Turbofan CMAPSS) usando un autoencoder entrenado con datos nominales, expuesto como API REST y con demo interactiva en Streamlit.

## Motivación

Este proyecto es el "pivote de ingeniería" del portafolio: combina deep learning aplicado con las habilidades que las empresas esperan de un MLE junior (APIs, contenedores, demos reproducibles).

La idea: un autoencoder aprende a reconstruir series normales. Cuando recibe una serie anómala, la reconstrucción falla y el error de reconstrucción sube → se marca como anomalía.

## Stack

- Python 3.11+
- PyTorch 2.x
- FastAPI + Uvicorn + Pydantic
- Docker + docker-compose
- Streamlit (demo frontend)
- Dataset: [NASA Turbofan CMAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

## Estructura

```
03-anomaly-detection-api/
├── README.md
├── requirements.txt
├── Dockerfile             # (pendiente)
├── src/
│   ├── autoencoder.py     # Modelo autoencoder LSTM / 1D-CNN
│   ├── data.py            # Carga y preprocesamiento del dataset
│   ├── train.py           # Training loop
│   ├── api.py             # FastAPI: endpoint /predict
│   └── app_streamlit.py   # Demo interactiva
└── notebooks/
```

## Roadmap

- [ ] Descargar y preprocesar dataset CMAPSS.
- [ ] Implementar autoencoder y entrenar con series "sanas".
- [ ] Definir umbral de anomalía (percentil 99 del error en validación).
- [ ] Endpoint FastAPI `/predict` que recibe una serie y devuelve `{is_anomaly, score}`.
- [ ] Demo Streamlit para subir CSV y visualizar resultados.
- [ ] Dockerfile + docker-compose.
- [ ] README final con métricas (precision/recall/ROC-AUC) y screenshot de la demo.

## Cómo correrlo (cuando esté listo)

```bash
docker-compose up --build
# API en http://localhost:8000/docs
# Demo en http://localhost:8501
```

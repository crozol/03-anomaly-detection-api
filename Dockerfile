# syntax=docker/dockerfile:1.6
# Multi-stage Dockerfile for the CMAPSS anomaly-detection service.
#
# Stage 1 (builder)  — install Python dependencies into a virtualenv that
#                      we copy into the runtime stage. CPU-only torch
#                      wheel keeps the image well under 1 GB.
# Stage 2 (runtime)  — slim Python base with the venv copied in and the
#                      application code on top. Default command runs the
#                      FastAPI server; docker-compose overrides for the
#                      Streamlit demo.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.* \
 && /opt/venv/bin/pip install -r requirements.txt


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/checkpoints/autoencoder.pt \
    THRESHOLD_PATH=/app/data/eval_report.json

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code, trained model, and minimal data needed by the API
# and the Streamlit demo (eval_report for the threshold; raw FD001 only
# if you want the "built-in engine" button to work — kept opt-in via
# the docker-compose mount).
COPY src ./src
COPY scripts ./scripts
COPY main.py ./main.py
COPY checkpoints/autoencoder.pt ./checkpoints/autoencoder.pt
COPY data/eval_report.json ./data/eval_report.json

EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

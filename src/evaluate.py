"""Threshold calibration and binary-classification metrics.

The autoencoder is trained on healthy data alone, so it does not produce
a probability — only a reconstruction error. Turning that into a yes/no
flag requires picking a threshold. The convention used here is the one
recommended in the anomaly-detection literature for this kind of model:

    threshold = quantile_q(reconstruction error on the healthy
                           validation windows)

with ``q = 0.99`` by default. A higher ``q`` trades recall for
precision; a lower ``q`` does the opposite.

The functions below compute that threshold, apply it to the test windows
to produce binary predictions, and report the standard set of binary
metrics plus a sweep over thresholds for the ROC curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import json

import numpy as np

from .autoencoder import LSTMAutoencoder
from .data import Dataset
from .train import reconstruction_errors


# --------------------------------------------------------------------------- #
#  threshold + metric containers
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    threshold: float
    quantile: float
    n_test: int
    n_anomalies: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    pr_auc: float
    confusion: dict[str, int]      # tn, fp, fn, tp
    err_healthy_train_mean: float
    err_healthy_val_mean: float
    err_test_healthy_mean: float
    err_test_anomalous_mean: float


# --------------------------------------------------------------------------- #
#  threshold and metrics
# --------------------------------------------------------------------------- #

def calibrate_threshold(err_val: np.ndarray, quantile: float = 0.99) -> float:
    """Threshold = ``quantile``-th quantile of validation reconstruction error."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    return float(np.quantile(err_val, quantile))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def roc_curve(score: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) sorted by descending threshold."""
    order = np.argsort(-score)
    s_sorted = score[order]
    y_sorted = y_true[order].astype(np.int64)

    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        raise ValueError("ROC needs both positive and negative samples")

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.where(np.diff(s_sorted))[0]
    keep = np.concatenate([distinct, [len(s_sorted) - 1]])
    tpr = np.concatenate([[0.0], tps[keep] / pos])
    fpr = np.concatenate([[0.0], fps[keep] / neg])
    thr = np.concatenate([[s_sorted[0] + 1e-12], s_sorted[keep]])
    return fpr, tpr, thr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC for monotonically ordered ``x``."""
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def pr_curve(score: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-recall curve sorted by descending threshold."""
    order = np.argsort(-score)
    s_sorted = score[order]
    y_sorted = y_true[order].astype(np.int64)

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    distinct = np.where(np.diff(s_sorted))[0]
    keep = np.concatenate([distinct, [len(s_sorted) - 1]])
    tps_k, fps_k = tps[keep], fps[keep]
    precision = tps_k / np.maximum(tps_k + fps_k, 1)
    recall = tps_k / max(int(y_true.sum()), 1)
    thr = s_sorted[keep]
    # Prepend the trivial (recall=0, precision=1) endpoint and append
    # (recall=1, precision = base rate) on the right.
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall, thr


# --------------------------------------------------------------------------- #
#  high-level evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: LSTMAutoencoder,
    ds: Dataset,
    quantile: float = 0.99,
    device: str = "cpu",
) -> tuple[EvalReport, dict[str, np.ndarray]]:
    """Run the full evaluation: errors, threshold, metrics, ROC + PR curves.

    Returns ``(report, arrays)`` where ``arrays`` carries the per-window
    reconstruction errors and the ROC / PR sweeps so they can be saved
    by the figures and JSON-export stages without re-running the model.
    """
    err_train = reconstruction_errors(model, ds.train.x, device=device)
    err_val = reconstruction_errors(model, ds.val.x, device=device)
    err_test = reconstruction_errors(model, ds.test.x, device=device)

    threshold = calibrate_threshold(err_val, quantile=quantile)
    y_pred = (err_test > threshold).astype(np.int32)
    metrics = binary_metrics(ds.test_label, y_pred)

    fpr, tpr, roc_thr = roc_curve(err_test, ds.test_label)
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, pr_thr = pr_curve(err_test, ds.test_label)
    pr_auc = float(np.trapezoid(prec_curve, rec_curve))

    healthy_test = err_test[ds.test_label == 0]
    anomalous_test = err_test[ds.test_label == 1]

    report = EvalReport(
        threshold=threshold,
        quantile=quantile,
        n_test=int(ds.test.x.shape[0]),
        n_anomalies=int(ds.test_label.sum()),
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        accuracy=metrics["accuracy"],
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion={
            "tn": metrics["tn"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tp": metrics["tp"],
        },
        err_healthy_train_mean=float(err_train.mean()),
        err_healthy_val_mean=float(err_val.mean()),
        err_test_healthy_mean=float(healthy_test.mean()),
        err_test_anomalous_mean=float(anomalous_test.mean()),
    )

    arrays = {
        "err_train": err_train,
        "err_val": err_val,
        "err_test": err_test,
        "y_test": ds.test_label.astype(np.int32),
        "rul_test": ds.test.rul.astype(np.float32),
        "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thr,
        "precision_curve": prec_curve, "recall_curve": rec_curve, "pr_thresholds": pr_thr,
    }
    return report, arrays


def save_report(path: str | Path, report: EvalReport, extra: dict | None = None) -> None:
    payload = asdict(report)
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


__all__ = [
    "EvalReport",
    "auc",
    "binary_metrics",
    "calibrate_threshold",
    "confusion_counts",
    "evaluate_model",
    "pr_curve",
    "roc_curve",
    "save_report",
]

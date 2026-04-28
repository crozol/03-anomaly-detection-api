"""Export evaluation arrays + metrics to JSON for the portfolio website.

Output is consumed by Plotly charts on
``website/projects/03-anomaly-detection-api.html``.

Usage:
    python -m scripts.export_json
    python -m scripts.export_json --out ../website/assets/data/03-anomaly.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _thin(arr: np.ndarray, max_points: int) -> np.ndarray:
    if arr.shape[0] <= max_points:
        return arr
    step = max(1, arr.shape[0] // max_points)
    return arr[::step]


def _hist(values: np.ndarray, *, bins: int = 60, vmin: float | None = None,
          vmax: float | None = None) -> dict:
    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(np.quantile(values, 0.99))
    counts, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    centers = 0.5 * (edges[1:] + edges[:-1])
    return {
        "centers": [round(float(x), 5) for x in centers],
        "counts": [int(c) for c in counts],
        "vmin": round(vmin, 5),
        "vmax": round(vmax, 5),
    }


def _engine_trace(err: np.ndarray, rul: np.ndarray, unit: np.ndarray,
                  unit_id: int, *, max_points: int = 320) -> dict:
    mask = unit == unit_id
    rul_u = rul[mask]
    err_u = err[mask]
    order = np.argsort(-rul_u)              # left to right = early to late life
    rul_u, err_u = rul_u[order], err_u[order]
    return {
        "unit": int(unit_id),
        "rul": [round(float(x), 2) for x in _thin(rul_u, max_points)],
        "err": [round(float(x), 5) for x in _thin(err_u, max_points)],
    }


def main(out_path: str = "../website/assets/data/03-anomaly.json",
         data_dir: str = "data",
         engine_units: tuple[int, ...] = (81, 95)) -> None:
    data_root = Path(data_dir)
    arrays = np.load(data_root / "eval_arrays.npz")
    with open(data_root / "metrics.json") as fh:
        metrics = json.load(fh)

    err_test = arrays["err_test"]
    y_test = arrays["y_test"]
    rul_test = arrays["rul_test"]

    # The test windows' engine ids are not stored in the npz (they live in
    # ds.test.unit). Recover them from the order of windows: rul_test is
    # monotonically decreasing within each engine, so a strict increase
    # marks an engine boundary.
    boundaries = np.concatenate([[0], np.where(np.diff(rul_test) > 0)[0] + 1,
                                 [len(rul_test)]])
    units = np.zeros_like(rul_test, dtype=np.int32)
    n_engines = len(boundaries) - 1
    # Engines were indexed 81..(81 + n_engines - 1) by build_dataset.
    base_unit = 81
    for i in range(n_engines):
        units[boundaries[i] : boundaries[i + 1]] = base_unit + i

    bundle = {
        "metrics": metrics,
        "loss": {
            "epoch": list(range(1, len(metrics["training"].get(
                "train_loss", [])) + 1)) if "train_loss" in metrics["training"] else None,
        },
        "histograms": {
            "healthy": _hist(err_test[y_test == 0], bins=60),
            "anomalous": _hist(err_test[y_test == 1], bins=60),
            "threshold": metrics["evaluation"]["threshold"],
        },
        "roc": {
            "fpr": [round(float(x), 5) for x in _thin(arrays["fpr"], 200)],
            "tpr": [round(float(x), 5) for x in _thin(arrays["tpr"], 200)],
            "auc": metrics["evaluation"]["roc_auc"],
        },
        "pr": {
            "recall": [round(float(x), 5) for x in _thin(arrays["recall_curve"], 200)],
            "precision": [round(float(x), 5) for x in _thin(arrays["precision_curve"], 200)],
            "auc": metrics["evaluation"]["pr_auc"],
        },
        "engines": [_engine_trace(err_test, rul_test, units, u) for u in engine_units],
        "anomaly_cutoff": metrics["dataset"]["anomaly_cutoff"],
    }

    # If we also kept the per-epoch loss curve, embed it.
    train_metrics_path = data_root / "train_metrics.json"
    if train_metrics_path.exists():
        with open(train_metrics_path) as fh:
            tm = json.load(fh)
        bundle["loss"] = {
            "epoch": list(range(1, len(tm["train_loss"]) + 1)),
            "train": [round(float(x), 6) for x in tm["train_loss"]],
            "val": [round(float(x), 6) for x in tm["val_loss"]],
        }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    size_kb = out.stat().st_size / 1024
    print(f"[ok] {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="../website/assets/data/03-anomaly.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    main(out_path=args.out, data_dir=args.data_dir)

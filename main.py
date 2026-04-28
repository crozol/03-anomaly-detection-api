"""End-to-end pipeline: build dataset, train autoencoder, evaluate, plot.

Outputs:
    figures/                       # PNGs embedded in the README
    data/metrics.json              # summary numbers
    data/eval_arrays.npz           # raw per-window errors + ROC sweep
    checkpoints/autoencoder.pt     # trained weights + normalisation stats

Usage:
    python main.py
    python main.py --no-train      # skip training if a checkpoint exists
    python main.py --epochs 40     # short run for smoke tests
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import plots
from src.autoencoder import AEConfig, LSTMAutoencoder
from src.data import (
    INFORMATIVE_SENSORS_FD001, build_dataset, load_split, summarise_dataset,
)
from src.evaluate import evaluate_model, save_report
from src.train import TrainConfig, fit_pipeline, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="checkpoints/autoencoder.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--no-train", action="store_true",
                   help="reuse the existing checkpoint instead of training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. dataset (also triggers download on first run)
    print("=" * 64)
    print("step 1/4 · dataset")
    print("=" * 64)
    ds = build_dataset(raw_dir=args.raw_dir)
    summary = summarise_dataset(ds)
    print(json.dumps(summary, indent=2))

    # 2. training (or reuse)
    print("\n" + "=" * 64)
    print("step 2/4 · training")
    print("=" * 64)
    ckpt_path = Path(args.checkpoint)
    if args.no_train and ckpt_path.exists():
        print(f"[main] reusing checkpoint at {ckpt_path}")
        model, payload = load_checkpoint(ckpt_path)
        train_loss = payload["train_result"]["train_loss"]
        val_loss = payload["train_result"]["val_loss"]
    else:
        ae_cfg = AEConfig(n_features=ds.train.n_features, seq_len=ds.seq_len)
        train_cfg = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, log_every=10,
        )
        model, ds, result = fit_pipeline(
            raw_dir=args.raw_dir,
            checkpoint_path=str(ckpt_path),
            train_cfg=train_cfg,
            ae_cfg=ae_cfg,
            metrics_path=str(data_dir / "train_metrics.json"),
        )
        train_loss = result.train_loss
        val_loss = result.val_loss

    # 3. evaluation
    print("\n" + "=" * 64)
    print("step 3/4 · evaluation")
    print("=" * 64)
    report, arrays = evaluate_model(model, ds, quantile=args.quantile)
    print(f"  threshold (q={args.quantile}) = {report.threshold:.4f}")
    print(f"  precision = {report.precision:.3f}  recall = {report.recall:.3f}  "
          f"F1 = {report.f1:.3f}")
    print(f"  ROC-AUC   = {report.roc_auc:.3f}    PR-AUC = {report.pr_auc:.3f}")
    print(f"  confusion = {report.confusion}")

    np.savez_compressed(data_dir / "eval_arrays.npz", **arrays)
    save_report(data_dir / "eval_report.json", report,
                extra={"dataset": summary})

    # 4. figures
    print("\n" + "=" * 64)
    print("step 4/4 · figures")
    print("=" * 64)
    raw_df = load_split("FD001", "train", raw_dir=args.raw_dir)
    sensor_cols = [f"sensor_{i}" for i in INFORMATIVE_SENSORS_FD001]
    norm_df = raw_df.copy()
    norm_df[sensor_cols] = (
        raw_df[sensor_cols].to_numpy(dtype=np.float32) - ds.mean
    ) / ds.std
    # keep one engine for the overview figure
    plots.plot_sensor_overview(
        raw_df, norm_df, sensor_cols, unit=1, out_path=out_dir / "sensors.png",
    )
    plots.plot_training_loss(train_loss, val_loss, out_path=out_dir / "training_loss.png")
    plots.plot_error_distributions(
        arrays["err_test"][arrays["y_test"] == 0],
        arrays["err_test"][arrays["y_test"] == 1],
        report.threshold, out_path=out_dir / "error_distributions.png",
    )
    # Operating point on the ROC curve at the calibrated threshold.
    op_fpr = report.confusion["fp"] / max(
        report.confusion["fp"] + report.confusion["tn"], 1)
    op_tpr = report.confusion["tp"] / max(
        report.confusion["tp"] + report.confusion["fn"], 1)
    plots.plot_roc(arrays["fpr"], arrays["tpr"], report.roc_auc,
                   operating_point=(op_fpr, op_tpr),
                   out_path=out_dir / "roc.png")
    plots.plot_confusion(report.confusion, out_path=out_dir / "confusion.png")
    plots.plot_engine_trajectory(
        arrays["err_test"], arrays["rul_test"],
        ds.test.unit, report.threshold, ds.anomaly_cutoff,
        out_path=out_dir / "engine_trajectory.png",
    )

    # 5. metrics summary for the README + website export
    metrics_payload = {
        "dataset": summary,
        "training": {
            "epochs": len(train_loss),
            "final_train_loss": float(train_loss[-1]) if train_loss else None,
            "final_val_loss": float(val_loss[-1]) if val_loss else None,
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        },
        "evaluation": {
            "threshold": report.threshold,
            "quantile": report.quantile,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "accuracy": report.accuracy,
            "roc_auc": report.roc_auc,
            "pr_auc": report.pr_auc,
            "confusion": report.confusion,
            "err_healthy_train_mean": report.err_healthy_train_mean,
            "err_healthy_val_mean": report.err_healthy_val_mean,
            "err_test_healthy_mean": report.err_test_healthy_mean,
            "err_test_anomalous_mean": report.err_test_anomalous_mean,
        },
    }
    with open(data_dir / "metrics.json", "w") as fh:
        json.dump(metrics_payload, fh, indent=2)
    print(f"\nfigures -> {out_dir.resolve()}")
    print(f"metrics -> {(data_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ndone in {time.time() - t0:.1f} s")

"""Baselines, engine-level uncertainty and lead time for the anomaly score.

Every number in the README's comparison table is produced here, on the
same engine-disjoint split and the same labels as the LSTM autoencoder:

* **Cycle index** — score = cycles elapsed since the start of the record.
  The label (RUL <= 30) is monotone in time within each engine, so this
  trivial score is the first thing the autoencoder has to beat.
* **PCA reconstruction** — linear autoencoder: PCA with as many components
  as the LSTM latent (16), fitted on the flattened healthy training
  windows; score = mean squared reconstruction error.
* **Isolation Forest** — fitted on the same flattened healthy windows.
* **One-Class SVM** — RBF kernel (``gamma="scale"``, ``nu=0.05``) on the
  same flattened healthy windows.
* **LSTM autoencoder** — the trained checkpoint.

Uncertainty. Windows overlap 29/30 and the 3 913 test windows come from
only 20 engines, so the effective sample size is ~20, not ~3 900. The 95 %
intervals are a bootstrap over *engines* (resample the 20 test engines with
replacement, recompute the metric).

Threshold rules (README section 7.3), on the autoencoder score: the q99
validation quantile, mean + 3 sigma of the validation score, and the
best-F1 cut chosen with the test labels (an oracle upper bound).

Lead time. For each test engine, the alarm cycle is the first window
after which the score stays above the q99 validation threshold until the
end of the record; the lead time is the RUL at that window.

Usage:
    python -m scripts.baselines            # writes data/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.data import build_dataset
from src.evaluate import auc, average_precision, calibrate_threshold, roc_curve
from src.train import load_checkpoint, reconstruction_errors


def _ranking(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(score, y)
    return auc(fpr, tpr), average_precision(score, y)


def engine_bootstrap(score: np.ndarray, y: np.ndarray, unit: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """95 % percentile intervals for ROC-AUC and AP, resampling engines."""
    rng = np.random.default_rng(seed)
    engines = np.unique(unit)
    idx_by_engine = {u: np.flatnonzero(unit == u) for u in engines}
    rocs, aps = [], []
    for _ in range(n_boot):
        pick = rng.choice(engines, size=len(engines), replace=True)
        idx = np.concatenate([idx_by_engine[u] for u in pick])
        r, a = _ranking(score[idx], y[idx])
        rocs.append(r)
        aps.append(a)
    return {
        "roc_auc_ci95": [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
        "ap_ci95": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
    }


def lead_times(score: np.ndarray, rul: np.ndarray, unit: np.ndarray,
               threshold: float) -> dict:
    """RUL at the first window after which the score stays above threshold."""
    per_engine = {}
    for u in np.unique(unit):
        m = unit == u
        order = np.argsort(-rul[m])                     # early life -> failure
        s_u, r_u = score[m][order], rul[m][order]
        above = s_u > threshold
        # stays above until the end: suffix of all-True
        stays = np.flip(np.cumprod(np.flip(above))).astype(bool)
        per_engine[int(u)] = float(r_u[np.argmax(stays)]) if stays.any() else None
    leads = np.array([v for v in per_engine.values() if v is not None])
    out = {"per_engine": per_engine,
           "n_missed": int(sum(v is None for v in per_engine.values()))}
    if leads.size:
        q1, med, q3 = np.percentile(leads, [25, 50, 75])
        out.update(median=float(med), iqr=[float(q1), float(q3)],
                   min=float(leads.min()), max=float(leads.max()))
    return out


def threshold_rules(s_val: np.ndarray, s_test: np.ndarray, y: np.ndarray,
                    quantile: float) -> dict:
    """README section 7.3: q99 vs mean + 3 sigma vs an oracle best-F1 cut."""
    def at(thr: float) -> dict:
        p = s_test > thr
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / int(y.sum())
        return {"threshold": float(thr), "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                "val_alarm_rate": float((s_val > thr).mean())}
    cands = np.unique(s_test)
    f1s = [at(t)["f1"] for t in cands]
    return {
        "quantile": at(calibrate_threshold(s_val, quantile)),
        "mean_plus_3sigma": at(float(s_val.mean() + 3 * s_val.std())),
        "oracle_best_f1_on_test": at(float(cands[int(np.argmax(f1s))])),
        "val_skewness": float(((s_val - s_val.mean()) ** 3).mean() / s_val.std() ** 3),
    }


def evaluate_score(name: str, s_val: np.ndarray, s_test: np.ndarray, ds,
                   y: np.ndarray, quantile: float, n_boot: int) -> dict:
    roc, ap = _ranking(s_test, y)
    thr = calibrate_threshold(s_val, quantile)
    pred = s_test > thr
    rul = ds.test.rul
    healthy, grey = rul > ds.healthy_cutoff, (rul > ds.anomaly_cutoff) & (rul <= ds.healthy_cutoff)
    res = {
        "name": name,
        "roc_auc": roc,
        "average_precision": ap,
        **engine_bootstrap(s_test, y, ds.test.unit, n_boot=n_boot),
        "threshold": thr,
        "recall": float(pred[y == 1].mean()),
        "fpr_healthy_rul_gt_100": float(pred[healthy].mean()),
        "fpr_grey_zone_30_100": float(pred[grey].mean()),
        "lead_time": lead_times(s_test, rul, ds.test.unit, thr),
    }
    lt = res["lead_time"]
    print(f"  {name:<20} ROC-AUC {roc:.3f} [{res['roc_auc_ci95'][0]:.3f}, "
          f"{res['roc_auc_ci95'][1]:.3f}]  AP {ap:.3f} [{res['ap_ci95'][0]:.3f}, "
          f"{res['ap_ci95'][1]:.3f}]  FPR>100 {res['fpr_healthy_rul_gt_100']:.3f}  "
          f"FPR grey {res['fpr_grey_zone_30_100']:.3f}  lead median "
          f"{lt.get('median', float('nan')):.0f} IQR {lt.get('iqr')}  missed {lt['n_missed']}")
    return res


def main(raw_dir: str, checkpoint: str, out: str, quantile: float,
         n_boot: int) -> None:
    ds = build_dataset(raw_dir=raw_dir)
    y = ds.test_label

    flat = lambda w: w.x.reshape(w.x.shape[0], -1)       # noqa: E731
    x_train, x_val, x_test = flat(ds.train), flat(ds.val), flat(ds.test)

    model, payload = load_checkpoint(checkpoint)
    latent = int(payload["model_cfg"]["latent_dim"])

    pca = PCA(n_components=latent, random_state=0).fit(x_train)
    pca_err = lambda x: np.mean((x - pca.inverse_transform(pca.transform(x))) ** 2, axis=1)  # noqa: E731

    iso = IsolationForest(n_estimators=300, random_state=0).fit(x_train)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(x_train)

    scores = {
        "Cycle index": (ds.val.cycle_end.astype(float), ds.test.cycle_end.astype(float)),
        f"PCA ({latent} comp.)": (pca_err(x_val), pca_err(x_test)),
        "Isolation Forest": (-iso.score_samples(x_val), -iso.score_samples(x_test)),
        "One-Class SVM": (-ocsvm.score_samples(x_val), -ocsvm.score_samples(x_test)),
        "LSTM autoencoder": (reconstruction_errors(model, ds.val.x),
                             reconstruction_errors(model, ds.test.x)),
    }

    print(f"[baselines] {len(np.unique(ds.test.unit))} test engines, "
          f"{len(y)} windows, {int(y.sum())} positives, {n_boot} engine bootstraps")
    results = [evaluate_score(name, sv, st, ds, y, quantile, n_boot)
               for name, (sv, st) in scores.items()]

    rules = threshold_rules(*scores["LSTM autoencoder"], y, quantile)
    for k, v in rules.items():
        print(f"  threshold rule {k}: {v}")

    payload_out = {
        "threshold_rules_lstm": rules,
        "quantile": quantile,
        "n_boot": n_boot,
        "n_test_engines": int(len(np.unique(ds.test.unit))),
        "n_test_windows": int(len(y)),
        "results": results,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload_out, fh, indent=2)
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--checkpoint", default="checkpoints/autoencoder.pt")
    p.add_argument("--out", default="data/baselines.json")
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    main(a.raw_dir, a.checkpoint, a.out, a.quantile, a.n_boot)

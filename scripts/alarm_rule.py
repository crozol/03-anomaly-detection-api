"""Causal alarm rule, lead time against the healthy false-alarm rate, and
paired ROC-AUC differences.

Section 6.6 of the README measures lead time with a rule that is not causal:
the alarm is the first window after which the score *stays* above the
threshold until the end of the record, which needs the future to the end of
the record and hides any earlier, transient crossing. With that rule every
alarm raised more than 30 cycles before failure leaves the windows between it
and RUL = 30 above the threshold, so the grey-zone (30 < RUL <= 100)
false-alarm rate is largely another measure of lead time. This script
replaces it with the rule of the TDA project's analysis plan and compares the
scores at equal false-alarm rates instead of at one threshold each:

* **Causal alarm.** A window is in alarm when the score exceeds the threshold
  on it and on the ``K - 1`` windows before it (``K = 3``): the alarm is known
  at that window, with no look-ahead.
* **Healthy false-alarm rate.** The fraction of test windows with RUL > 100 in
  alarm. The grey zone is not counted either way.
* **Lead time.** For each engine, the RUL at its first window in alarm with
  RUL <= 100; an engine with none is missed. Alarms in the healthy regime are
  charged to the false-alarm rate, not to the lead time.
* **Lead time at equal false-alarm rate.** Each score's threshold is swept
  over quantiles of its own healthy test windows, which traces lead time
  against the healthy false-alarm rate; the scores are compared at the same
  rates (median lead and engines alarmed first, with an exact sign test).
* **Paired ROC-AUC difference.** The difference between two scores' ROC-AUC
  with a 95 % interval from resampling the 20 test engines, both scores on
  the same resample. Overlapping marginal intervals do not show that two
  scores are equivalent; a narrow paired interval bounds the difference.

Usage:
    python -m scripts.alarm_rule        # writes data/alarm_rule.json and figures/lead_vs_false_alarms.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.baselines import _ranking, compute_scores
from src.evaluate import calibrate_threshold

K = 3
HEALTHY, ANOMALY = 100, 30
FA_LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05)       # healthy false-alarm rates compared


def alarm_state(score_in_time_order: np.ndarray, threshold: float, k: int = K) -> np.ndarray:
    """True at window i when windows i-k+1 .. i are all above the threshold."""
    above = (score_in_time_order > threshold).astype(int)
    run = np.convolve(above, np.ones(k, dtype=int), mode="full")[: len(above)]
    return run == k


def operating_point(score: np.ndarray, rul: np.ndarray, unit: np.ndarray,
                    threshold: float, k: int = K) -> dict:
    """Healthy and grey-zone alarm rates and per-engine lead times under the causal rule."""
    alarm = np.zeros(len(score), dtype=bool)
    lead = {}
    for u in np.unique(unit):
        idx = np.flatnonzero(unit == u)
        idx = idx[np.argsort(-rul[idx])]               # early life -> failure
        a = alarm_state(score[idx], threshold, k)
        alarm[idx] = a
        hit = np.flatnonzero(a & (rul[idx] <= HEALTHY))
        lead[int(u)] = float(rul[idx][hit[0]]) if hit.size else None
    healthy = rul > HEALTHY
    grey = (rul > ANOMALY) & (rul <= HEALTHY)
    leads = np.array([v for v in lead.values() if v is not None])
    return {"threshold": float(threshold),
            "healthy_false_alarm_rate": float(alarm[healthy].mean()),
            "grey_zone_alarm_rate": float(alarm[grey].mean()),
            "engines_with_a_healthy_alarm": int(sum(alarm[(unit == u) & healthy].any()
                                                    for u in np.unique(unit))),
            "lead_per_engine": lead,
            "n_missed": int(sum(v is None for v in lead.values())),
            "lead_median": float(np.median(leads)) if leads.size else None}


def sign_test(lead_a: dict, lead_b: dict) -> dict:
    """Engines alarmed first by a and by b (missed = lead 0), exact two-sided sign test."""
    d = np.array([(lead_a[u] or 0.0) - (lead_b[u] or 0.0) for u in lead_a])
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    n, m = n_pos + n_neg, min(n_pos, n_neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    return {"a_earlier": n_pos, "b_earlier": n_neg, "ties": int((d == 0).sum()),
            "median_difference": float(np.median(d)), "sign_test_p": float(p)}


def at_false_alarm_rate(score: np.ndarray, rul: np.ndarray, unit: np.ndarray,
                        rate: float, k: int = K) -> dict:
    """Operating point with the lowest threshold whose healthy alarm rate <= rate.
    Candidate thresholds are the score's own healthy test values (and the max)."""
    cands = np.unique(score[rul > HEALTHY])[::-1]       # high -> low
    best = operating_point(score, rul, unit, float(cands[0]), k)
    for thr in cands[1:]:
        op = operating_point(score, rul, unit, float(thr), k)
        if op["healthy_false_alarm_rate"] > rate:
            break
        best = op
    return best


def sweep(score: np.ndarray, rul: np.ndarray, unit: np.ndarray, n: int = 60, k: int = K) -> list:
    qs = 1 - np.logspace(-3.5, -0.3, n)
    thr = np.quantile(score[rul > HEALTHY], qs)
    pts = [operating_point(score, rul, unit, float(t), k) for t in np.unique(thr)]
    return [{"healthy_false_alarm_rate": p["healthy_false_alarm_rate"],
             "lead_median": p["lead_median"], "n_missed": p["n_missed"]} for p in pts]


def paired_auc_difference(sa: np.ndarray, sb: np.ndarray, y: np.ndarray, unit: np.ndarray,
                          n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    engines = np.unique(unit)
    by = {u: np.flatnonzero(unit == u) for u in engines}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(engines, size=len(engines), replace=True)])
        diffs.append(_ranking(sa[idx], y[idx])[0] - _ranking(sb[idx], y[idx])[0])
    return {"difference": float(_ranking(sa, y)[0] - _ranking(sb, y)[0]),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def figure(curves: dict, out: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, pts in curves.items():
        x = [max(p["healthy_false_alarm_rate"], 1e-4) for p in pts]
        ax.plot(x, [p["lead_median"] or 0 for p in pts], marker="o", ms=2.5, lw=1.2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("false-alarm rate on healthy windows (RUL > 100), causal rule; 0 plotted at 1e-4")
    ax.set_ylabel("median lead time (cycles)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)


def main(raw_dir: str, checkpoint: str, out: str, fig: str, quantile: float, n_boot: int) -> None:
    ds, y, scores = compute_scores(raw_dir, checkpoint)
    rul, unit = ds.test.rul.astype(float), ds.test.unit
    names = [n for n in scores if n != "Cycle index"]
    ref = "LSTM autoencoder"

    q99 = {n: operating_point(scores[n][1], rul, unit, calibrate_threshold(scores[n][0], quantile))
           for n in names}
    matched = {f"{r:g}": {n: at_false_alarm_rate(scores[n][1], rul, unit, r) for n in names}
               for r in FA_LEVELS}
    res = {
        "k": K, "quantile": quantile, "n_boot": n_boot,
        "at_validation_quantile": q99,
        "at_validation_quantile_vs_ae": {n: sign_test(q99[n]["lead_per_engine"], q99[ref]["lead_per_engine"])
                                         for n in names if n != ref},
        "at_equal_false_alarm_rate": matched,
        "at_equal_false_alarm_rate_vs_ae": {
            r: {n: sign_test(m[n]["lead_per_engine"], m[ref]["lead_per_engine"]) for n in names if n != ref}
            for r, m in matched.items()},
        "paired_roc_auc_ae_minus": {n: paired_auc_difference(scores[ref][1], scores[n][1], y, unit, n_boot)
                                    for n in names if n != ref},
        "curves": {n: sweep(scores[n][1], rul, unit) for n in names},
    }
    for n in names:
        o = q99[n]
        print(f"  q{quantile:g} {n:<18} healthy FA {o['healthy_false_alarm_rate']:.4f}  grey {o['grey_zone_alarm_rate']:.3f}"
              f"  lead median {o['lead_median']}  missed {o['n_missed']}  engines w/ healthy alarm {o['engines_with_a_healthy_alarm']}")
    for n, v in res["at_validation_quantile_vs_ae"].items():
        print(f"  q{quantile:g} {n} vs AE: {v}")
    for r, m in matched.items():
        print(f"  FA <= {r}: " + "  ".join(f"{n} {m[n]['lead_median']} (miss {m[n]['n_missed']})" for n in names))
        for n, v in res["at_equal_false_alarm_rate_vs_ae"][r].items():
            print(f"      {n} vs AE: {v}")
    for n, v in res["paired_roc_auc_ae_minus"].items():
        print(f"  ROC-AUC AE - {n}: {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    figure(res["curves"], fig)
    print(f"[ok] {out}, {fig}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--checkpoint", default="checkpoints/autoencoder.pt")
    p.add_argument("--out", default="data/alarm_rule.json")
    p.add_argument("--figure", default="figures/lead_vs_false_alarms.png")
    p.add_argument("--quantile", type=float, default=0.99)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    main(a.raw_dir, a.checkpoint, a.out, a.figure, a.quantile, a.n_boot)

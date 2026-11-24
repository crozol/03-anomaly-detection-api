"""Paired lead-time differences at equal healthy false-alarm rates, every score against the
autoencoder, with the multiplicity of the rates declared.

scripts/alarm_rule.py compares the scores at equal
healthy false-alarm rates with a sign test only; at three rates (0, 0.5 % and 1 %) the One-Class
SVM alarms first in more engines, while the autoencoder's median lead is longer. This script adds,
at every rate of data/alarm_rule.json, the per-engine difference the README already reports
against the Isolation Forest at the validation threshold (scripts/baselines.py,
paired_lead_difference): autoencoder lead minus the other score's lead, over the engines alarmed
by both, its median with a 95 % bootstrap interval over engines (10 000 resamples), and the exact
sign test. The three rates the README discusses are one family: their sign-test p-values are also
given with Holm's correction.

Since 2026-11-20, the interval the README reports is the distribution-free
one that matches the sign test (scripts/baselines.py, sign_test_interval): at 95 % and, for the
three rates of the family, at the level of the first step of Holm's correction, 1 - 0.05 / 3. A
percentile bootstrap of the median is not reliable with 20 discrete differences: with no false
alarm it excluded zero while the sign test did not reject. The bootstrap stays in the output.

    python -m scripts.equal_false_alarm_pairs     # reads data/alarm_rule.json -> data/equal_false_alarm_pairs.json
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.baselines import paired_lead_difference, sign_test_interval

REF = "LSTM autoencoder"
FAMILY = ("0", "0.005", "0.01")            # the three rates the README compares


def holm(pvalues: dict) -> dict:
    """Holm's step-down adjustment of a family of p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def main(src: str = "data/alarm_rule.json", out: str = "data/equal_false_alarm_pairs.json") -> dict:
    matched = json.loads(Path(src).read_text())["at_equal_false_alarm_rate"]
    res = {"reference": REF, "family": list(FAMILY), "rates": {}}
    for rate, scores in matched.items():
        res["rates"][rate] = {}
        for name, op in scores.items():
            if name == REF:
                continue
            a, b = scores[REF]["lead_per_engine"], op["lead_per_engine"]
            d = paired_lead_difference(a, b)
            if rate in FAMILY:                 # the engines paired_lead_difference compares
                diffs = [a[u] - b[u] for u in a if a[u] is not None and b.get(u) is not None]
                d["median_ci_sign_holm"] = sign_test_interval(diffs, level=1 - 0.05 / len(FAMILY))
            d["missed"] = {REF: scores[REF]["n_missed"], name: op["n_missed"]}
            d["lead_median"] = {REF: scores[REF]["lead_median"], name: op["lead_median"]}
            res["rates"][rate][name] = d
    for name in {n for r in res["rates"].values() for n in r}:
        adj = holm({r: res["rates"][r][name]["sign_test_p"] for r in FAMILY if r in res["rates"]})
        for r, p in adj.items():
            res["rates"][r][name]["sign_test_p_holm"] = p
    Path(out).write_text(json.dumps(res, indent=2))
    def interval(ci: dict | None) -> str:
        if not ci or ci["interval"] is None:
            return "n/a"
        return f"[{ci['interval'][0]:+.1f}, {ci['interval'][1]:+.1f}] at {100 * ci['coverage']:.1f} %"

    for rate, scores in res["rates"].items():
        for name, d in scores.items():
            holm_p = d.get("sign_test_p_holm")
            print(f"FA <= {rate:>5}  AE - {name:<16} median {d['median']:+.1f} "
                  f"sign-test interval {interval(d['median_ci_sign'])}"
                  + (f", Holm level {interval(d['median_ci_sign_holm'])}" if "median_ci_sign_holm" in d else "")
                  + f"; bootstrap [{d['median_ci95'][0]:+.1f}, {d['median_ci95'][1]:+.1f}] over {d['n_engines']} engines; "
                  f"AE first {d['a_earlier']}, other first {d['b_earlier']}, ties {d['ties']}; "
                  f"sign p {d['sign_test_p']:.3f}" + (f" (Holm {holm_p:.3f})" if holm_p is not None else "")
                  + f"; missed AE {d['missed'][REF]}, other {d['missed'][name]}")
    print(f"[ok] {out}")
    return res


if __name__ == "__main__":
    main()

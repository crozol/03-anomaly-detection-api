"""Lead time per engine and the paired comparison between two scores."""

from __future__ import annotations

import numpy as np

from scripts.baselines import lead_times, paired_lead_difference


def test_lead_time_is_rul_at_persistent_alarm() -> None:
    rul = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    unit = np.zeros(6, dtype=int)
    # crosses at RUL 4, drops back, then stays above from RUL 2 on
    score = np.array([0, 1, 0, 1, 1, 1], dtype=float)
    out = lead_times(score, rul, unit, threshold=0.5)
    assert out["per_engine"] == {0: 2.0}
    assert out["n_missed"] == 0


def test_paired_difference_and_sign_test() -> None:
    a = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: None}
    b = {0: 5.0, 1: 15.0, 2: 30.0, 3: 45.0, 4: 40.0, 5: 7.0}
    out = paired_lead_difference(a, b, n_boot=200)
    # engine 5 is missed by a and dropped; differences 5, 5, 0, -5, 10
    assert out["n_engines"] == 5
    assert out["median"] == 5.0
    assert (out["a_earlier"], out["b_earlier"], out["ties"]) == (3, 1, 1)
    # exact two-sided sign test, n = 4, k = 1: 2 * (1 + 4) / 16
    assert abs(out["sign_test_p"] - 10 / 16) < 1e-12

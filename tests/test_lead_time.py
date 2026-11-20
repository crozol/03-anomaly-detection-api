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
    # five differences cannot give a 95 % interval matching the sign test
    assert out["median_ci_sign"]["interval"] is None


def test_sign_test_interval_uses_the_order_statistics_of_the_sign_test() -> None:
    from scripts.baselines import sign_test_interval
    d = np.arange(20, 0, -1, dtype=float)              # 20 distinct differences, unsorted
    out = sign_test_interval(d)
    # P(B <= 5) = 21700 / 2^20 for B ~ Binomial(20, 1/2): [d(6), d(15)] covers 95.9 %
    assert out["k"] == 6 and out["interval"] == [6.0, 15.0]
    assert abs(out["coverage"] - (1 - 2 * 21700 / 2 ** 20)) < 1e-12
    holm = sign_test_interval(d, level=1 - 0.05 / 3)  # first step of Holm over three tests
    assert holm["k"] == 5 and holm["interval"] == [5.0, 16.0]
    # 14 negative differences, one tie and 5 positive: the interval reaches the tie at zero
    d = np.r_[-np.arange(14, 0, -1), 0, np.arange(1, 6)].astype(float)
    assert sign_test_interval(d)["interval"] == [-9.0, 0.0]
    assert sign_test_interval([1.0, 2.0])["interval"] is None


def test_causal_alarm_needs_k_consecutive_windows_and_no_look_ahead() -> None:
    from scripts.alarm_rule import alarm_state, operating_point
    score = np.array([1, 0, 1, 1, 1, 0, 1], dtype=float)
    # alarm is known at the third consecutive window above, not before
    assert alarm_state(score, 0.5, k=3).tolist() == [False, False, False, False, True, False, False]
    # one engine, RUL 140 .. 20 in steps of 20: 3 windows above inside the healthy
    # regime (RUL > 100) charge the false-alarm rate; the lead counts from RUL <= 100
    rul = np.array([140, 120, 100, 80, 60, 40, 20], dtype=float)
    s = np.array([1, 1, 1, 0, 1, 1, 1], dtype=float)
    op = operating_point(s, rul, np.zeros(7, dtype=int), 0.5, k=3)
    assert op["lead_per_engine"] == {0: 100.0}
    assert op["healthy_false_alarm_rate"] == 0.0          # the alarm completes at RUL 100
    # an alarm completed inside the healthy regime is a false alarm, not a lead
    rul2 = np.array([180, 160, 140, 120, 100, 80], dtype=float)
    s2 = np.array([1, 1, 1, 0, 0, 1], dtype=float)
    op2 = operating_point(s2, rul2, np.zeros(6, dtype=int), 0.5, k=3)
    assert op2["healthy_false_alarm_rate"] == 0.25 and op2["n_missed"] == 1


def test_holm_adjustment_is_step_down_and_monotone():
    import pytest

    from scripts.equal_false_alarm_pairs import holm

    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)                  # 3 x 0.01
    assert adj["c"] == pytest.approx(0.06)                  # 2 x 0.03
    assert adj["b"] == pytest.approx(0.06)                  # max(1 x 0.04, 0.06): monotone
    assert holm({"x": 0.6, "y": 0.7})["y"] == pytest.approx(1.0)

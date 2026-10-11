"""The hand-written ranking metrics must agree with scikit-learn.

ROC-AUC and average precision are the headline numbers of the project, so
they are checked against the reference implementation on random scores,
including heavy ties (rounded scores), to 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import auc, average_precision, roc_curve


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("decimals", [None, 1])
def test_metrics_match_sklearn(seed: int, decimals: int | None) -> None:
    rng = np.random.default_rng(seed)
    y = (rng.random(2000) < 0.16).astype(np.int32)
    score = rng.normal(size=2000) + 1.5 * y
    if decimals is not None:
        score = np.round(score, decimals)   # many tied scores

    fpr, tpr, _ = roc_curve(score, y)
    assert auc(fpr, tpr) == pytest.approx(roc_auc_score(y, score), abs=1e-9)
    assert average_precision(score, y) == pytest.approx(
        average_precision_score(y, score), abs=1e-9
    )

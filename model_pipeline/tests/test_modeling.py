from __future__ import annotations

import pandas as pd

from modeling import build_lgbm_params, metric_summary


def test_build_lgbm_params_applies_overrides() -> None:
    params = build_lgbm_params({"n_estimators": 123, "learning_rate": 0.07}, random_state=99)
    assert params["n_estimators"] == 123
    assert params["learning_rate"] == 0.07
    assert params["random_state"] == 99


def test_metric_summary_returns_expected_keys() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = metric_summary(y_true, scores)
    assert set(metrics) == {"roc_auc", "pr_auc", "ks", "precision_at_0_5", "recall_at_0_5"}

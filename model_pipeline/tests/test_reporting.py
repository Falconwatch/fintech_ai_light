from __future__ import annotations

from pathlib import Path

import pandas as pd

from reporting import build_report_context, load_report_template, render_report


def test_report_uses_relative_figure_paths() -> None:
    template = load_report_template(Path("templates/model_report.md.j2"))
    train_df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1],
            "MonthlyIncome": [1000.0, 2000.0],
            "NumberOfDependents": [0.0, 1.0],
            "age": [30, 40],
        }
    )
    test_df = train_df.copy()
    dev_df = train_df.copy()
    oot_df = train_df.copy()
    shap_sample = pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [1.0, 2.0]})
    context = build_report_context(
        train_df=train_df,
        test_df=test_df,
        dev_df=dev_df,
        oot_df=oot_df,
        feature_descriptions=[{"feature": "feature_a", "description": "demo"}],
        cv_scores=pd.DataFrame([{"fold": 1, "roc_auc": 0.8, "pr_auc": 0.4, "ks": 0.3}]),
        dev_metrics={"roc_auc": 0.8, "pr_auc": 0.4, "ks": 0.3, "precision_at_0_5": 0.5, "recall_at_0_5": 0.6},
        oot_metrics={"roc_auc": 0.7, "pr_auc": 0.3, "ks": 0.2, "precision_at_0_5": 0.4, "recall_at_0_5": 0.5},
        shap_sample=shap_sample,
        shap_values=[[0.1, 0.2], [0.3, 0.4]],
        best_params={"n_estimators": 100},
        best_cv_score=0.8,
        error_case_one="case one",
        error_case_two="case two",
        figure_paths={
            "target_distribution": "figures/target_distribution.png",
            "missing_values": "figures/missing_values.png",
            "numeric_distributions": "figures/numeric_distributions.png",
            "outlier_boxplots": "figures/outlier_boxplots.png",
            "numeric_distributions_clipped": "figures/numeric_distributions_clipped.png",
            "correlation_heatmap": "figures/correlation_heatmap.png",
            "cv_scores": "figures/cv_scores.png",
            "score_distribution": "figures/score_distribution.png",
            "feature_importance": "figures/feature_importance.png",
            "shap_summary": "figures/shap_summary.png",
        },
        cv_folds=5,
        optuna_trials=5,
    )
    report = render_report(template, context)
    assert "![Target distribution](figures/target_distribution.png)" in report

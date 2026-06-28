from __future__ import annotations

import pandas as pd

from features import build_features, fit_imputation_reference, prepare_model_matrix


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3, 4],
            "SeriousDlqin2yrs": [0, 1, 0, 1],
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.9, 0.3, 0.7],
            "age": [30, 45, 60, 35],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 0, 2],
            "DebtRatio": [0.2, 0.5, 0.7, 0.4],
            "MonthlyIncome": [3000.0, None, 8000.0, 4500.0],
            "NumberOfOpenCreditLinesAndLoans": [4, 8, 10, 6],
            "NumberOfTimes90DaysLate": [0, 1, 0, 1],
            "NumberRealEstateLoansOrLines": [1, 2, 1, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 1],
            "NumberOfDependents": [1.0, None, 0.0, 2.0],
        }
    )


def test_build_features_fills_missing_and_adds_columns() -> None:
    df = _sample_df()
    reference = fit_imputation_reference(df.iloc[:3], [0, 25, 35, 45, 55, 65, 120])
    featured, descriptions = build_features(df, reference)

    assert featured["MonthlyIncome"].isna().sum() == 0
    assert featured["NumberOfDependents"].isna().sum() == 0
    assert "weighted_past_due_score" in featured.columns
    assert "income_per_dependent" in featured.columns
    assert len(descriptions) > 0


def test_prepare_model_matrix_removes_target_and_id() -> None:
    df = _sample_df()
    reference = fit_imputation_reference(df.iloc[:3], [0, 25, 35, 45, 55, 65, 120])
    dev_df, _ = build_features(df.iloc[:3], reference)
    oot_df, _ = build_features(df.iloc[3:], reference)

    X_dev, X_oot, feature_cols, _ = prepare_model_matrix(dev_df, oot_df)

    assert "SeriousDlqin2yrs" not in feature_cols
    assert "Unnamed: 0" not in feature_cols
    assert X_dev.isna().sum().sum() == 0
    assert X_oot.isna().sum().sum() == 0

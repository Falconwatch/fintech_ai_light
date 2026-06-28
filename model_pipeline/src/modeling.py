from __future__ import annotations

from dataclasses import dataclass

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class HyperparameterSearchResult:
    best_params: dict[str, float | int]
    best_score: float
    trials_df: pd.DataFrame


def build_lgbm_params(
    overrides: dict[str, float | int] | None = None,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    params: dict[str, float | int | str] = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_samples": 25,
        "min_child_weight": 1e-3,
        "reg_lambda": 2.5,
        "reg_alpha": 0.0,
        "objective": "binary",
        "random_state": random_state,
        "metric": "auc",
        "n_jobs": 4,
        "verbosity": -1,
    }
    if overrides:
        params.update(overrides)
    return params


def fit_cv_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_params: dict[str, float | int | str] | None = None,
) -> tuple[pd.DataFrame, list[LGBMClassifier]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    models: list[LGBMClassifier] = []

    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    base_params = build_lgbm_params(model_params)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        fold_params = dict(base_params)
        fold_params["random_state"] = 42 + fold
        fold_params["scale_pos_weight"] = scale_pos_weight
        model = LGBMClassifier(**fold_params)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_valid)[:, 1]
        rows.append(
            {
                "fold": fold,
                "roc_auc": roc_auc_score(y_valid, preds),
                "pr_auc": average_precision_score(y_valid, preds),
                "ks": ks_2samp(preds[y_valid == 0], preds[y_valid == 1]).statistic,
            }
        )
        models.append(model)
    return pd.DataFrame(rows), models


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict[str, float | int | str] | None = None,
) -> LGBMClassifier:
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    final_params = build_lgbm_params(model_params, random_state=42)
    final_params["scale_pos_weight"] = scale_pos_weight
    model = LGBMClassifier(**final_params)
    model.fit(X, y)
    return model


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
    }


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_trials: int,
) -> HyperparameterSearchResult:
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_lgbm_params(trial)
        cv_scores, _ = fit_cv_model(X, y, n_splits=n_splits, model_params=trial_params)
        return float(cv_scores["roc_auc"].mean())

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = study.trials_dataframe()
    return HyperparameterSearchResult(
        best_params={key: study.best_params[key] for key in study.best_params},
        best_score=float(study.best_value),
        trials_df=trials_df,
    )


def metric_summary(y_true: pd.Series, score: pd.Series | list[float] | tuple[float, ...]) -> dict[str, float]:
    score_series = pd.Series(score, index=y_true.index if hasattr(y_true, "index") else None)
    preds = (score_series >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, score_series),
        "pr_auc": average_precision_score(y_true, score_series),
        "ks": ks_2samp(score_series[y_true == 0], score_series[y_true == 1]).statistic,
        "precision_at_0_5": tp / max(tp + fp, 1),
        "recall_at_0_5": tp / max(tp + fn, 1),
    }

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import shap

from model_pipeline.src.config import DEFAULT_OPTUNA_TRIALS, ID_COL, PipelineConfig, TARGET
from model_pipeline.src.features import build_features, clip_outliers, fit_imputation_reference, prepare_model_matrix
from model_pipeline.src.modeling import fit_cv_model, fit_final_model, metric_summary, tune_hyperparameters
from model_pipeline.src.plotting import save_figures
from model_pipeline.src.reporting import (
    build_error_cases,
    build_report_context,
    compute_binary_shap_values,
    load_report_template,
    print_best_params,
    print_metrics_table,
    render_report,
    save_run_metadata,
)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [train_pipeline] {message}", flush=True)


def error(message: str) -> None:
    print(f"\033[31m{message}\033[0m", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    reports = output_dir / "reports"
    figures = reports / "figures"
    data = output_dir / "data"
    models = output_dir / "models"
    for path in (figures, reports, data, models):
        path.mkdir(parents=True, exist_ok=True)
    return {"figures": figures, "reports": reports, "data": data, "models": models}


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_input_files(train_path: Path, test_path: Path) -> None:
    if train_path.exists() and test_path.exists():
        return
    expected_dir = train_path.parent
    error(
        "Упс, кажется вы не скачали данные для обучения. Для работы пайплайна скачайте данные с "
        "https://www.kaggle.com/c/GiveMeSomeCredit/data и разместите их в "
        f"{expected_dir}/ так, чтобы там лежали файлы cs-training.csv и cs-test.csv. "
        "Если вы запускаете через run_pipeline.sh, можно также передать внешний путь через "
        "--data-dir /path/to/GiveMeSomeCredit",
    )
    raise SystemExit(1)


def split_train_oot(train_df: pd.DataFrame, dev_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(train_df) * dev_share)
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def main() -> None:
    started_at = perf_counter()
    args = parse_args()
    config = PipelineConfig(optuna_trials=args.optuna_trials)

    output_dir = Path(args.output_dir)
    template_path = Path(__file__).resolve().parent.parent / "templates" / "model_report.md.j2"

    log(f"Подготавливаю директории артефактов в {output_dir}")
    paths = ensure_dirs(output_dir)

    log("Загружаю train и test датасеты")
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    validate_input_files(train_path, test_path)
    train_df, test_df = load_data(train_path, test_path)
    log(f"Train shape: {train_df.shape}, test shape: {test_df.shape}")

    dev_raw, oot_raw = split_train_oot(train_df, config.dev_share)

    log("Готовлю reference-статистики для импутации по development")
    imputation_reference = fit_imputation_reference(dev_raw, config.age_bins)

    log("Выполняю генерацию признаков для development, OOT и test")
    dev_df, feature_descriptions = build_features(dev_raw, imputation_reference)
    oot_df, _ = build_features(oot_raw, imputation_reference)
    enriched_test, _ = build_features(test_df, imputation_reference)

    log("Формирую модельные матрицы и выполняю финальную импутацию")
    X_dev, X_oot, feature_cols, imputer = prepare_model_matrix(dev_df, oot_df)
    y_dev = dev_df[TARGET].astype(int)
    y_oot = oot_df[TARGET].astype(int)
    log(f"Количество признаков после feature engineering: {len(feature_cols)}")

    log(f"Запускаю Optuna-подбор гиперпараметров на development, trials={config.optuna_trials}")
    tuning_result = tune_hyperparameters(
        X_dev,
        y_dev,
        n_splits=config.cv_folds,
        n_trials=config.optuna_trials,
    )
    print_best_params(tuning_result.best_params, tuning_result.best_score, log)

    log("Пересчитываю cross-validation на development с лучшими гиперпараметрами")
    cv_scores, _ = fit_cv_model(
        X_dev,
        y_dev,
        n_splits=config.cv_folds,
        model_params=tuning_result.best_params,
    )

    log("Обучаю финальную модель на development с лучшими гиперпараметрами")
    model = fit_final_model(X_dev, y_dev, model_params=tuning_result.best_params)

    log("Считаю предсказания и метрики на development, OOT и test")
    dev_scores = model.predict_proba(X_dev)[:, 1]
    oot_scores = model.predict_proba(X_oot)[:, 1]
    clipped_test = clip_outliers(dev_df, enriched_test, feature_cols)
    X_test = pd.DataFrame(
        imputer.transform(clipped_test[feature_cols]),
        columns=feature_cols,
        index=clipped_test.index,
    )
    test_scores = model.predict_proba(X_test)[:, 1]

    dev_metrics = metric_summary(y_dev, dev_scores)
    oot_metrics = metric_summary(y_oot, oot_scores)

    log("Считаю feature importance")
    feature_importance = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    sample_size = min(5000, len(X_oot))
    log(f"Запускаю SHAP-анализ на сэмпле OOT размером {sample_size}")
    shap_sample = X_oot.sample(sample_size, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = compute_binary_shap_values(explainer, shap_sample)

    log("Строю графики для отчета")
    figure_paths = save_figures(
        train_df=train_df,
        dev_scores=cv_scores,
        y_oot=y_oot.loc[shap_sample.index],
        oot_scores=model.predict_proba(shap_sample)[:, 1],
        feature_importance=feature_importance,
        shap_values=shap_values,
        X_oot=shap_sample,
        figure_dir=paths["figures"],
        report_dir=paths["reports"],
    )

    log("Разбираю ошибки модели на OOT")
    error_case_one, error_case_two = build_error_cases(
        oot_raw=oot_df,
        X_oot=X_oot,
        y_oot=y_oot,
        oot_scores=oot_scores,
        explainer=explainer,
    )

    log("Собираю markdown-отчет")
    template = load_report_template(template_path)
    report_context = build_report_context(
        train_df=train_df,
        test_df=test_df,
        dev_df=dev_df,
        oot_df=oot_df,
        feature_descriptions=feature_descriptions,
        cv_scores=cv_scores,
        dev_metrics=dev_metrics,
        oot_metrics=oot_metrics,
        shap_sample=shap_sample,
        shap_values=shap_values,
        best_params=tuning_result.best_params,
        best_cv_score=tuning_result.best_score,
        error_case_one=error_case_one,
        error_case_two=error_case_two,
        figure_paths=figure_paths,
        cv_folds=config.cv_folds,
        optuna_trials=config.optuna_trials,
    )
    report = render_report(template, report_context)

    log("Сохраняю отчет, таблицы, предсказания и модель")
    (paths["reports"] / "model_report.md").write_text(report, encoding="utf-8")
    cv_scores.to_csv(paths["data"] / "cv_scores.csv", index=False)
    feature_importance.to_csv(paths["data"] / "feature_importance.csv", index=False)
    pd.DataFrame({ID_COL: test_df[ID_COL], "score": test_scores}).to_csv(
        paths["data"] / "test_predictions.csv",
        index=False,
    )
    save_run_metadata(
        paths["data"],
        best_params=tuning_result.best_params,
        dev_metrics=dev_metrics,
        oot_metrics=oot_metrics,
        optuna_trials_df=tuning_result.trials_df,
        run_config={
            "train_path": str(Path(args.train_path)),
            "test_path": str(Path(args.test_path)),
            "output_dir": str(output_dir),
            "optuna_trials": config.optuna_trials,
            "cv_folds": config.cv_folds,
            "dev_share": config.dev_share,
            "feature_count": len(feature_cols),
        },
    )
    model.booster_.save_model((paths["models"] / "lightgbm_model.txt").as_posix())

    print_metrics_table(dev_metrics, oot_metrics, log)
    elapsed = perf_counter() - started_at
    log(f"Готово. Общее время выполнения: {elapsed:.1f} сек.")


if __name__ == "__main__":
    main()

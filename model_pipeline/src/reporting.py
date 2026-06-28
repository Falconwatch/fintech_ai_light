from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from jinja2 import Template

from config import ID_COL, TARGET


def load_report_template(template_path: Path) -> Template:
    return Template(template_path.read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_empty_"
    render = df.copy()
    for col in render.select_dtypes(include=["float", "float64", "float32"]).columns:
        render[col] = render[col].map(lambda x: format(x, floatfmt))
    return render.to_markdown(index=index)


def summarize_eda(train_df: pd.DataFrame) -> str:
    target_rate = train_df[TARGET].mean()
    missing_income = train_df["MonthlyIncome"].isna().mean()
    missing_dep = train_df["NumberOfDependents"].isna().mean()
    age_desc = train_df["age"].describe()[["mean", "50%", "min", "max"]]
    return (
        f"- Доля дефолтов в train составляет `{target_rate:.2%}`, что указывает на выраженный дисбаланс классов.\n"
        f"- Основные пропуски сосредоточены в `MonthlyIncome` (`{missing_income:.2%}`) и `NumberOfDependents` (`{missing_dep:.2%}`).\n"
        f"- Возрастная структура умеренно широкая: среднее `{age_desc['mean']:.1f}`, медиана `{age_desc['50%']:.1f}`, диапазон `{age_desc['min']:.0f}`-`{age_desc['max']:.0f}`.\n"
        f"- Признаки delinquency, `DebtRatio`, `MonthlyIncome` и utilization имеют тяжелые хвосты, поэтому в EDA добавлен отдельный анализ выбросов.\n"
        f"- Для сравнения распределений до и после обработки строятся boxplot'ы и гистограммы после winsorization по 1/99 процентилям."
    )


def describe_imputation() -> str:
    return (
        "- `MonthlyIncome` заполняется медианой внутри возрастного бина, затем глобальной медианой как резервным вариантом.\n"
        "- `NumberOfDependents` заполняется по той же схеме: медиана возрастного бина, затем глобальная медиана.\n"
        "- Для обеих колонок сохраняются индикаторы пропуска (`has_missing_income`, `has_missing_dependents`), чтобы модель могла использовать сам факт отсутствия значения."
    )


def compute_binary_shap_values(explainer: shap.TreeExplainer, features: pd.DataFrame) -> np.ndarray:
    explanation = explainer(features)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        # Binary classifiers may expose contributions for both classes.
        # We use the positive-class explanation for scoring interpretation.
        return values[:, :, 1]
    if values.ndim == 2:
        return values
    if values.ndim == 1:
        return values.reshape(1, -1)
    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def build_error_cases(
    oot_raw: pd.DataFrame,
    X_oot: pd.DataFrame,
    y_oot: pd.Series,
    oot_scores: np.ndarray,
    explainer: shap.TreeExplainer,
) -> tuple[str, str]:
    analysis = oot_raw.copy()
    analysis["score"] = oot_scores
    analysis["pred"] = (oot_scores >= 0.5).astype(int)
    analysis["actual"] = y_oot.values
    analysis["error_type"] = np.where(
        (analysis["pred"] == 1) & (analysis["actual"] == 0),
        "false_positive",
        np.where((analysis["pred"] == 0) & (analysis["actual"] == 1), "false_negative", "correct"),
    )

    cases = []
    for err in ("false_positive", "false_negative"):
        subset = analysis[analysis["error_type"] == err].copy()
        if subset.empty:
            cases.append(f"Для типа ошибки `{err}` наблюдений не найдено при пороге 0.5.")
            continue
        subset["confidence_gap"] = np.abs(subset["score"] - subset["actual"])
        case_idx = subset.sort_values("confidence_gap", ascending=False).index[0]
        row = analysis.loc[case_idx]
        shap_row_values = compute_binary_shap_values(explainer, X_oot.loc[[case_idx]])
        shap_row = pd.Series(shap_row_values[0], index=X_oot.columns)
        top_positive = shap_row.sort_values(ascending=False).head(3)
        top_negative = shap_row.sort_values().head(3)
        text = (
            f"ID `{int(row[ID_COL])}`: фактический класс `{int(row['actual'])}`, прогноз `{int(row['pred'])}`, score `{row['score']:.4f}`. "
            f"Ключевые факторы в сторону дефолта: "
            + ", ".join([f"`{k}` ({v:.3f})" for k, v in top_positive.items()])
            + ". Факторы против дефолта: "
            + ", ".join([f"`{k}` ({v:.3f})" for k, v in top_negative.items()])
            + ". "
        )
        if err == "false_positive":
            text += "Интерпретация: модель увидела сильный риск по паттерну просрочек/нагрузки, но фактического дефолта не произошло; это похоже на консервативную переоценку риска."
        else:
            text += "Интерпретация: по наблюдаемым признакам клиент выглядел относительно стабильным, однако дефолт произошел; вероятно, в данных не хватает внешних факторов риска."
        cases.append(text)
    return cases[0], cases[1]


def build_report_context(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_descriptions: list[dict[str, str]],
    cv_scores: pd.DataFrame,
    dev_metrics: dict[str, float],
    oot_metrics: dict[str, float],
    shap_sample: pd.DataFrame,
    shap_values: np.ndarray,
    best_params: dict[str, float | int],
    best_cv_score: float,
    error_case_one: str,
    error_case_two: str,
    figure_paths: dict[str, str],
    cv_folds: int,
    optuna_trials: int,
) -> dict[str, object]:
    feature_table = markdown_table(pd.DataFrame(feature_descriptions), index=False)
    missing_table = markdown_table(
        train_df.isna().mean().sort_values(ascending=False).reset_index().rename(
            columns={"index": "feature", 0: "missing_share"}
        ),
        index=False,
    )
    cv_table = markdown_table(cv_scores, index=False)
    final_metrics_table = markdown_table(
        pd.DataFrame(
            [
                {"sample": "development", **dev_metrics},
                {"sample": "oot", **oot_metrics},
            ]
        ),
        index=False,
    )
    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_summary = markdown_table(
        pd.DataFrame({"feature": shap_sample.columns, "mean_abs_shap": shap_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15),
        index=False,
    )
    best_params_table = markdown_table(
        pd.DataFrame([{"parameter": key, "value": value} for key, value in sorted(best_params.items())]),
        index=False,
    )
    return {
        "target": TARGET,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "target_rate": f"{train_df[TARGET].mean():.2%}",
        "dev_share": f"{len(dev_df) / len(train_df):.2%}",
        "oot_share": f"{len(oot_df) / len(train_df):.2%}",
        "dev_rows": len(dev_df),
        "oot_rows": len(oot_df),
        "missing_table": missing_table,
        "eda_summary": summarize_eda(train_df),
        "feature_table": feature_table,
        "imputation_summary": describe_imputation(),
        "cv_folds": cv_folds,
        "optuna_trials": optuna_trials,
        "best_params_table": best_params_table,
        "best_cv_score": f"{best_cv_score:.4f}",
        "cv_table": cv_table,
        "final_metrics_table": final_metrics_table,
        "shap_summary": shap_summary,
        "error_case_one": error_case_one,
        "error_case_two": error_case_two,
        "conclusion": (
            "Модель демонстрирует устойчивое разделение классов на development и сохраняет качество на псевдо-OOT. "
            "При этом интерпретация ограничена отсутствием реальной временной оси в исходном датасете, поэтому для production-валидации стоит повторить анализ на данных с настоящей датой наблюдения."
        ),
        "figures": figure_paths,
    }


def render_report(template: Template, context: dict[str, object]) -> str:
    return template.render(**context)


def print_metrics_table(dev_metrics: dict[str, float], oot_metrics: dict[str, float], log) -> None:
    metrics_df = pd.DataFrame(
        [
            {"sample": "development", **dev_metrics},
            {"sample": "oot", **oot_metrics},
        ]
    )
    log("Итоговые метрики:")
    print(markdown_table(metrics_df, index=False), flush=True)


def print_best_params(best_params: dict[str, float | int], best_cv_score: float, log) -> None:
    params_df = pd.DataFrame([{"parameter": key, "value": value} for key, value in sorted(best_params.items())])
    log(f"Лучшие гиперпараметры Optuna. Лучший mean CV ROC-AUC: {best_cv_score:.4f}")
    print(params_df.to_string(index=False), flush=True)


def save_run_metadata(
    data_dir: Path,
    *,
    best_params: dict[str, float | int],
    dev_metrics: dict[str, float],
    oot_metrics: dict[str, float],
    optuna_trials_df: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    (data_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (data_dir / "metrics.json").write_text(
        json.dumps({"development": dev_metrics, "oot": oot_metrics}, indent=2),
        encoding="utf-8",
    )
    (data_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    optuna_trials_df.to_csv(data_dir / "optuna_trials.csv", index=False)

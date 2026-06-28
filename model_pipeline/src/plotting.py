from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import roc_curve

from config import BASE_FEATURES, EDA_NUMERIC_COLS, TARGET
from features import clip_outliers


def _save_figure(fig: plt.Figure, path: Path, report_dir: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.relative_to(report_dir).as_posix()


def plot_target_distribution(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    palette = ["#7aa6c2", "#d56b6b"]
    fig, ax = plt.subplots(figsize=(6, 4))
    target_share = train_df[TARGET].value_counts(normalize=True).sort_index()
    ax.pie(
        target_share.values,
        labels=["Нет просрочки", "Серьезная просрочка"],
        autopct="%1.1f%%",
        startangle=90,
        colors=palette,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Распределение целевой переменной")
    ax.axis("equal")
    return _save_figure(fig, figure_dir / "target_distribution.png", report_dir)


def plot_missing_values(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_share = train_df.isna().mean().sort_values(ascending=False)
    missing_share[missing_share > 0].plot(kind="bar", ax=ax, color="#d8a657")
    ax.set_title("Доля пропусков по признакам")
    ax.set_xlabel("Признак")
    ax.set_ylabel("Доля пропусков")
    return _save_figure(fig, figure_dir / "missing_values.png", report_dir)


def plot_numeric_distributions(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), EDA_NUMERIC_COLS):
        sns.histplot(train_df[col], kde=False, bins=40, ax=ax, color="#6c8ebf")
        ax.set_title(f"{col} (исходное)")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Количество наблюдений")
    return _save_figure(fig, figure_dir / "numeric_distributions.png", report_dir)


def plot_outlier_boxplots(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), EDA_NUMERIC_COLS):
        sns.boxplot(x=train_df[col], ax=ax, color="#d8a657")
        ax.set_title(f"{col}: выбросы")
        ax.set_xlabel("Значение")
    return _save_figure(fig, figure_dir / "outlier_boxplots.png", report_dir)


def plot_numeric_distributions_clipped(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    clipped_numeric = clip_outliers(train_df, train_df, EDA_NUMERIC_COLS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), EDA_NUMERIC_COLS):
        sns.histplot(clipped_numeric[col], kde=False, bins=40, ax=ax, color="#4c956c")
        ax.set_title(f"{col} (без выбросов)")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Количество наблюдений")
    return _save_figure(fig, figure_dir / "numeric_distributions_clipped.png", report_dir)


def plot_correlation_heatmap(train_df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = train_df[[TARGET] + BASE_FEATURES].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Тепловая карта корреляций")
    return _save_figure(fig, figure_dir / "correlation_heatmap.png", report_dir)


def plot_cv_scores(cv_scores: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=cv_scores, x="fold", y="roc_auc", color="#7aa6c2", ax=ax)
    ax.axhline(cv_scores["roc_auc"].mean(), color="#d56b6b", linestyle="--", label="mean")
    ax.set_title("ROC-AUC по фолдам CV")
    ax.set_xlabel("Фолд")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    return _save_figure(fig, figure_dir / "cv_scores.png", report_dir)


def plot_score_distribution(y_oot: pd.Series, oot_scores, figure_dir: Path, report_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(x=oot_scores[y_oot == 0], label="OOT: нет просрочки", ax=ax, color="#7aa6c2")
    sns.kdeplot(x=oot_scores[y_oot == 1], label="OOT: серьезная просрочка", ax=ax, color="#d56b6b")
    ax.set_title("Распределение скорингов на OOT")
    ax.set_xlabel("Скор модели")
    ax.set_ylabel("Плотность")
    ax.legend()
    return _save_figure(fig, figure_dir / "score_distribution.png", report_dir)


def plot_roc_curve(y_oot: pd.Series, oot_scores, figure_dir: Path, report_dir: Path) -> str:
    fpr, tpr, _ = roc_curve(y_oot, oot_scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#6c8ebf", linewidth=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Случайная модель")
    ax.set_title("ROC-кривая на OOT")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return _save_figure(fig, figure_dir / "roc_curve.png", report_dir)


def plot_feature_importance(feature_importance: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    top_imp = feature_importance.head(15)
    sns.barplot(data=top_imp, x="importance", y="feature", color="#6c8ebf", ax=ax)
    ax.set_title("Топ признаков по важности")
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признак")
    return _save_figure(fig, figure_dir / "feature_importance.png", report_dir)


def plot_shap_summary(shap_values, X_oot: pd.DataFrame, figure_dir: Path, report_dir: Path) -> str:
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_oot, show=False, max_display=15)
    plt.title("SHAP summary")
    path = figure_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path.relative_to(report_dir).as_posix()


def save_figures(
    train_df: pd.DataFrame,
    dev_scores: pd.DataFrame,
    y_oot: pd.Series,
    oot_scores,
    feature_importance: pd.DataFrame,
    shap_values,
    X_oot: pd.DataFrame,
    figure_dir: Path,
    report_dir: Path,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid")
    return {
        "target_distribution": plot_target_distribution(train_df, figure_dir, report_dir),
        "missing_values": plot_missing_values(train_df, figure_dir, report_dir),
        "numeric_distributions": plot_numeric_distributions(train_df, figure_dir, report_dir),
        "outlier_boxplots": plot_outlier_boxplots(train_df, figure_dir, report_dir),
        "numeric_distributions_clipped": plot_numeric_distributions_clipped(train_df, figure_dir, report_dir),
        "correlation_heatmap": plot_correlation_heatmap(train_df, figure_dir, report_dir),
        "cv_scores": plot_cv_scores(dev_scores, figure_dir, report_dir),
        "score_distribution": plot_score_distribution(y_oot, oot_scores, figure_dir, report_dir),
        "roc_curve": plot_roc_curve(y_oot, oot_scores, figure_dir, report_dir),
        "feature_importance": plot_feature_importance(feature_importance, figure_dir, report_dir),
        "shap_summary": plot_shap_summary(shap_values, X_oot, figure_dir, report_dir),
    }

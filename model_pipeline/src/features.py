from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer

from config import ID_COL, TARGET


@dataclass(frozen=True)
class ImputationReference:
    age_bins: list[int]
    income_by_age: dict
    dependents_by_age: dict
    income_global_median: float
    dependents_global_median: float


def fit_imputation_reference(dev_df: pd.DataFrame, age_bins: list[int]) -> ImputationReference:
    age_cuts = pd.cut(dev_df["age"], bins=age_bins)
    income_by_age = dev_df.groupby(age_cuts, observed=False)["MonthlyIncome"].median().to_dict()
    dependents_by_age = dev_df.groupby(age_cuts, observed=False)["NumberOfDependents"].median().to_dict()
    return ImputationReference(
        age_bins=age_bins,
        income_by_age=income_by_age,
        dependents_by_age=dependents_by_age,
        income_global_median=float(dev_df["MonthlyIncome"].median()),
        dependents_global_median=float(dev_df["NumberOfDependents"].median()),
    )


def build_features(df: pd.DataFrame, imputation_reference: ImputationReference) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    data = df.copy()

    data["has_missing_income"] = data["MonthlyIncome"].isna().astype(int)
    data["has_missing_dependents"] = data["NumberOfDependents"].isna().astype(int)

    age_cuts = pd.cut(data["age"], bins=imputation_reference.age_bins)
    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(age_cuts.map(imputation_reference.income_by_age))
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(
        age_cuts.map(imputation_reference.dependents_by_age)
    )

    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(imputation_reference.income_global_median)
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(imputation_reference.dependents_global_median)

    data["dependents_plus_one"] = data["NumberOfDependents"] + 1.0
    data["income_per_dependent"] = data["MonthlyIncome"] / data["dependents_plus_one"]
    data["debt_per_income"] = data["DebtRatio"] * data["MonthlyIncome"]
    data["utilization_per_credit_line"] = data["RevolvingUtilizationOfUnsecuredLines"] / (
        data["NumberOfOpenCreditLinesAndLoans"] + 1.0
    )
    data["real_estate_share"] = data["NumberRealEstateLoansOrLines"] / (
        data["NumberOfOpenCreditLinesAndLoans"] + 1.0
    )
    data["total_past_due_events"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )
    data["weighted_past_due_score"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + 2 * data["NumberOfTime60-89DaysPastDueNotWorse"]
        + 3 * data["NumberOfTimes90DaysLate"]
    )
    data["any_past_due"] = (data["total_past_due_events"] > 0).astype(int)
    data["severe_past_due"] = (data["NumberOfTimes90DaysLate"] > 0).astype(int)
    data["past_due_per_open_line"] = data["total_past_due_events"] / (data["NumberOfOpenCreditLinesAndLoans"] + 1.0)
    data["utilization_times_debt"] = data["RevolvingUtilizationOfUnsecuredLines"] * data["DebtRatio"]
    data["is_senior"] = (data["age"] >= 60).astype(int)
    data["is_young"] = (data["age"] <= 30).astype(int)

    feature_descriptions = [
        {"feature": "has_missing_income", "description": "Индикатор отсутствия MonthlyIncome до импутации."},
        {"feature": "has_missing_dependents", "description": "Индикатор отсутствия NumberOfDependents до импутации."},
        {"feature": "income_per_dependent", "description": "Доход на одного иждивенца с защитой от деления на ноль."},
        {"feature": "debt_per_income", "description": "Прокси денежной долговой нагрузки: DebtRatio * MonthlyIncome."},
        {"feature": "utilization_per_credit_line", "description": "Утилизация незалоговых линий на одну открытую кредитную линию."},
        {"feature": "real_estate_share", "description": "Доля ипотечных/real estate линий среди всех открытых кредитных линий."},
        {"feature": "total_past_due_events", "description": "Сумма всех событий просрочки по трем окнам delinquency."},
        {"feature": "weighted_past_due_score", "description": "Взвешенный скор просрочек: 30-59, 60-89 и 90+ дней с ростом штрафа."},
        {"feature": "any_past_due", "description": "Бинарный признак наличия хотя бы одной просрочки."},
        {"feature": "severe_past_due", "description": "Бинарный признак наличия хотя бы одной просрочки 90+ дней."},
        {"feature": "past_due_per_open_line", "description": "Интенсивность просрочек относительно числа открытых линий."},
        {"feature": "utilization_times_debt", "description": "Нелинейное взаимодействие utilization и debt ratio."},
        {"feature": "is_senior", "description": "Индикатор клиента 60+."},
        {"feature": "is_young", "description": "Индикатор молодого клиента 30-."},
    ]

    return data, feature_descriptions


def clip_outliers(train_df: pd.DataFrame, apply_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clipped = apply_df.copy()
    quantiles = train_df[feature_cols].quantile([0.01, 0.99]).to_dict()
    for feature in feature_cols:
        lower = quantiles[feature][0.01]
        upper = quantiles[feature][0.99]
        clipped[feature] = clipped[feature].clip(lower=lower, upper=upper)
    return clipped


def prepare_model_matrix(
    dev_df: pd.DataFrame,
    oot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], SimpleImputer]:
    feature_cols = [col for col in dev_df.columns if col not in {TARGET, ID_COL}]

    clipped_dev = clip_outliers(dev_df, dev_df, feature_cols)
    clipped_oot = clip_outliers(dev_df, oot_df, feature_cols)

    imputer = SimpleImputer(strategy="median")
    dev_matrix = pd.DataFrame(imputer.fit_transform(clipped_dev[feature_cols]), columns=feature_cols, index=clipped_dev.index)
    oot_matrix = pd.DataFrame(imputer.transform(clipped_oot[feature_cols]), columns=feature_cols, index=clipped_oot.index)

    return dev_matrix, oot_matrix, feature_cols, imputer

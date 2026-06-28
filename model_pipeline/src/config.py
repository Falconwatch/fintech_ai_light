from __future__ import annotations

from dataclasses import dataclass, field


TARGET = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"
DEFAULT_OPTUNA_TRIALS = 5
DEFAULT_CV_FOLDS = 5
DEFAULT_DEV_SHARE = 0.8

BASE_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

EDA_NUMERIC_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
]


@dataclass(frozen=True)
class PipelineConfig:
    target: str = TARGET
    id_col: str = ID_COL
    cv_folds: int = DEFAULT_CV_FOLDS
    dev_share: float = DEFAULT_DEV_SHARE
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS
    base_features: list[str] = field(default_factory=lambda: list(BASE_FEATURES))
    eda_numeric_cols: list[str] = field(default_factory=lambda: list(EDA_NUMERIC_COLS))
    age_bins: list[int] = field(default_factory=lambda: [0, 25, 35, 45, 55, 65, 120])


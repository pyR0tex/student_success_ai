from __future__ import annotations

import pandas as pd

# Simple, interpretable thresholds chosen from the training-data distributions.
BASELINE_THRESHOLDS = {
    "GPA_Change": -0.20,
    "Course_Pass_Rate_Last_2_Terms": 0.80,
    "Assignment_Submission_Rate_4wk": 0.80,
    "Late_Assignment_Rate_4wk": 0.25,
    "Days_Since_Last_LMS_Activity": 4,
}

MIN_RISK_INDICATORS = 2


def add_baseline_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with interpretable baseline risk indicators added."""
    required = set(BASELINE_THRESHOLDS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing baseline feature columns: {missing}")

    result = df.copy()
    result["Risk_GPA_Drop"] = result["GPA_Change"] <= BASELINE_THRESHOLDS["GPA_Change"]
    result["Risk_Low_Pass_Rate"] = (
        result["Course_Pass_Rate_Last_2_Terms"]
        < BASELINE_THRESHOLDS["Course_Pass_Rate_Last_2_Terms"]
    )
    result["Risk_Low_Submission"] = (
        result["Assignment_Submission_Rate_4wk"]
        < BASELINE_THRESHOLDS["Assignment_Submission_Rate_4wk"]
    )
    result["Risk_High_Late_Rate"] = (
        result["Late_Assignment_Rate_4wk"]
        > BASELINE_THRESHOLDS["Late_Assignment_Rate_4wk"]
    )
    result["Risk_LMS_Inactivity"] = (
        result["Days_Since_Last_LMS_Activity"]
        >= BASELINE_THRESHOLDS["Days_Since_Last_LMS_Activity"]
    )

    indicator_cols = [
        "Risk_GPA_Drop",
        "Risk_Low_Pass_Rate",
        "Risk_Low_Submission",
        "Risk_High_Late_Rate",
        "Risk_LMS_Inactivity",
    ]
    result["Baseline_Risk_Count"] = result[indicator_cols].sum(axis=1)
    result["Baseline_Prediction"] = (
        result["Baseline_Risk_Count"] >= MIN_RISK_INDICATORS
    ).astype(int)

    return result


def predict_baseline(df: pd.DataFrame) -> pd.Series:
    """Return 0/1 advisor-review predictions from the rule-based baseline."""
    return add_baseline_indicators(df)["Baseline_Prediction"]

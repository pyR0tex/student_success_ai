from __future__ import annotations

import json

import pandas as pd

from .config import ID_COLUMN, METRICS_DIR, TARGET
from .data_loader import load_csv

PREDICTION_PATH = METRICS_DIR / "heldout_test_predictions.csv"

FEATURES_TO_REPORT = [
    "Prior_Term_GPA",
    "GPA_Change",
    "Course_Pass_Rate_Last_2_Terms",
    "Failed_Courses_Last_2_Terms",
    "Withdrawals_Last_2_Terms",
    "LMS_Login_Days_4wk",
    "Assignment_Submission_Rate_4wk",
    "Late_Assignment_Rate_4wk",
    "Average_Assignment_Score_4wk",
    "Days_Since_Last_LMS_Activity",
]


def _case_payload(row: pd.Series, error_type: str) -> dict:
    if error_type == "False Negative":
        interpretation = (
            "The model assigned a low risk score even though the held-out "
            "outcome indicated intervention was needed. This shows that "
            "apparently favorable observed signals do not guarantee that a "
            "student will not need support."
        )
        consequence = (
            "A student who may need support could be missed or reviewed "
            "later than appropriate."
        )
        improvement = (
            "Keep advisor judgment in the loop and consider richer "
            "longitudinal/contextual evidence in future versions; the "
            "current signals should not be treated as a complete explanation "
            "of student need."
        )
    else:
        interpretation = (
            "The model assigned a high risk score even though the held-out "
            "outcome indicated no intervention was needed. This shows that "
            "concerning academic or engagement signals are not themselves "
            "causes or guarantees of difficulty."
        )
        consequence = (
            "An unnecessary flag could consume advisor time or create "
            "avoidable concern if communicated as a diagnosis."
        )
        improvement = (
            "Use the score only to prioritize human review, provide the "
            "supporting evidence and uncertainty, and require an advisor to "
            "consider additional context before any outreach or academic action."
        )

    observed = {}
    for feature in FEATURES_TO_REPORT:
        value = row[feature]
        observed[feature] = value.item() if hasattr(value, "item") else value

    return {
        "Error_Type": error_type,
        "Student_ID": str(row[ID_COLUMN]),
        "Observation_Term": str(row["Observation_Term"]),
        "Actual_Outcome": int(row[TARGET]),
        "ML_Prediction": int(row["ML_Prediction"]),
        "ML_Risk_Score": round(float(row["ML_Risk_Score"]), 6),
        "Observed_Features": observed,
        "Interpretation": interpretation,
        "Potential_Consequence": consequence,
        "Potential_Improvement": improvement,
    }


def analyze_heldout_failures() -> dict:
    """Select and document one strong false negative and false positive."""
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Held-out predictions not found at {PREDICTION_PATH}. "
            "Run 'python main.py --evaluate-heldout' first."
        )

    predictions = pd.read_csv(PREDICTION_PATH)
    test = load_csv("modeling_test.csv")

    merged = test.merge(
        predictions[
            [
                ID_COLUMN,
                "Observation_Term",
                "ML_Risk_Score",
                "ML_Prediction",
            ]
        ],
        on=[ID_COLUMN, "Observation_Term"],
        how="inner",
        validate="one_to_one",
    )

    merged[TARGET] = merged[TARGET].astype(int)
    merged["ML_Prediction"] = merged["ML_Prediction"].astype(int)

    false_negatives = merged[
        (merged[TARGET] == 1) & (merged["ML_Prediction"] == 0)
    ]
    false_positives = merged[
        (merged[TARGET] == 0) & (merged["ML_Prediction"] == 1)
    ]

    if false_negatives.empty or false_positives.empty:
        raise ValueError(
            "Expected at least one false negative and false positive "
            "for failure analysis."
        )

    # Use the most confident error in each direction so the failure analysis
    # highlights genuine model limitations rather than only borderline cases.
    selected_fn = false_negatives.sort_values(
        "ML_Risk_Score",
        ascending=True,
    ).iloc[0]
    selected_fp = false_positives.sort_values(
        "ML_Risk_Score",
        ascending=False,
    ).iloc[0]

    cases = [
        _case_payload(selected_fn, "False Negative"),
        _case_payload(selected_fp, "False Positive"),
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / "failure_analysis.json"
    csv_path = METRICS_DIR / "failure_analysis_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    pd.DataFrame(
        [
            {
                "Error_Type": item["Error_Type"],
                "Student_ID": item["Student_ID"],
                "Observation_Term": item["Observation_Term"],
                "Actual_Outcome": item["Actual_Outcome"],
                "ML_Prediction": item["ML_Prediction"],
                "ML_Risk_Score": item["ML_Risk_Score"],
                "Potential_Consequence": item["Potential_Consequence"],
                "Potential_Improvement": item["Potential_Improvement"],
            }
            for item in cases
        ]
    ).to_csv(csv_path, index=False)

    result = {
        "heldout_false_negatives": int(len(false_negatives)),
        "heldout_false_positives": int(len(false_positives)),
        "selected_failure_cases": [
            {
                "type": item["Error_Type"],
                "student_id": item["Student_ID"],
                "risk_score": item["ML_Risk_Score"],
            }
            for item in cases
        ],
        "selection_rule": (
            "Most confident false negative and most confident false positive "
            "from the locked held-out evaluation."
        ),
        "outputs": {
            "failure_analysis_json": str(json_path),
            "failure_analysis_csv": str(csv_path),
        },
    }

    with open(
        METRICS_DIR / "failure_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result

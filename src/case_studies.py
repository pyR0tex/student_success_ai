from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .academic_planner import AcademicPlanner
from .baseline import add_baseline_indicators
from .config import ID_COLUMN, LOGS_DIR, METRICS_DIR, MODEL_DIR, OUTPUT_DIR
from .data_loader import load_csv, load_modeling_data
from .decision_logger import write_jsonl
from .early_signal_model import MODEL_FEATURES, NUMERIC_FEATURES

CASE_DIR = OUTPUT_DIR / "cases"
MODEL_PATH = MODEL_DIR / "early_signal_model.joblib"
MODEL_SUMMARY_PATH = METRICS_DIR / "model_development_summary.json"
DECISION_THRESHOLD = 0.5

FEATURE_LABELS = {
    "Academic_Level": "academic level",
    "Credits_Completed": "credits completed",
    "Prior_Term_GPA": "prior-term GPA",
    "GPA_Change": "GPA change",
    "Course_Pass_Rate_Last_2_Terms": "recent course pass rate",
    "Failed_Courses_Last_2_Terms": "failed courses in the last two terms",
    "Withdrawals_Last_2_Terms": "withdrawals in the last two terms",
    "Current_Course_Load_Credits": "current course-load credits",
    "LMS_Login_Days_4wk": "LMS login days over four weeks",
    "Assignment_Submission_Rate_4wk": "four-week assignment submission rate",
    "Late_Assignment_Rate_4wk": "four-week late-assignment rate",
    "Average_Assignment_Score_4wk": "four-week average assignment score",
    "Days_Since_Last_LMS_Activity": "days since last LMS activity",
}

PERCENT_FEATURES = {
    "Course_Pass_Rate_Last_2_Terms",
    "Assignment_Submission_Rate_4wk",
    "Late_Assignment_Rate_4wk",
}


def _selected_model_name() -> str:
    if MODEL_SUMMARY_PATH.exists():
        with open(MODEL_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return str(json.load(f).get("selected_model", "Selected ML Model"))
    return "Selected ML Model"


def _format_value(feature: str, value) -> str:
    if feature in PERCENT_FEATURES:
        return f"{float(value):.1%}"
    if feature in {
        "Prior_Term_GPA",
        "GPA_Change",
        "LMS_Login_Days_4wk",
        "Average_Assignment_Score_4wk",
    }:
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, float) and float(value).is_integer()
    ):
        return str(int(value))
    return str(value)


def _model_evidence(
    model,
    row: pd.Series,
    prediction: int,
    limit: int = 3,
) -> list[dict]:
    """
    Return local Logistic Regression contributions as supporting evidence.

    These contributions describe what pushed the model score up or down.
    They are not causal explanations of student behavior.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        coefficients = np.asarray(estimator.coef_[0])
        transformed = np.asarray(
            preprocessor.transform(row[MODEL_FEATURES].to_frame().T)
        )[0]
        names = list(preprocessor.get_feature_names_out())
    except (AttributeError, KeyError, IndexError):
        return []

    contributions = transformed * coefficients
    evidence: list[dict] = []

    for transformed_name, contribution in zip(names, contributions):
        # Keep the case-study evidence focused on observed numeric signals.
        if not transformed_name.startswith("numeric__"):
            continue

        feature = transformed_name.removeprefix("numeric__")
        if feature not in NUMERIC_FEATURES:
            continue

        # For a positive prediction show factors pushing toward higher risk;
        # for a negative prediction show factors pushing toward lower risk.
        if prediction == 1 and contribution <= 0:
            continue
        if prediction == 0 and contribution >= 0:
            continue

        evidence.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "observed_value": _format_value(feature, row[feature]),
                "contribution": round(float(contribution), 4),
                "direction": (
                    "higher estimated risk"
                    if contribution > 0
                    else "lower estimated risk"
                ),
            }
        )

    evidence.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return evidence[:limit]


def _fallback_evidence(row: pd.Series) -> list[dict]:
    """Fallback for a future selected model that does not expose coefficients."""
    scored = add_baseline_indicators(row.to_frame().T).iloc[0]
    evidence: list[dict] = []
    mappings = [
        ("Risk_GPA_Drop", "GPA_Change"),
        ("Risk_Low_Pass_Rate", "Course_Pass_Rate_Last_2_Terms"),
        ("Risk_Low_Submission", "Assignment_Submission_Rate_4wk"),
        ("Risk_High_Late_Rate", "Late_Assignment_Rate_4wk"),
        ("Risk_LMS_Inactivity", "Days_Since_Last_LMS_Activity"),
    ]

    for indicator, feature in mappings:
        if bool(scored[indicator]):
            evidence.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "observed_value": _format_value(feature, row[feature]),
                    "direction": "baseline warning indicator",
                }
            )

    return evidence[:3]


def _constraint_notes(plan: dict) -> list[str]:
    notes = [
        (
            f"Maximum recommended load: "
            f"{plan['Maximum_Recommended_Credits']} credits per term."
        ),
        (
            f"Remaining before planning: {plan['Remaining_Core_Before_Plan']} "
            f"core course(s) and {plan['Remaining_Electives_Before_Plan']} "
            "elective requirement(s)."
        ),
        (
            "Courses are limited to T6_Spring/T7_Fall offerings and must "
            "satisfy prerequisites."
        ),
    ]
    if plan["T6_Spring"] and plan["T7_Fall"]:
        notes.append(
            "Spring selections may satisfy Fall prerequisites under the "
            "stated successful-completion assumption."
        )
    return notes


def _uncertainty_statement(risk_score: float) -> str:
    parts = [
        (
            "The risk score is a statistical early signal, not a diagnosis "
            "or explanation of why a student may need support."
        ),
        (
            "Important context available to a real advisor may be absent "
            "from the synthetic data."
        ),
    ]
    if abs(risk_score - DECISION_THRESHOLD) <= 0.05:
        parts.append(
            "The score is close to the 0.50 decision threshold, so the "
            "classification should be treated as especially uncertain."
        )
    parts.append(
        "Any academic plan requires advisor review, and Fall prerequisite "
        "assumptions must be revisited if planned Spring courses are not "
        "completed successfully."
    )
    return " ".join(parts)


def generate_end_to_end_case_studies() -> dict:
    """Run prediction, evidence, planning, validation, and logging for advisor cases."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected model not found at {MODEL_PATH}. "
            "Run 'python main.py --develop-model' first."
        )

    _, _, current = load_modeling_data()
    cases = load_csv("advisor_cases.csv")
    case_rows = cases.merge(
        current,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if case_rows[MODEL_FEATURES].isna().any().any():
        missing_students = case_rows.loc[
            case_rows[MODEL_FEATURES].isna().any(axis=1),
            ID_COLUMN,
        ].tolist()
        raise ValueError(
            f"Advisor cases missing current-student modeling data: "
            f"{missing_students}"
        )

    model = joblib.load(MODEL_PATH)
    planner = AcademicPlanner()
    model_name = _selected_model_name()

    baseline_scored = add_baseline_indicators(case_rows)
    risk_scores = model.predict_proba(case_rows[MODEL_FEATURES])[:, 1]
    predictions = (risk_scores >= DECISION_THRESHOLD).astype(int)

    generated_at = datetime.now(timezone.utc).isoformat()
    detailed_cases: list[dict] = []
    summary_rows: list[dict] = []
    decision_logs: list[dict] = []

    for position, (_, row) in enumerate(case_rows.iterrows()):
        case_id = str(row["Case_ID"])
        student_id = str(row[ID_COLUMN])
        risk_score = float(risk_scores[position])
        prediction = int(predictions[position])
        baseline_row = baseline_scored.iloc[position]
        plan = planner.generate_plan(student_id)

        evidence = _model_evidence(model, row, prediction)
        if not evidence:
            evidence = _fallback_evidence(row)

        uncertainty = _uncertainty_statement(risk_score)
        constraints = _constraint_notes(plan)

        case = {
            "Case_ID": case_id,
            "Student_ID": student_id,
            "Program_ID": str(row["Program_ID"]),
            "Early_Signal": {
                "Model": model_name,
                "Risk_Score": round(risk_score, 6),
                "Decision_Threshold": DECISION_THRESHOLD,
                "Advisor_Review_Flag": bool(prediction),
                "Baseline_Risk_Count": int(
                    baseline_row["Baseline_Risk_Count"]
                ),
                "Baseline_Flag": bool(
                    baseline_row["Baseline_Prediction"]
                ),
                "Observed_Evidence": evidence,
            },
            "Academic_Plan": {
                "T6_Spring": plan["T6_Spring"],
                "T6_Spring_Credits": plan["T6_Spring_Credits"],
                "T7_Fall": plan["T7_Fall"],
                "T7_Fall_Credits": plan["T7_Fall_Credits"],
                "Planned_Requirement_Courses": (
                    plan["Planned_Requirement_Courses"]
                ),
                "Validation_Passed": bool(plan["Validation_Passed"]),
                "Validation_Issues": plan["Validation_Issues"],
                "Constraints_and_Assumptions": constraints,
            },
            "Uncertainty_and_Oversight": uncertainty,
            "Human_Review_Required": True,
        }
        detailed_cases.append(case)

        evidence_text = " | ".join(
            (
                f"{item['label']}: {item['observed_value']} "
                f"({item['direction']})"
            )
            for item in evidence
        )

        summary_rows.append(
            {
                "Case_ID": case_id,
                "Student_ID": student_id,
                "Program_ID": str(row["Program_ID"]),
                "ML_Risk_Score": risk_score,
                "Advisor_Review_Flag": prediction,
                "Baseline_Risk_Count": int(
                    baseline_row["Baseline_Risk_Count"]
                ),
                "Observed_Evidence": evidence_text,
                "T6_Spring": ";".join(plan["T6_Spring"]),
                "T6_Spring_Credits": plan["T6_Spring_Credits"],
                "T7_Fall": ";".join(plan["T7_Fall"]),
                "T7_Fall_Credits": plan["T7_Fall_Credits"],
                "Validation_Passed": bool(plan["Validation_Passed"]),
                "Human_Review_Required": True,
            }
        )

        decision_logs.append(
            {
                "timestamp_utc": generated_at,
                "case_id": case_id,
                "student_id": student_id,
                "model": model_name,
                "risk_score": round(risk_score, 6),
                "decision_threshold": DECISION_THRESHOLD,
                "advisor_review_flag": bool(prediction),
                "observed_evidence": evidence,
                "recommended_plan": {
                    "T6_Spring": plan["T6_Spring"],
                    "T7_Fall": plan["T7_Fall"],
                },
                "plan_validation_passed": bool(
                    plan["Validation_Passed"]
                ),
                "plan_validation_issues": plan["Validation_Issues"],
                "constraints_and_assumptions": constraints,
                "human_approval_required": True,
                "autonomous_student_contact_allowed": False,
                "autonomous_course_registration_allowed": False,
            }
        )

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = CASE_DIR / "advisor_case_studies.json"
    csv_path = CASE_DIR / "advisor_case_study_summary.csv"
    log_path = LOGS_DIR / "advisor_case_decision_log.jsonl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_cases, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    write_jsonl(decision_logs, log_path)

    result = {
        "cases": len(detailed_cases),
        "advisor_review_flags": int(
            sum(
                item["Early_Signal"]["Advisor_Review_Flag"]
                for item in detailed_cases
            )
        ),
        "not_flagged": int(
            sum(
                not item["Early_Signal"]["Advisor_Review_Flag"]
                for item in detailed_cases
            )
        ),
        "valid_plans": int(
            sum(
                item["Academic_Plan"]["Validation_Passed"]
                for item in detailed_cases
            )
        ),
        "decision_logs_written": len(decision_logs),
        "model": model_name,
        "decision_threshold": DECISION_THRESHOLD,
        "human_review_required": True,
        "outputs": {
            "case_studies_json": str(json_path),
            "case_studies_csv": str(csv_path),
            "decision_log": str(log_path),
        },
    }

    with open(
        METRICS_DIR / "advisor_case_study_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result

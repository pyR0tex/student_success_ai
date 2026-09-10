from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import FIGURES_DIR, LOGS_DIR, METRICS_DIR, OUTPUT_DIR, PLANS_DIR, PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
CASE_DIR = OUTPUT_DIR / "cases"

# Only final, presentation/report-worthy artifacts are copied here.
# Raw professor-provided CSV files are intentionally not exported.
EXPORT_MAP = {
    # Data understanding
    METRICS_DIR / "data_understanding_summary.json":
        RESULTS_DIR / "data" / "data_understanding_summary.json",
    METRICS_DIR / "training_feature_correlations.csv":
        RESULTS_DIR / "data" / "training_feature_correlations.csv",
    METRICS_DIR / "training_feature_means_by_target.csv":
        RESULTS_DIR / "data" / "training_feature_means_by_target.csv",

    # Model development/evaluation
    METRICS_DIR / "baseline_training_metrics.json":
        RESULTS_DIR / "model" / "baseline_training_metrics.json",
    METRICS_DIR / "model_development_comparison.csv":
        RESULTS_DIR / "model" / "model_development_comparison.csv",
    METRICS_DIR / "model_development_summary.json":
        RESULTS_DIR / "model" / "model_development_summary.json",
    METRICS_DIR / "logistic_regression_coefficients.csv":
        RESULTS_DIR / "model" / "logistic_regression_coefficients.csv",
    METRICS_DIR / "heldout_test_comparison.csv":
        RESULTS_DIR / "model" / "heldout_test_comparison.csv",
    METRICS_DIR / "heldout_evaluation_summary.json":
        RESULTS_DIR / "model" / "heldout_evaluation_summary.json",
    METRICS_DIR / "fairness_error_pattern_audit.csv":
        RESULTS_DIR / "model" / "fairness_error_pattern_audit.csv",
    METRICS_DIR / "failure_analysis_summary.csv":
        RESULTS_DIR / "model" / "failure_analysis_summary.csv",
    METRICS_DIR / "failure_analysis.json":
        RESULTS_DIR / "model" / "failure_analysis.json",
    METRICS_DIR / "required_pipeline_summary.json":
        RESULTS_DIR / "model" / "required_pipeline_summary.json",

    # Planner
    METRICS_DIR / "academic_planner_summary.json":
        RESULTS_DIR / "planner" / "academic_planner_summary.json",
    PLANS_DIR / "advisor_case_plan_summary.csv":
        RESULTS_DIR / "planner" / "advisor_case_plan_summary.csv",
    PLANS_DIR / "advisor_case_course_plans.csv":
        RESULTS_DIR / "planner" / "advisor_case_course_plans.csv",
    PLANS_DIR / "advisor_case_plans.json":
        RESULTS_DIR / "planner" / "advisor_case_plans.json",

    # End-to-end cases and logs
    CASE_DIR / "advisor_case_study_summary.csv":
        RESULTS_DIR / "cases" / "advisor_case_study_summary.csv",
    CASE_DIR / "advisor_case_studies.json":
        RESULTS_DIR / "cases" / "advisor_case_studies.json",
    METRICS_DIR / "advisor_case_study_summary.json":
        RESULTS_DIR / "cases" / "advisor_case_run_summary.json",
    LOGS_DIR / "advisor_case_decision_log.jsonl":
        RESULTS_DIR / "logs" / "advisor_case_decision_log.jsonl",

    # Existing figures
    FIGURES_DIR / "heldout_test_metric_comparison.png":
        RESULTS_DIR / "figures" / "heldout_test_metric_comparison.png",
    FIGURES_DIR / "baseline_heldout_confusion_matrix.png":
        RESULTS_DIR / "figures" / "baseline_heldout_confusion_matrix.png",
    FIGURES_DIR / "ml_heldout_confusion_matrix.png":
        RESULTS_DIR / "figures" / "ml_heldout_confusion_matrix.png",
    FIGURES_DIR / "fairness_error_rates.png":
        RESULTS_DIR / "figures" / "fairness_error_rates.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_presentation_data() -> dict:
    data = _load_json(METRICS_DIR / "data_understanding_summary.json")
    model_development = _load_json(
        METRICS_DIR / "model_development_summary.json"
    )
    heldout = _load_json(METRICS_DIR / "heldout_evaluation_summary.json")
    planner = _load_json(METRICS_DIR / "academic_planner_summary.json")
    cases = _load_json(METRICS_DIR / "advisor_case_study_summary.json")
    failures = _load_json(METRICS_DIR / "failure_analysis_summary.json")

    baseline = heldout["baseline"]
    ml = heldout["machine_learning"]
    fairness = heldout["fairness_audit"]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "training_rows": data["training_rows"],
            "training_students": data["training_students"],
            "test_rows": data["test_rows"],
            "test_students": data["test_students"],
            "current_rows": data["current_rows"],
            "positive_training_rate": data["positive_training_rate"],
            "train_test_student_overlap": data["train_test_student_overlap"],
        },
        "selected_model": model_development["selected_model"],
        "decision_threshold": heldout["decision_threshold"],
        "heldout_comparison": {
            "baseline": baseline,
            "machine_learning": ml,
            "recall_change_percentage_points": (
                ml["recall"] - baseline["recall"]
            ) * 100,
            "f1_change_percentage_points": (
                ml["f1"] - baseline["f1"]
            ) * 100,
            "false_negatives_reduced_by": (
                baseline["false_negative"] - ml["false_negative"]
            ),
            "false_positives_added": (
                ml["false_positive"] - baseline["false_positive"]
            ),
        },
        "fairness_error_pattern_audit": fairness,
        "planner": {
            "cases": planner["cases"],
            "valid_plans": planner["valid_plans"],
            "invalid_plans": planner["invalid_plans"],
            "cases_with_no_recommended_courses":
                planner["cases_with_no_recommended_courses"],
            "planning_terms": planner["planning_terms"],
            "planning_assumption": planner["planning_assumption"],
        },
        "advisor_cases": {
            "cases": cases["cases"],
            "advisor_review_flags": cases["advisor_review_flags"],
            "not_flagged": cases["not_flagged"],
            "valid_plans": cases["valid_plans"],
            "decision_logs_written": cases["decision_logs_written"],
            "human_review_required": cases["human_review_required"],
        },
        "failure_analysis": {
            "heldout_false_negatives": failures["heldout_false_negatives"],
            "heldout_false_positives": failures["heldout_false_positives"],
            "selected_failure_cases": failures["selected_failure_cases"],
            "selection_rule": failures["selection_rule"],
        },
        "responsible_ai_controls": {
            "audit_group_predictive_feature": False,
            "student_id_predictive_feature": False,
            "autonomous_academic_decisions": False,
            "human_review_required": True,
            "prediction_interpretation":
                "Early signal for advisor review, not a diagnosis or causal explanation.",
        },
    }


def export_final_results() -> dict:
    """
    Copy final results into a tracked directory for reporting/presentation use.

    Fails clearly rather than creating a partial export if a required source
    artifact is missing.
    """
    missing = [str(src.relative_to(PROJECT_ROOT)) for src in EXPORT_MAP if not src.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Cannot export final results because required artifacts are missing:\n"
            f"  - {formatted}\n"
            "Run 'python main.py --run-required-pipeline' first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove only generated result subdirectories so stale exports cannot remain.
    for name in ("data", "model", "planner", "cases", "logs", "figures"):
        path = RESULTS_DIR / name
        if path.exists():
            shutil.rmtree(path)

    copied: list[Path] = []
    for source, destination in EXPORT_MAP.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    presentation_path = RESULTS_DIR / "presentation_data.json"
    with open(presentation_path, "w", encoding="utf-8") as f:
        json.dump(_build_presentation_data(), f, indent=2)
    copied.append(presentation_path)

    manifest_entries = []
    for path in sorted(copied):
        manifest_entries.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Derived project results only. Raw professor-provided synthetic "
            "dataset files are not included."
        ),
        "files": manifest_entries,
    }
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "results_directory": str(RESULTS_DIR),
        "exported_files": len(copied),
        "manifest": str(manifest_path),
        "presentation_data": str(presentation_path),
        "raw_data_exported": False,
    }

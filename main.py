import argparse
import json
from pprint import pprint

from src.academic_planner import generate_advisor_case_plans
from src.baseline import BASELINE_THRESHOLDS, MIN_RISK_INDICATORS, add_baseline_indicators
from src.case_studies import generate_end_to_end_case_studies
from src.config import METRICS_DIR, TARGET
from src.data_inspection import inspect_package
from src.data_loader import load_modeling_data
from src.early_signal_model import (
    compare_models_with_group_cv,
    fit_selected_model,
    logistic_regression_coefficients,
    save_model,
)
from src.evaluation import classification_metrics
from src.failure_analysis import analyze_heldout_failures
from src.heldout_evaluation import evaluate_heldout_test
from src.results_export import export_final_results


def run_training_baseline() -> dict:
    """Evaluate the rule-based baseline on training data only."""
    train, _, _ = load_modeling_data()
    scored = add_baseline_indicators(train)

    metrics = classification_metrics(
        train[TARGET].astype(int),
        scored["Baseline_Prediction"],
    )

    result = {
        "dataset": "modeling_train.csv",
        "note": "Training-only development result; held-out test set not used.",
        "thresholds": BASELINE_THRESHOLDS,
        "minimum_risk_indicators": MIN_RISK_INDICATORS,
        "flagged_rows": int(scored["Baseline_Prediction"].sum()),
        "flag_rate": float(scored["Baseline_Prediction"].mean()),
        **metrics,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "baseline_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    output_columns = [
        "Student_ID",
        "Observation_Term",
        TARGET,
        "Risk_GPA_Drop",
        "Risk_Low_Pass_Rate",
        "Risk_Low_Submission",
        "Risk_High_Late_Rate",
        "Risk_LMS_Inactivity",
        "Baseline_Risk_Count",
        "Baseline_Prediction",
    ]
    scored[output_columns].to_csv(
        METRICS_DIR / "baseline_training_predictions.csv",
        index=False,
    )

    return result


def run_model_development() -> dict:
    """Compare simple ML models without using the held-out test set."""
    train, _, _ = load_modeling_data()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    summary, predictions, selected_model = compare_models_with_group_cv(train)
    summary.to_csv(METRICS_DIR / "model_development_comparison.csv", index=False)
    predictions.to_csv(METRICS_DIR / "model_development_predictions.csv", index=False)

    fitted_model = fit_selected_model(train, selected_model)
    model_path = save_model(fitted_model)

    coefficient_path = None
    if selected_model == "Logistic Regression":
        coefficient_path = METRICS_DIR / "logistic_regression_coefficients.csv"
        logistic_regression_coefficients(fitted_model).to_csv(coefficient_path, index=False)

    result = {
        "dataset": "modeling_train.csv",
        "validation_method": "5-fold stratified group cross-validation by Student_ID",
        "held_out_test_used": False,
        "selected_model": selected_model,
        "selection_rule": "Highest cross-validated F1; recall used as tie-breaker.",
        "model_file": str(model_path),
        "coefficient_file": str(coefficient_path) if coefficient_path else None,
        "models": summary.to_dict(orient="records"),
    }

    with open(METRICS_DIR / "model_development_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def run_required_pipeline() -> dict:
    """
    Reproduce the complete required backend in the intended order.

    The held-out set remains evaluation-only: baseline/model development occur
    before held-out evaluation, and no later step feeds test results back into
    model or threshold selection.
    """
    print("[1/7] Data Understanding")
    data_understanding = inspect_package()

    print("[2/7] Rule-Based Baseline")
    baseline = run_training_baseline()

    print("[3/7] Machine-Learning Early-Signal Model")
    model_development = run_model_development()

    print("[4/7] Held-Out Evaluation + Fairness/Error-Pattern Audit")
    _, test, _ = load_modeling_data()
    heldout = evaluate_heldout_test(test)

    print("[5/7] Academic Plan Generation + Validation")
    planner = generate_advisor_case_plans()

    print("[6/7] End-to-End Advisor Case Studies + Decision Logging")
    case_studies = generate_end_to_end_case_studies()

    print("[7/7] Failure Analysis")
    failure_analysis = analyze_heldout_failures()

    result = {
        "completed": True,
        "execution_order": [
            "Data Understanding",
            "Rule-Based Baseline",
            "Machine-Learning Early-Signal Model",
            "Held-Out Evaluation + Fairness/Error-Pattern Audit",
            "Academic Plan Generation + Validation",
            "End-to-End Advisor Case Studies + Decision Logging",
            "Failure Analysis",
        ],
        "key_checks": {
            "train_test_student_overlap": data_understanding[
                "train_test_student_overlap"
            ],
            "heldout_selected_model": heldout["selected_model"],
            "heldout_decision_threshold": heldout["decision_threshold"],
            "valid_advisor_case_plans": planner["valid_plans"],
            "invalid_advisor_case_plans": planner["invalid_plans"],
            "advisor_cases_logged": case_studies["decision_logs_written"],
            "heldout_false_negatives": failure_analysis[
                "heldout_false_negatives"
            ],
            "heldout_false_positives": failure_analysis[
                "heldout_false_positives"
            ],
        },
        "notes": [
            "modeling_test.csv is used only for final held-out evaluation and failure analysis.",
            "Audit_Group is used only after prediction for the fairness/error-pattern audit.",
            "Academic plans require independent validation and human advisor review.",
        ],
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(
        METRICS_DIR / "required_pipeline_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student Success Early-Signal & Planning System"
    )
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="Run data-package inspection and data-understanding checks.",
    )
    parser.add_argument(
        "--baseline-train",
        action="store_true",
        help="Run the rule-based baseline on training data only.",
    )
    parser.add_argument(
        "--develop-model",
        action="store_true",
        help="Compare candidate early-signal ML models using training data only.",
    )
    parser.add_argument(
        "--evaluate-heldout",
        action="store_true",
        help=(
            "Evaluate the locked baseline and selected ML model on the held-out "
            "test set and run the fairness audit."
        ),
    )
    parser.add_argument(
        "--generate-plans",
        action="store_true",
        help=(
            "Generate and independently validate two-term academic plans for "
            "the required advisor cases."
        ),
    )
    parser.add_argument(
        "--run-case-studies",
        action="store_true",
        help=(
            "Run the full early-signal + planning flow for the required advisor "
            "cases and write decision logs."
        ),
    )
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help=(
            "Analyze representative false-negative and false-positive failures "
            "from the held-out evaluation."
        ),
    )
    parser.add_argument(
        "--run-required-pipeline",
        action="store_true",
        help=(
            "Reproduce the complete required backend in the intended order. "
            "This can take several minutes because model cross-validation is rerun."
        ),
    )
    parser.add_argument(
        "--export-results",
        action="store_true",
        help=(
            "Copy final report/presentation artifacts into the tracked results/ "
            "directory and build a manifest/presentation summary."
        ),
    )
    args = parser.parse_args()

    if args.inspect_data:
        print("Data understanding and package inspection")
        pprint(inspect_package())
    elif args.baseline_train:
        print("Rule-based baseline (training data only)")
        pprint(run_training_baseline())
    elif args.develop_model:
        print("Machine-learning early-signal model development")
        pprint(run_model_development())
    elif args.evaluate_heldout:
        print("Held-out model evaluation and fairness/error-pattern audit")
        _, test, _ = load_modeling_data()
        pprint(evaluate_heldout_test(test))
    elif args.generate_plans:
        print("Constraint-based academic plan generation")
        pprint(generate_advisor_case_plans())
    elif args.run_case_studies:
        print("End-to-end advisor case studies and decision logging")
        pprint(generate_end_to_end_case_studies())
    elif args.analyze_failures:
        print("Held-out failure analysis")
        pprint(analyze_heldout_failures())
    elif args.run_required_pipeline:
        print("Required backend reproducibility run")
        pprint(run_required_pipeline())
    elif args.export_results:
        print("Exporting final report/presentation results")
        pprint(export_final_results())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

import argparse
import json
from pprint import pprint

from src.baseline import BASELINE_THRESHOLDS, MIN_RISK_INDICATORS, add_baseline_indicators
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
from src.heldout_evaluation import evaluate_heldout_test


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Student Success Early-Signal & Planning System")
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
        help="Evaluate the locked baseline and selected ML model on the held-out test set and run the fairness audit.",
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

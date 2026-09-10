import argparse
import json
from pprint import pprint

from src.baseline import BASELINE_THRESHOLDS, MIN_RISK_INDICATORS, add_baseline_indicators
from src.config import METRICS_DIR, TARGET
from src.data_inspection import inspect_package
from src.data_loader import load_modeling_data
from src.evaluation import classification_metrics


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
    args = parser.parse_args()

    if args.inspect_data:
        print("Data understanding and package inspection")
        pprint(inspect_package())
    elif args.baseline_train:
        print("Rule-based baseline (training data only)")
        pprint(run_training_baseline())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

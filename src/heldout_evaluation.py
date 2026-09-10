from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .baseline import add_baseline_indicators
from .config import (
    AUDIT_FILE,
    FIGURES_DIR,
    ID_COLUMN,
    METRICS_DIR,
    MODEL_DIR,
    TARGET,
)
from .early_signal_model import MODEL_FEATURES
from .evaluation import classification_metrics
from .fairness_audit import audit_error_rates

MODEL_PATH = MODEL_DIR / "early_signal_model.joblib"
MODEL_DEVELOPMENT_SUMMARY = METRICS_DIR / "model_development_summary.json"
DECISION_THRESHOLD = 0.5


def _load_selected_model_name() -> str:
    if MODEL_DEVELOPMENT_SUMMARY.exists():
        with open(MODEL_DEVELOPMENT_SUMMARY, "r", encoding="utf-8") as f:
            return str(json.load(f).get("selected_model", "Selected ML Model"))
    return "Selected ML Model"


def _plot_metric_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    chart = comparison.set_index("model")[metrics].T
    ax = chart.plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_title("Held-Out Test Performance")
    ax.legend(title="Approach")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["No intervention", "Intervention"],
        cmap=None,
        colorbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_fairness_rates(audit: pd.DataFrame, output_path: Path) -> None:
    chart = audit.set_index("Audit_Group")[["recall", "false_positive_rate", "false_negative_rate"]]
    ax = chart.plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Synthetic Audit Group")
    ax.set_title("Selected Model Error Rates by Audit Group")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_heldout_test(test: pd.DataFrame) -> dict:
    """Evaluate the locked baseline and selected ML model on modeling_test.csv once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected model not found at {MODEL_PATH}. Run 'python main.py --develop-model' first."
        )

    required = set(MODEL_FEATURES + [ID_COLUMN, "Observation_Term", TARGET])
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"Missing held-out test columns: {missing}")

    model = joblib.load(MODEL_PATH)
    selected_model_name = _load_selected_model_name()

    baseline_scored = add_baseline_indicators(test)
    baseline_predictions = baseline_scored["Baseline_Prediction"].astype(int)

    ml_probabilities = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    ml_predictions = (ml_probabilities >= DECISION_THRESHOLD).astype(int)
    y_true = test[TARGET].astype(int)

    baseline_metrics = classification_metrics(y_true, baseline_predictions)
    ml_metrics = classification_metrics(y_true, ml_predictions)

    comparison = pd.DataFrame(
        [
            {"model": "Rule-Based Baseline", "decision_threshold": None, **baseline_metrics},
            {"model": selected_model_name, "decision_threshold": DECISION_THRESHOLD, **ml_metrics},
        ]
    )

    predictions = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].values,
            "Observation_Term": test["Observation_Term"].values,
            TARGET: y_true.values,
            "Baseline_Risk_Count": baseline_scored["Baseline_Risk_Count"].values,
            "Baseline_Prediction": baseline_predictions.values,
            "ML_Risk_Score": ml_probabilities,
            "ML_Prediction": ml_predictions,
        }
    )

    audit_groups = pd.read_csv(AUDIT_FILE)
    fairness = audit_error_rates(predictions, audit_groups, prediction_column="ML_Prediction")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(METRICS_DIR / "heldout_test_comparison.csv", index=False)
    predictions.to_csv(METRICS_DIR / "heldout_test_predictions.csv", index=False)
    fairness.to_csv(METRICS_DIR / "fairness_error_pattern_audit.csv", index=False)

    _plot_metric_comparison(comparison, FIGURES_DIR / "heldout_test_metric_comparison.png")
    _plot_confusion_matrix(
        y_true,
        baseline_predictions,
        "Rule-Based Baseline Confusion Matrix",
        FIGURES_DIR / "baseline_heldout_confusion_matrix.png",
    )
    _plot_confusion_matrix(
        y_true,
        ml_predictions,
        f"{selected_model_name} Confusion Matrix",
        FIGURES_DIR / "ml_heldout_confusion_matrix.png",
    )
    _plot_fairness_rates(fairness, FIGURES_DIR / "fairness_error_rates.png")

    result = {
        "dataset": "modeling_test.csv",
        "note": "Locked baseline and selected ML model evaluated on the held-out test set.",
        "selected_model": selected_model_name,
        "decision_threshold": DECISION_THRESHOLD,
        "baseline": baseline_metrics,
        "machine_learning": ml_metrics,
        "fairness_audit": fairness.to_dict(orient="records"),
    }

    with open(METRICS_DIR / "heldout_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

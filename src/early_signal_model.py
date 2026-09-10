from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COLUMN, MODEL_DIR, TARGET
from .evaluation import classification_metrics

# Student_ID is an identifier, not a predictive feature. Observation_Term is excluded
# because the current-student data is from a future term (T5_Fall) not seen during model development.
CATEGORICAL_FEATURES = [
    "Program_ID",
    "Entry_Type",
]

NUMERIC_FEATURES = [
    "Academic_Level",
    "Credits_Completed",
    "Prior_Term_GPA",
    "GPA_Change",
    "Course_Pass_Rate_Last_2_Terms",
    "Failed_Courses_Last_2_Terms",
    "Withdrawals_Last_2_Terms",
    "Current_Course_Load_Credits",
    "LMS_Login_Days_4wk",
    "Assignment_Submission_Rate_4wk",
    "Late_Assignment_Rate_4wk",
    "Average_Assignment_Score_4wk",
    "Days_Since_Last_LMS_Activity",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class CandidateModel:
    name: str
    factory: Callable[[], object]


def _check_modeling_columns(df: pd.DataFrame) -> None:
    required = set(MODEL_FEATURES + [ID_COLUMN, TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing modeling columns: {missing}")


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing shared by all candidate models."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def candidate_models() -> list[CandidateModel]:
    """Return a small, intentionally simple set of candidate models."""
    return [
        CandidateModel(
            name="Logistic Regression",
            factory=lambda: LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        CandidateModel(
            name="Random Forest",
            factory=lambda: RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        CandidateModel(
            name="Gradient Boosting",
            factory=lambda: HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def compare_models_with_group_cv(
    train: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Compare candidate models using out-of-fold predictions.

    Splits are grouped by Student_ID so observations from the same student cannot
    appear in both the training and validation fold.
    """
    _check_modeling_columns(train)

    X = train[MODEL_FEATURES]
    y = train[TARGET].astype(int)
    groups = train[ID_COLUMN]

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    summary_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for candidate in candidate_models():
        pipeline = build_pipeline(candidate.factory())
        probabilities = cross_val_predict(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = classification_metrics(y, predictions)
        metrics.update(
            {
                "model": candidate.name,
                "cv_folds": n_splits,
                "decision_threshold": 0.5,
                "flag_rate": float(predictions.mean()),
                "roc_auc": float(roc_auc_score(y, probabilities)),
            }
        )
        summary_rows.append(metrics)

        prediction_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: train[ID_COLUMN].values,
                    "Observation_Term": train["Observation_Term"].values,
                    TARGET: y.values,
                    "Model": candidate.name,
                    "Predicted_Probability": probabilities,
                    "Prediction": predictions,
                }
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary[
        [
            "model",
            "cv_folds",
            "decision_threshold",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "roc_auc",
            "flag_rate",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]
    ].sort_values(["f1", "recall"], ascending=[False, False])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    selected_model = str(summary.iloc[0]["model"])
    return summary, predictions, selected_model


def fit_selected_model(train: pd.DataFrame, model_name: str) -> Pipeline:
    """Fit the selected candidate on all development data after model comparison."""
    _check_modeling_columns(train)
    candidates = {candidate.name: candidate for candidate in candidate_models()}
    if model_name not in candidates:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = build_pipeline(candidates[model_name].factory())
    pipeline.fit(train[MODEL_FEATURES], train[TARGET].astype(int))
    return pipeline


def save_model(model: Pipeline, filename: str = "early_signal_model.joblib") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    return path


def logistic_regression_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return transformed feature coefficients when the selected model is logistic regression."""
    estimator = model.named_steps["model"]
    if not isinstance(estimator, LogisticRegression):
        raise TypeError("Coefficient extraction is only available for Logistic Regression.")

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = estimator.coef_[0]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    )
    return result.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)

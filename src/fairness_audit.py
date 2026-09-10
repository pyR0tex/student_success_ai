from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import AUDIT_COLUMN, ID_COLUMN, TARGET


def audit_error_rates(
    predictions: pd.DataFrame,
    audit_groups: pd.DataFrame,
    prediction_column: str = "ML_Prediction",
) -> pd.DataFrame:
    """Compute required post-hoc error rates by synthetic audit group."""
    required_prediction_columns = {ID_COLUMN, TARGET, prediction_column}
    missing_prediction = sorted(required_prediction_columns - set(predictions.columns))
    if missing_prediction:
        raise ValueError(f"Missing prediction columns: {missing_prediction}")

    required_audit_columns = {ID_COLUMN, AUDIT_COLUMN}
    missing_audit = sorted(required_audit_columns - set(audit_groups.columns))
    if missing_audit:
        raise ValueError(f"Missing audit columns: {missing_audit}")

    merged = predictions.merge(
        audit_groups[[ID_COLUMN, AUDIT_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    if merged[AUDIT_COLUMN].isna().any():
        missing_count = int(merged[AUDIT_COLUMN].isna().sum())
        raise ValueError(f"{missing_count} prediction rows are missing an audit-group label.")

    rows: list[dict] = []
    for group_name, group_df in merged.groupby(AUDIT_COLUMN, sort=True):
        y_true = group_df[TARGET].astype(int)
        y_pred = group_df[prediction_column].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

        rows.append(
            {
                AUDIT_COLUMN: group_name,
                "rows": int(len(group_df)),
                "actual_positive_rate": float(y_true.mean()),
                "recall": float(recall),
                "false_positive_rate": float(false_positive_rate),
                "false_negative_rate": float(false_negative_rate),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    return pd.DataFrame(rows)

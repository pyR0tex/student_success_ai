import json
from pathlib import Path
import pandas as pd

from .config import DATA_DIR, METRICS_DIR, TARGET, ID_COLUMN
from .data_loader import validate_data_package, load_modeling_data


def inspect_package(data_dir: Path = DATA_DIR) -> dict:
    validate_data_package(data_dir)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    file_rows = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        file_rows.append({
            "file": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        })

    package_summary = pd.DataFrame(file_rows)
    package_summary.to_csv(METRICS_DIR / "data_package_summary.csv", index=False)

    train, test, current = load_modeling_data(data_dir)

    train_ids = set(train[ID_COLUMN])
    test_ids = set(test[ID_COLUMN])
    target_counts = train[TARGET].value_counts().sort_index()

    numeric_cols = train.select_dtypes(include="number").columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    correlations = (
        train[numeric_cols + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    correlations.rename("target_correlation").to_csv(
        METRICS_DIR / "training_feature_correlations.csv"
    )

    group_means = train.groupby(TARGET)[numeric_cols].mean().T
    group_means.to_csv(METRICS_DIR / "training_feature_means_by_target.csv")

    result = {
        "training_rows": len(train),
        "training_students": train[ID_COLUMN].nunique(),
        "test_rows": len(test),
        "test_students": test[ID_COLUMN].nunique(),
        "current_rows": len(current),
        "train_test_student_overlap": len(train_ids & test_ids),
        "positive_training_examples": int(target_counts.get(1, 0)),
        "negative_training_examples": int(target_counts.get(0, 0)),
        "positive_training_rate": float(train[TARGET].mean()),
        "missing_cells_in_training": int(train.isna().sum().sum()),
        "duplicate_training_rows": int(train.duplicated().sum()),
        "observation_terms": sorted(train["Observation_Term"].unique().tolist()),
    }

    with open(METRICS_DIR / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

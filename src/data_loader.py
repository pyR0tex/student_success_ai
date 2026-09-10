from pathlib import Path
import pandas as pd

from .config import DATA_DIR, TARGET, AUDIT_COLUMN

EXPECTED_FILES = {
    "modeling_train.csv",
    "modeling_test.csv",
    "current_students_unlabeled.csv",
    "audit_groups.csv",
    "students.csv",
    "engagement_history.csv",
    "student_course_history.csv",
    "course_catalog.csv",
    "prerequisites.csv",
    "degree_requirements.csv",
    "future_course_offerings.csv",
    "advisor_cases.csv",
}


def validate_data_package(data_dir: Path = DATA_DIR) -> None:
    missing = sorted(EXPECTED_FILES - {p.name for p in data_dir.glob("*.csv")})
    if missing:
        raise FileNotFoundError(f"Missing required dataset files: {missing}")


def load_csv(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_modeling_data(data_dir: Path = DATA_DIR):
    """Load train/test/current data without joining audit labels into model inputs."""
    train = load_csv("modeling_train.csv", data_dir)
    test = load_csv("modeling_test.csv", data_dir)
    current = load_csv("current_students_unlabeled.csv", data_dir)

    if AUDIT_COLUMN in train.columns or AUDIT_COLUMN in test.columns or AUDIT_COLUMN in current.columns:
        raise ValueError("Audit_Group must never be present in predictive modeling data.")
    if TARGET not in train.columns or TARGET not in test.columns:
        raise ValueError(f"Expected target column {TARGET!r} in train and test data.")
    if TARGET in current.columns:
        raise ValueError("Current-student data must be unlabeled.")

    return train, test, current

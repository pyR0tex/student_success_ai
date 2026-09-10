from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
PLANS_DIR = OUTPUT_DIR / "plans"
LOGS_DIR = OUTPUT_DIR / "logs"

TARGET = "Intervention_Needed_Next_8_Weeks"
ID_COLUMN = "Student_ID"
AUDIT_COLUMN = "Audit_Group"

TRAIN_FILE = DATA_DIR / "modeling_train.csv"
TEST_FILE = DATA_DIR / "modeling_test.csv"
CURRENT_FILE = DATA_DIR / "current_students_unlabeled.csv"
AUDIT_FILE = DATA_DIR / "audit_groups.csv"

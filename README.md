# Student Success Early-Signal & Academic Planning System

A synthetic-data decision-support prototype that combines:

1. a rule-based early-alert baseline,
2. a supervised machine-learning early-signal model,
3. held-out evaluation and post-hoc fairness/error analysis,
4. a constraint-based two-term academic planner,
5. end-to-end advisor case studies with decision logs.

The dataset is fully synthetic. The system is designed to support advisor judgment and does not make autonomous academic decisions.

## Setup

### Python Version

```text
Python 3.13.5
```

### Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The professor-provided CSV files are expected locally in `data/`. They are intentionally excluded from Git.

## Main Commands

```bash
python main.py --inspect-data
python main.py --baseline-train
python main.py --develop-model
python main.py --evaluate-heldout
python main.py --generate-plans
python main.py --run-case-studies
python main.py --analyze-failures
```

## Reproduce the Required Backend

Run the safeguard tests:

```bash
python -m unittest discover -s tests -v
```

Then reproduce the complete required backend in the intended order:

```bash
python main.py --run-required-pipeline
```

This reruns model cross-validation, so it may take several minutes. The held-out test set remains evaluation-only: it is not used for model fitting, feature selection, rule design, hyperparameter tuning, or decision-threshold tuning.

## Export Final Results

After the required pipeline finishes:

```bash
python main.py --export-results
```

This creates a tracked `results/` directory containing selected derived artifacts for the report and presentation, including model metrics, fairness results, academic-plan outputs, advisor case studies, decision logs, figures, a file manifest, and `presentation_data.json`.

Raw professor-provided dataset files are not copied into `results/`.

## Responsible AI

- `Audit_Group` is used only for post-hoc fairness analysis and never as a predictive feature.
- `Student_ID` is an identifier and is never used as a predictive feature.
- Predictions are treated as early signals, not diagnoses or explanations of student behavior.
- Academic plans are recommendations for advisor review and must pass the independent validator.
- The system does not autonomously contact students, register courses, or make academic decisions.

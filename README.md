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

### Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --inspect-data
```

## Responsible AI

- Audit_Group is used only for post-hoc fairness analysis and never as a predictive feature.
- Predictions are treated as early signals, not diagnoses or explanations of student behavior.
- Academic plans are recommendations for advisor review.
- The system does not autonomously contact students, register courses, or make academic decisions.

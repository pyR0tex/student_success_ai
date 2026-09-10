# Student Success Early-Signal & Academic Planning System

MSCS 2201 final-project prototype combining:

1. a rule-based early-alert baseline,
2. a supervised ML early-signal model,
3. held-out evaluation and post-hoc fairness/error analysis,
4. a constraint-based two-term academic planner,
5. end-to-end advisor case studies with decision logs.

The dataset is fully synthetic. The system supports advisor judgment and does not make autonomous academic decisions.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --inspect-data
```

## Current status

Phase 1 scaffold and data-package inspection are implemented. Modeling, fairness, planning, case studies, and demo layers will be added incrementally.

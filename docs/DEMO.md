# Presentation demo

From the repository root, activate the environment and launch:

```powershell
.venv\Scripts\Activate.ps1
python -m streamlit run demo_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. Stop with Ctrl+C. For a fresh environment, install
`python -m pip install -r requirements-lock.txt` first. Only tracked `results/`
artifacts are needed; missing artifacts show a restore message.

## Live sequence (5–7 minutes)

1. **Project overview — 1 minute:** Logistic Regression improved recall and F1.
   Highlight 20 fewer false negatives, 16 additional false positives, and 12/12
   validated academic plans. Read the chart in order: Recall → F1 → Precision.
   Open “Why was Logistic Regression selected?” only if asked about training-only
   group cross-validation, the three candidate models, or the F1/recall selection rule.
2. **Case_06 — 1 minute:** The default early-signal example shows 77.6% risk,
   a locked 50% threshold, YES flag, and 5 baseline indicators. Point to 7 days
   since LMS activity, 68% submission, and −0.40 GPA change, then the validated
   Spring/Fall plan and conditional prerequisite link.
3. **Case_02 — 1 minute:** Use the prerequisite shortcut. Risk is 49.5%, just
   below threshold: the early-signal flag is NO, but the academic plan still
   requires advisor review. Spring Program_A_C02 unlocks Fall Program_A_C04
   and Program_A_C05 only if successfully completed. Show the near-threshold warning.
4. **Fairness audit — 45 seconds:** Recall is similar; Audit_C has the highest
   false-positive rate (43.52%). Synthetic audit groups were excluded from
   prediction and do not establish real-world demographic fairness.
5. **Failure analysis — 1 minute:** Contrast Student_1383 (18.4%, false negative)
   with Student_0880 (90.3%, false positive). Point to observed signals and the
   risk of missed support versus unnecessary review. Even confident predictions
   can be wrong; signals are not causes.
6. **Decision audit / logging — 45 seconds:** Case_06 opens by default. Show
   validation and the three safety controls. For “what gets logged?”, open
   “Full raw decision log.” Evidence and plan constraints are also expandable.

## Backup / Q&A

- **Reproducibility:** 5,148 training rows, 1,452 held-out rows, zero student
  overlap. The test set was used only for final evaluation. Methodology,
  export timestamp, and manifest are collapsed.
- **Planner details:** Open “Planner constraints and assumptions.” Completed
  courses are excluded; prerequisites must precede the dependent term; availability,
  remaining degree requirements, and credit limits constrain every plan.
- **Limitations:** Open “Full uncertainty and limitations.” Scores are early
  signals, not diagnoses; advisor context may be missing from synthetic data.
- **Log metadata:** The secondary detail area explains that distinct model-version
  and completed-human-approval fields are absent from the saved export.

Before presenting, check readability at projector resolution, the Case_06/Case_02
shortcuts, and the collapsed raw log. Human review is required; autonomous contact
and registration are not allowed. The dashboard does not change backend results.

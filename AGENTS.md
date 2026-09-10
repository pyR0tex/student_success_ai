# Repository Rules

These are project and assignment constraints, not optional style preferences.

1. `Audit_Group` must NEVER be used as a predictive feature. It is only for post-hoc fairness/error analysis.
2. `modeling_test.csv` is reserved for final evaluation and must NEVER be used to fit models, tune thresholds, select hyperparameters, or engineer rules.
3. `Student_ID` is an identifier and must not be used as a predictive feature.
4. Academic plans must pass an independent validator before being shown as feasible.
5. A successfully completed course must not be recommended again.
6. Prerequisites must be completed before the dependent course is taken; do not assume same-term prerequisite completion is sufficient.
7. Plans may include only courses marked available for that term and must respect `Maximum_Recommended_Credits`.
8. Predictions are early signals for advisor review, not diagnoses or autonomous academic decisions.
9. Never infer personal causes from behavioral signals. Report only evidence supported by the synthetic data.
10. Any future outreach-generation extension must label messages `DRAFT FOR HUMAN REVIEW` and must not autonomously contact students or register them.
11. Decision outputs should be logged with model/version information, score/threshold, evidence, planner constraints, validation result, warnings, and human-approval status.
12. Run tests after changing model, planner, or validator logic.

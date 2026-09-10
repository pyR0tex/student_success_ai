# Presentation demo

From the repository root, activate the project's virtual environment and launch:

```powershell
.venv\Scripts\Activate.ps1
python -m streamlit run demo_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501 if the browser does not open automatically. Stop with Ctrl+C.
For a fresh environment, install `python -m pip install -r requirements-lock.txt` first.
The dashboard reads tracked `results/` artifacts only. Raw CSVs, fitted model files,
and a backend rerun are unnecessary. If an artifact is missing, restore the complete
tracked results export and reload; the UI displays the affected filename.

## Suggested sequence (5–7 minutes)

1. **Project overview:** Explain synthetic decision support. Compare held-out precision,
   recall, and F1; ML missed 20 fewer positives but added 16 false positives. Show 12/12
   validated plans. Explain that Logistic Regression won on training-only group-CV F1,
   with recall as the tie-breaker, before held-out evaluation.
2. **Advisor case explorer:** Start with default **Case_02**. Spring `Program_A_C02`
   enables Fall `Program_A_C04` and `Program_A_C05` only if successfully completed.
   Point to credits, saved independent validation, and the near-threshold uncertainty.
   Use the **Case_06** shortcut to show the flagged score and observed LMS inactivity,
   submission rate, and GPA change. All 12 cases are available in the dropdown.
3. **Fairness / error-pattern audit:** Compare recall, false-positive rate, and
   false-negative rate. Highlight Audit_C's higher false-positive rate despite similar
   recall. Audit_Group was excluded from prediction; synthetic subgroup patterns do
   not establish real-world demographic fairness.
4. **Failure analysis:** Contrast Student_1383 (false negative) and Student_0880
   (false positive). Read their observed features and explain the consequences and
   potential improvements. Confident predictions can still be wrong.
5. **Decision audit / logging:** Select Case_06. Show the saved score, threshold,
   evidence, plan, constraints, validator result, and approval/autonomy fields, then
   the full JSON record. Model version and completed approval are not in the saved
   log; the UI does not invent them or write new decisions.
6. **Reproducibility:** Show training/held-out row counts, zero student overlap,
   and the exported manifest. Distinguish derived tracked results from excluded source CSVs.

## Responsible AI and scope

Signals are evidence for advisor review, not diagnoses or personal causal explanations.
Plans require independent validation and human review; successful Spring completion
is conditional. There is no autonomous outreach, registration, or optional agentic extension.
The demo does not change models, thresholds, plans, logs, or reported results.

The optional threshold explorer is omitted because full held-out predictions are not
exported in `results/`. Any future explorer must be labeled **POST-HOC EDUCATIONAL
VISUALIZATION**, must not tune the model, and must preserve the official 0.50 threshold
and all final reported metrics.

Before presenting, check chart/table readability at your projector resolution and
rehearse the Case_02 → Case_06 shortcuts and saved-log dropdown.

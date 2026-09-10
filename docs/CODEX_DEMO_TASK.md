You are working on the demo-polish branch of my Student Success Early-Signal
& Academic Planning System.

Before making any changes, inspect the repository carefully.

Read:
- AGENTS.md
- README.md
- requirements.txt
- requirements-lock.txt
- main.py
- the full src/ directory
- tests/
- results/presentation_data.json
- results/model/
- results/planner/
- results/cases/
- results/logs/

Also inspect any existing documentation that is relevant.

IMPORTANT:
The required backend is complete, tested, and should remain unchanged.
This task is ONLY to add a lightweight presentation/demo layer.

Do NOT redesign or change:
- predictive features
- Student_ID exclusion
- Audit_Group exclusion
- selected model
- Logistic Regression implementation
- official 0.50 decision threshold
- baseline thresholds
- model-development methodology
- train/test separation
- fairness/error-pattern methodology
- academic planner constraints
- plan-validation rules
- generated advisor-case results
- held-out results
- failure-analysis selection logic

Do not add the optional agentic extension.

The goal is a lightweight Streamlit dashboard that helps me demonstrate
the existing project clearly during a university AI course presentation.
It should not look or behave like a production system.

Create a presentation-ready Streamlit dashboard, preferably as:

demo_app.py

Use the already-exported tracked results/ artifacts whenever possible.
Do not duplicate machine-learning or planning logic inside the UI.

The dashboard should contain the following sections.

1. PROJECT OVERVIEW

Show:
- project title
- short explanation that this is a synthetic-data decision-support system
- selected model: Logistic Regression
- held-out baseline vs ML:
  - precision
  - recall
  - F1
- prominently show:
  - ML reduced false negatives by 20
  - ML added 16 false positives
- show that all 12/12 required advisor-case academic plans passed validation
- briefly show why Logistic Regression was selected over Random Forest and
  Gradient Boosting using existing development results

Keep the visualizations simple and readable.

2. ADVISOR CASE EXPLORER

Provide a dropdown for all 12 advisor cases.

For the selected case show:
- Case_ID
- Student_ID
- Program_ID
- ML risk score
- locked decision threshold
- advisor-review flag
- baseline risk-indicator count
- observed evidence supporting the model output
- T6_Spring recommended courses
- T6_Spring credits
- T7_Fall recommended courses
- T7_Fall credits
- independent plan-validation result
- planning constraints / assumptions
- uncertainty / human-review statement

Make Case_02 easy to select and demonstrate because it clearly shows
cross-term prerequisite reasoning:
a Spring course enables Fall courses.

Also make Case_06 easy to demonstrate because it is a clearly flagged case
with several observable risk signals.

Where possible, visually show prerequisite relationships such as:

Spring course -> satisfies prerequisite -> Fall course

Do not imply that successful Spring completion is guaranteed.

3. FAIRNESS / ERROR-PATTERN AUDIT

Visualize:
- recall
- false-positive rate
- false-negative rate

for:
- Audit_A
- Audit_B
- Audit_C

Clearly call out the main finding:
recall is relatively similar across the synthetic groups, but Audit_C has
a noticeably higher false-positive rate.

Explicitly state:
- Audit_Group was never a predictive feature
- these groups are synthetic
- this analysis identifies subgroup error patterns
- it does not establish real-world demographic fairness

4. FAILURE ANALYSIS

Show the project's two selected held-out failure examples:

False Negative:
- Student_1383
- risk score approximately 0.1837

False Positive:
- Student_0880
- risk score approximately 0.902924

Read the exact values and observed evidence from results/ rather than
hardcoding unnecessary data.

For each example show:
- predicted result
- actual result
- risk score
- important observed features
- potential consequence
- lesson / potential improvement

Emphasize:
academic and engagement indicators are signals, not causes, and even confident
predictions can be wrong.

5. DECISION AUDIT / LOGGING

Allow me to select an advisor case and inspect its saved decision record.

Show clearly:
- model
- risk score
- threshold
- advisor-review flag
- observed evidence
- recommended academic plan
- constraints / planning assumptions
- validation result
- human approval required
- autonomous student contact allowed = false
- autonomous course registration allowed = false

This section should make it easy for me to answer the professor's question:

"Show us what gets logged when it makes a decision."

6. REPRODUCIBILITY

Briefly show:
- training rows
- held-out rows
- train/test student overlap = 0
- exported-results manifest if useful
- results are derived artifacts
- professor-provided source CSV files remain excluded from Git

OPTIONAL: THRESHOLD SENSITIVITY EXPLORER

If it can be implemented simply, add an interactive slider that lets me
demonstrate how changing a hypothetical classification threshold affects:
- precision
- recall
- false positives
- false negatives

IMPORTANT:
This must be clearly labeled as a POST-HOC EDUCATIONAL VISUALIZATION.

The official project threshold remains locked at 0.50.

The slider:
- must not modify the saved model
- must not modify project outputs
- must not overwrite results
- must not be presented as model tuning
- must not affect any final reported metric

Only add this if the required held-out prediction data already available in
results/ is sufficient. Do not weaken reproducibility just to support it.

DEPENDENCIES

If Streamlit is not currently installed:

1. Add an appropriate Streamlit dependency to requirements.txt.
2. Install the updated requirements into the currently active Python virtual
   environment using:

   python -m pip install -r requirements.txt

3. Update requirements-lock.txt from the resulting environment.

Do not globally install Python packages.

DOCUMENTATION

Create:

docs/DEMO.md

It should include:
- exact command to launch the dashboard
- recommended presentation/demo sequence
- recommended cases to show
- what to explain in each section
- a short Responsible AI / human-oversight statement
- reminder that the threshold explorer, if present, is post-hoc only

Keep this concise and practical.

CODE QUALITY

- Keep the implementation simple.
- Avoid unnecessary helper modules unless they improve readability.
- Do not add authentication.
- Do not add a database.
- Do not add external APIs.
- Do not add deployment infrastructure.
- Do not add animations or complicated custom CSS.
- Do not add the optional agentic AI extension.
- Do not modify raw professor-provided CSV files.
- Do not require those raw files merely to VIEW already-exported dashboard
  results where the results/ artifacts are sufficient.
- Handle missing result files with a clear user-facing message.

VERIFICATION

After implementation:

1. Run:
   python -m unittest discover -s tests -v

2. Confirm all existing tests still pass.

3. If you add tests, run them as well.

4. Verify the Streamlit application can start successfully.
   You may use a headless/local launch for verification and stop it afterward.

5. Inspect git diff and confirm that no required ML, evaluation, fairness,
   planner, or validator logic was changed.

6. Do NOT commit or push anything yet.

At the end, give me a concise report containing:
- files created
- files modified
- dependencies added
- test results
- Streamlit launch result
- confirmation that required backend logic was unchanged
- exact command I should use to launch the demo
- anything I should manually inspect before committing

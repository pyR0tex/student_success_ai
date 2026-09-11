Perform one final presentation-focused cleanup of the Streamlit dashboard on
the demo-polish branch.

Read AGENTS.md and docs/DEMO.md first.

IMPORTANT:
The backend is complete. Do not modify any ML, baseline, evaluation, fairness,
planner, validator, case-study, logging, or result-generation logic.

Do not change any final metrics or saved project results.

Only modify:

- the Streamlit presentation layer
- demo-specific tests if needed
- docs/DEMO.md
- remove the temporary docs/CODEX_DEMO_TASK.md file

The goal is to make the dashboard easier to present live:
less dense, less repetitive, easier to scan, and still technically accurate.

==================================================
GENERAL PRESENTATION / WORDING RULES
==================================================

The current dashboard has good information but is too dense in several places.

Apply these rules throughout the dashboard:

1. Prefer short presentation wording over report-style paragraphs.

2. Visible text should usually be:
   - one short sentence
   - one short callout
   - a few bullets
   - a compact metric/card

3. Move long explanations into collapsed expanders.

4. Avoid repeating information already visible elsewhere on the same page.

5. Do not explain implementation details unless they help the audience
   understand the result.

6. Keep important Responsible AI language, but make it concise.

7. Replace raw technical labels with audience-friendly wording where possible.

Examples:

Instead of:
"The risk score is a statistical early signal, not a diagnosis or explanation
of why a student may need support. Important context available to a real advisor
may be absent from the synthetic data."

Prefer:
"Risk scores are early signals, not diagnoses. Real advisor context may be
missing from the synthetic data."

Instead of:
"Courses are limited to T6_Spring/T7_Fall offerings and must satisfy
prerequisites."

Prefer:
"Courses must be offered that term and satisfy prerequisites."

Instead of:
"Spring selections may satisfy Fall prerequisites under the stated
successful-completion assumption."

Prefer:
"Spring courses can unlock Fall courses only if successfully completed."

Instead of:
"Academic and engagement indicators are signals, not causes. Even confident
predictions can be wrong."

Keep this wording. It is concise and important.

8. Use bullets instead of paragraphs when listing multiple constraints or
   limitations.

9. Do not expose unnecessary field names such as:
   - Maximum_Recommended_Credits
   - advisor_review_flag
   - autonomous_student_contact_allowed
     unless showing the raw audit log.

Use presentation labels instead:

- Credit limit
- Early-signal advisor flag
- Autonomous student contact

10. Format risk scores as percentages with approximately one decimal place.

Examples:
0.494534 -> 49.5%
0.775765 -> 77.6%
0.183700 -> 18.4%
0.902924 -> 90.3%

Do not change stored values.

11. Use YES / NO / REQUIRED / NOT ALLOWED rather than raw true / false values
    in the main presentation UI.

12. Avoid excessive decimal precision everywhere in the presentation UI.

13. Detailed raw values may remain available in expandable technical sections.

==================================================
PROJECT OVERVIEW
==================================================

Keep prominently visible:

- Selected model: Logistic Regression
- 20 fewer false negatives
- 16 additional false positives
- 12/12 required academic plans passed validation
- held-out model comparison

Change metric/chart ordering to:

Recall -> F1 -> Precision

Recall should be visually emphasized because missing a student who may need
support is an important project concern.

The main visible explanation should be concise.

Use wording similar to:

"Logistic Regression improved recall and F1 over the rule baseline.
It caught 20 more intervention cases, with 16 additional false positives."

Do not add a long paragraph.

Move the entire model-selection explanation into a collapsed expander:

"Why was Logistic Regression selected?"

Inside retain:

- 5-fold stratified group cross-validation
- Student_ID grouping
- training data only
- model comparison
- F1 selection rule
- recall tie-breaker
- Logistic Regression vs Random Forest vs Gradient Boosting

Keep the technical detail intact inside the expander.

==================================================
ADVISOR CASE EXPLORER
==================================================

Keep Case_02 and Case_06 presentation shortcuts.

Use:

- Case_06 as the main early-signal example
- Case_02 as the main academic-planning / prerequisite example

Rename:

"Advisor-review flag"
to
"Early-signal advisor flag"

Use wording:

"Academic plan requires advisor review"

instead of a generic "human review required" label where the context is the
academic plan.

This distinction must remain clear:

- the early-signal flag depends on the 0.50 threshold
- every academic plan requires advisor review

---

## CASE SUMMARY

At the top of each case, show a compact summary such as:

Risk score
Threshold
Early-signal advisor flag
Baseline risk indicators

Use percentage formatting.

Do not display unnecessary precision.

---

## CASE_02

Risk score is approximately 49.5%, just below the locked 50% threshold.

Show a visible concise warning:

"Near-threshold result — treat with extra uncertainty."

Do not change the threshold or classification.

Make the prerequisite relationship visually compact:

Program_A_C02
T6 Spring
↓ if successfully completed
Program_A_C04 + Program_A_C05
T7 Fall

Equivalent compact visual layouts are acceptable.

Avoid repeating the same prerequisite explanation multiple times.

Use a concise note:

"Spring courses can unlock Fall courses only if successfully completed."

---

## CASE_06

Show prominently:

Risk score: 77.6%
Threshold: 50%
Early-signal advisor flag: YES
Baseline risk indicators: 5

Keep these three observed signals visible:

- 7 days since last LMS activity
- 68% assignment submission rate
- GPA change: -0.40

Avoid a dense technical table if these can be shown cleanly as three rows,
cards, or bullets.

Keep the validated Spring/Fall academic plan visible.

Keep the important cross-term prerequisite relationship visible.

---

## PLANNER DETAILS FOR ALL CASES

The main visible planner section should communicate only:

- prerequisites
- course availability
- previously completed courses
- degree progress
- credit limits

Use short bullets.

Move the full detailed constraints into a collapsed expander:

"Planner constraints and assumptions"

Do not remove the full underlying information.

Replace long visible uncertainty paragraphs with a short message such as:

"Risk scores are early signals, not diagnoses. Academic plans require advisor
review. Fall recommendations must be revisited if required Spring courses are
not successfully completed."

Move longer uncertainty text into:

"Full uncertainty and limitations"

collapsed by default.

==================================================
FAIRNESS / ERROR-PATTERN AUDIT
==================================================

Keep this page simple.

Order the chart/legend:

Recall
False-positive rate
False-negative rate

Make the main finding prominent:

"Main finding: recall is similar across groups, but Audit_C has the highest
false-positive rate (43.52%)."

Below it, use a very short disclaimer such as:

"Audit_Group was not used for prediction. These synthetic groups support
error-pattern analysis, not claims of real-world demographic fairness."

Do not turn this into a paragraph-heavy section.

Move the exact subgroup table into a collapsed expander:

"Exact subgroup metrics"

Keep the full table available.

==================================================
FAILURE ANALYSIS
==================================================

Keep the side-by-side False Negative vs False Positive layout.

Use:

False Negative
Student_1383
Risk score: 18.4%

False Positive
Student_0880
Risk score: 90.3%

Replace raw labels:

Predicted result: 0
Actual result: 1

with:

Predicted: No intervention needed
Actual: Intervention needed

and the reverse for the false positive.

Keep this message prominently visible:

"Academic and engagement indicators are signals, not causes. Even confident
predictions can be wrong."

Do not show all raw observed features by default.

For Student_1383, surface several useful signals:

- GPA change: +0.74
- pass rate: 86.7%
- failed courses: 0
- assignment submission: 100%
- days since last LMS activity: 0

For Student_0880, surface:

- GPA change: -1.34
- failed courses: 1
- withdrawals: 1
- assignment submission: about 56%
- average assignment score: 58.5

Use exact exported values.

Put complete feature tables inside:

"View all observed features"

Use short visible consequence statements.

False negative:

"Risk: a student who needs support could be missed or reviewed late."

False positive:

"Risk: unnecessary advisor review or avoidable concern."

Use a short visible lesson:

"Lesson: model signals are incomplete, so advisor context remains necessary."

Move longer interpretation and improvement paragraphs into collapsed details.

==================================================
DECISION AUDIT / LOGGING
==================================================

This page should become much shorter.

Use Case_06 as the default decision record.

At the top, show:

Model: Logistic Regression
Risk score: 77.6%
Threshold: 50%
Early-signal advisor flag: YES
Plan validation: PASSED

Then prominently show three safety controls:

Human approval
REQUIRED

Autonomous student contact
NOT ALLOWED

Autonomous course registration
NOT ALLOWED

These should be visible without scrolling.

Do not repeat the full Advisor Case Explorer content by default.

Move details into collapsed expanders:

"Evidence recorded"

"Plan and constraints recorded"

"Full raw decision log"

The full raw JSON log MUST be collapsed by default.

Keep it available for Q&A because the professor may ask:

"Show us what gets logged when it makes a decision."

Move the note about:

- no distinct model-version field
- no completed-human-approval field

into a secondary detail area.

Do not add or invent backend fields.

==================================================
REPRODUCIBILITY
==================================================

Keep visible:

Training rows: 5,148
Held-out rows: 1,452
Train/test student overlap: 0

Use one concise explanation:

"Held-out students were separate from training students, and the test set was
used only for final evaluation."

Add one short line if useful:

"The dashboard reads tracked derived results rather than raw source data."

Move detailed methodology and export timestamp into:

"Reproducibility details"

Keep the exported-results manifest collapsed.

REMOVE the visible message explaining that the threshold sensitivity explorer
was omitted.

Do not add a threshold explorer.

==================================================
SIDEBAR
==================================================

Keep:

- section navigation
- official threshold: 0.50 locked
- concise human-oversight statement

Shorten the oversight text if needed to something like:

"Human review required. No autonomous contact or registration."

==================================================
DOCUMENTATION
==================================================

Update docs/DEMO.md to match the final UI.

Recommended live sequence:

1. Project overview
2. Case_06 — early-signal example
3. Case_02 — academic-planning / prerequisite example
4. Fairness audit
5. Failure analysis
6. Decision audit / logging

Treat Reproducibility as backup / Q&A unless time allows.

Target approximately 5–7 minutes.

docs/DEMO.md should be concise and practical, not report-style documentation.

==================================================
REPOSITORY CLEANUP
==================================================

Delete:

docs/CODEX_DEMO_TASK.md

It was a temporary implementation instruction file.

Keep:

docs/DEMO.md

==================================================
VERIFICATION
==================================================

After making changes:

1. Run:

python -m unittest discover -s tests -v

2. Confirm all existing tests still pass.

3. Start the Streamlit app and confirm it loads without errors.

4. Inspect git diff.

5. Confirm that no required backend files were modified, including:
   - model training
   - baseline
   - evaluation
   - fairness
   - planner
   - validator
   - case-study generation
   - decision logging
   - tracked final results

6. Do not change saved metrics or results.

7. Do not commit or push automatically.

At the end give me a concise summary:

- files modified
- files deleted
- test results
- Streamlit launch result
- confirmation that backend logic/results were unchanged
- anything I should manually inspect before committing

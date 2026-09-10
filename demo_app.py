"""Read-only presentation of the project's exported synthetic results."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SECTIONS = (
    "Project overview",
    "Advisor case explorer",
    "Fairness / error-pattern audit",
    "Failure analysis",
    "Decision audit / logging",
    "Reproducibility",
)


def read_result(relative_path):
    path = RESULTS_DIR / relative_path
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path, keep_default_na=False)
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        return json.loads(content)
    except (OSError, ValueError) as error:
        st.error(f"Cannot read results/{relative_path}: {error}")
        st.info(
            "Restore the tracked results/ artifacts from a complete project checkout "
            "and reload the dashboard. Viewing the demo requires no raw CSV files."
        )
        st.stop()


def show_evidence(evidence):
    st.subheader("Observed evidence")
    st.dataframe(
        pd.DataFrame(evidence).rename(columns={
            "label": "Observed signal",
            "observed_value": "Observed value",
            "direction": "Effect on model score",
            "contribution": "Saved log-odds contribution",
        }).drop(columns="feature", errors="ignore"),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Saved local model contributions describe score direction, not personal "
        "causes or a complete explanation of student need."
    )


def show_validation(passed, issues):
    if passed is True and not issues:
        st.success("Independent plan validation: PASSED (saved result; advisor review required).")
    else:
        st.error("Independent plan validation: NOT PASSED. Do not treat this plan as feasible.")
        st.write(issues or "A passing validation result is unavailable.")


def case_label(case_id):
    notes = {
        "Case_02": "Spring prerequisite enables Fall courses",
        "Case_06": "Flagged case with observable risk signals",
    }
    return f"{case_id} — {notes[case_id]}" if case_id in notes else case_id


def overview(summary):
    st.header("Project overview")
    st.write(f"Selected model: **{summary['selected_model']}**")
    comparison = summary["heldout_comparison"]
    columns = st.columns(3)
    columns[0].metric("ML reduced false negatives by", comparison["false_negatives_reduced_by"])
    columns[1].metric("ML added false positives", comparison["false_positives_added"])
    planner = summary["planner"]
    columns[2].metric("Required plans passed validation", f"{planner['valid_plans']}/{planner['cases']}")
    st.subheader("Held-out baseline vs ML")
    metrics = pd.DataFrame({
        "Rule-Based Baseline": comparison["baseline"],
        summary["selected_model"]: comparison["machine_learning"],
    }).loc[["precision", "recall", "f1"]].rename(index={"f1": "F1"})
    st.bar_chart(metrics, stack=False, y_label="Score (0–1)")
    st.dataframe(metrics.style.format("{:.2%}"), width="stretch")
    st.caption("Fewer missed positives came with more unnecessary flags; predictions still require review.")
    development = read_result("model/model_development_summary.json")
    st.subheader("Why Logistic Regression?")
    st.write(development["validation_method"] + "; training data only.")
    st.write(development["selection_rule"])
    st.dataframe(
        pd.DataFrame(development["models"])[["model", "precision", "recall", "f1"]]
        .set_index("model").style.format("{:.2%}"),
        width="stretch",
    )
    st.write("Logistic Regression achieved the highest development F1 and recall of the three candidates.")


def case_explorer():
    st.header("Advisor case explorer")
    cases = {case["Case_ID"]: case for case in read_result("cases/advisor_case_studies.json")}
    plans = {plan["Case_ID"]: plan for plan in read_result("planner/advisor_case_plans.json")}
    courses = read_result("planner/advisor_case_course_plans.csv")
    st.session_state.setdefault("advisor_case", "Case_02")
    shortcuts = st.columns(2)
    if shortcuts[0].button("Show Case_02: prerequisites"):
        st.session_state["advisor_case"] = "Case_02"
    if shortcuts[1].button("Show Case_06: risk signals"):
        st.session_state["advisor_case"] = "Case_06"
    case_ids = sorted(cases)
    selected = st.selectbox(
        "Advisor case", case_ids,
        format_func=case_label, key="advisor_case",
    )
    case = cases[selected]
    plan = plans[selected]
    signal = case["Early_Signal"]
    st.subheader(f"{case['Case_ID']} · {case['Student_ID']} · {case['Program_ID']}")
    columns = st.columns(3)
    columns[0].metric("ML risk score", f"{signal['Risk_Score']:.6f}")
    columns[1].metric("Locked decision threshold", f"{signal['Decision_Threshold']:.2f}")
    columns[2].metric("Baseline risk-indicator count", signal["Baseline_Risk_Count"])
    st.write(f"**Advisor-review flag:** {str(signal['Advisor_Review_Flag']).lower()}")
    show_evidence(signal["Observed_Evidence"])
    st.subheader("Saved recommended academic plan")
    show_validation(plan.get("Validation_Passed"), plan.get("Validation_Issues", []))
    st.caption("This displays the backend's independent validation; the dashboard does not generate or revalidate plans.")
    for column, term in zip(st.columns(2), ("T6_Spring", "T7_Fall")):
        with column:
            st.metric(f"{term} credits", plan[f"{term}_Credits"])
            st.write(", ".join(plan[term]) or "No courses recommended")
            term_rows = courses[(courses["Case_ID"] == selected) & (courses["Term_ID"] == term)]
            st.dataframe(
                term_rows[["Course_ID", "Credits", "Prerequisites"]],
                hide_index=True, width="stretch",
            )
    st.subheader("Cross-term prerequisite relationships")
    relationships = []
    fall_rows = courses[(courses["Case_ID"] == selected) & (courses["Term_ID"] == "T7_Fall")]
    for course in fall_rows.itertuples(index=False):
        for prerequisite in str(course.Prerequisites).split(";"):
            if prerequisite in plan["T6_Spring"]:
                relationships.append(
                    f"{prerequisite} (Spring) → if successfully completed, satisfies prerequisite → "
                    f"{course.Course_ID} (Fall)"
                )
    for relationship in relationships:
        st.info(relationship)
    if not relationships:
        st.write("No Spring-to-Fall prerequisite links appear in this case's exported course rows.")
    st.warning("Successful Spring completion is not guaranteed. Revisit Fall selections if it does not occur.")
    st.subheader("Planning constraints / assumptions")
    st.write(f"**Maximum_Recommended_Credits:** {plan['Maximum_Recommended_Credits']} per term")
    for note in case["Academic_Plan"]["Constraints_and_Assumptions"]:
        st.write(note)
    st.write(plan["Planning_Assumption"])
    st.write(
        "Successfully completed courses cannot be recommended again. Prerequisites must "
        "be completed before the dependent term; same-term completion does not count. "
        "Courses must be available that term and contribute to remaining degree requirements."
    )
    st.subheader("Uncertainty and human review")
    st.write(case["Uncertainty_and_Oversight"])
    st.write(f"**Human review required:** {str(case['Human_Review_Required']).lower()}")


def fairness(summary):
    st.header("Fairness / error-pattern audit")
    rates = pd.DataFrame(summary["fairness_error_pattern_audit"]).set_index("Audit_Group")
    chart = rates[["recall", "false_positive_rate", "false_negative_rate"]].rename(columns={
        "recall": "Recall", "false_positive_rate": "False-positive rate",
        "false_negative_rate": "False-negative rate",
    })
    st.bar_chart(chart, stack=False, y_label="Rate (0–1)")
    st.dataframe(chart.style.format("{:.2%}"), width="stretch")
    st.warning(
        "Recall is relatively similar across the synthetic groups, but Audit_C "
        "has a noticeably higher false-positive rate."
    )
    st.write(
        "Audit_Group was never a predictive feature. These groups are synthetic. "
        "This post-hoc analysis identifies subgroup error patterns; it does not "
        "establish real-world demographic fairness."
    )


def failures(summary):
    st.header("Failure analysis")
    st.caption(summary["failure_analysis"]["selection_rule"])
    st.warning("Academic and engagement indicators are signals, not causes. Even confident predictions can be wrong.")
    for column, failure in zip(st.columns(2), read_result("model/failure_analysis.json")):
        with column:
            st.subheader(f"{failure['Error_Type']} · {failure['Student_ID']}")
            st.caption(f"Observation term: {failure['Observation_Term']}")
            st.metric("ML risk score", f"{failure['ML_Risk_Score']:.6f}")
            st.write(f"**Predicted result:** {failure['ML_Prediction']}")
            st.write(f"**Actual result:** {failure['Actual_Outcome']}")
            st.caption("Synthetic target: 1 = intervention needed within eight weeks; 0 = not needed.")
            st.dataframe(
                pd.DataFrame(failure["Observed_Features"].items(), columns=["Observed feature", "Value"]),
                hide_index=True, width="stretch",
            )
            st.write(failure["Interpretation"])
            st.write("**Potential consequence:** " + failure["Potential_Consequence"])
            st.write("**Lesson / potential improvement:** " + failure["Potential_Improvement"])


def decision_audit():
    st.header("Decision audit / logging")
    records = {record["case_id"]: record for record in read_result("logs/advisor_case_decision_log.jsonl")}
    selected = st.selectbox("Saved decision record", sorted(records), format_func=case_label)
    record = records[selected]
    st.subheader(f"{record['case_id']} · {record['student_id']}")
    st.write(f"**Model:** {record['model']} · **Saved at:** {record['timestamp_utc']}")
    columns = st.columns(3)
    columns[0].metric("Risk score", f"{record['risk_score']:.6f}")
    columns[1].metric("Threshold", f"{record['decision_threshold']:.2f}")
    columns[2].metric("Advisor-review flag", str(record["advisor_review_flag"]).lower())
    show_evidence(record["observed_evidence"])
    st.subheader("Saved recommended plan")
    show_validation(record.get("plan_validation_passed"), record.get("plan_validation_issues", []))
    st.json(record["recommended_plan"], expanded=True)
    st.subheader("Constraints / planning assumptions")
    for note in record["constraints_and_assumptions"]:
        st.write(note)
    for field in (
        "human_approval_required", "autonomous_student_contact_allowed",
        "autonomous_course_registration_allowed",
    ):
        st.write(f"**{field.replace('_', ' ')}:** {str(record[field]).lower()}")
    st.caption(
        "The saved record names the model and requires approval. It does not record "
        "a distinct model version or completed human approval; neither is inferred here."
    )
    with st.expander("Full saved decision record", expanded=True):
        st.json(record, expanded=True)


def reproducibility(summary):
    st.header("Reproducibility")
    dataset = summary["dataset"]
    columns = st.columns(3)
    columns[0].metric("Training rows", dataset["training_rows"])
    columns[1].metric("Held-out rows", dataset["test_rows"])
    columns[2].metric("Train/test student overlap", dataset["train_test_student_overlap"])
    st.write(
        "Results are derived artifacts from the completed backend. Professor-provided "
        "source CSV files remain excluded from Git. This demo reads only tracked results/ "
        "files and needs neither the raw data nor the fitted model."
    )
    st.write(
        "Student_ID is an identifier and Audit_Group is for post-hoc auditing; neither "
        "is a predictive feature. The held-out data is reserved for final evaluation "
        "and is never used to fit models, select hyperparameters, tune thresholds, or engineer rules."
    )
    st.caption(f"Results exported at: {summary['generated_at_utc']}")
    st.info(
        "Threshold sensitivity explorer omitted: results/ does not contain the full "
        "held-out predictions needed for this optional post-hoc educational visualization. "
        "The official threshold remains locked at 0.50."
    )
    with st.expander("Exported-results manifest"):
        manifest = read_result("manifest.json")
        st.write(manifest["note"])
        st.dataframe(pd.DataFrame(manifest["files"]), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Student Success · Course Demo", layout="wide")
    st.title("Student Success Early-Signal & Academic Planning System")
    st.info("Early signals for advisor review, not diagnoses or autonomous academic decisions.")
    summary = read_result("presentation_data.json")
    st.sidebar.title("Presentation")
    section = st.sidebar.radio("Section", SECTIONS)
    st.sidebar.write(f"Official threshold: **{summary['decision_threshold']:.2f} (locked)**")
    st.sidebar.caption("Human approval required. No autonomous student contact or course registration.")
    try:
        if section == SECTIONS[0]:
            overview(summary)
        elif section == SECTIONS[1]:
            case_explorer()
        elif section == SECTIONS[2]:
            fairness(summary)
        elif section == SECTIONS[3]:
            failures(summary)
        elif section == SECTIONS[4]:
            decision_audit()
        else:
            reproducibility(summary)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        st.error(f"The exported results have an unexpected structure: {error}")
        st.info("Restore a complete, consistent tracked results/ export and reload the dashboard.")


if __name__ == "__main__":
    main()

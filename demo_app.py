"""Read-only presentation of the project's exported synthetic results."""

import json
from pathlib import Path

import altair as alt
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
    for item in evidence:
        st.write(f"- **{item['label'].capitalize()}:** {item['observed_value']}")
    with st.expander("Evidence details"):
        st.dataframe(pd.DataFrame(evidence), hide_index=True, width="stretch")
        st.caption("Saved model contributions describe score direction, not personal causes.")


def show_validation(passed, issues):
    if passed is True and not issues:
        st.success("Independent plan validation: PASSED. Academic plan requires advisor review.")
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
    }).loc[["recall", "f1", "precision"]].rename(index={"recall": "Recall", "f1": "F1", "precision": "Precision"})
    st.info(
        "Logistic Regression improved recall and F1 over the rule baseline. "
        f"It caught {comparison['false_negatives_reduced_by']} more intervention cases, "
        f"with {comparison['false_positives_added']} additional false positives."
    )
    st.markdown("**Recall — finding students who may need support**")
    chart_data = metrics.rename_axis("Metric").reset_index().melt("Metric", var_name="Model", value_name="Score")
    st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Metric:N", sort=["Recall", "F1", "Precision"]),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset="Model:N", color="Model:N",
        tooltip=["Metric", "Model", alt.Tooltip("Score:Q", format=".1%")],
    ), width="stretch")
    st.dataframe(metrics.style.format("{:.1%}"), width="stretch")
    with st.expander("Why was Logistic Regression selected?"):
        development = read_result("model/model_development_summary.json")
        st.write(development["validation_method"] + "; training data only.")
        st.write(development["selection_rule"])
        st.dataframe(
            pd.DataFrame(development["models"])[["model", "recall", "f1", "precision"]]
            .rename(columns={"recall": "Recall", "f1": "F1", "precision": "Precision"})
            .set_index("model").style.format("{:.1%}"), width="stretch",
        )
        st.write("Logistic Regression achieved the highest development F1 and recall of the three candidates.")


def case_explorer():
    st.header("Advisor case explorer")
    cases = {case["Case_ID"]: case for case in read_result("cases/advisor_case_studies.json")}
    plans = {plan["Case_ID"]: plan for plan in read_result("planner/advisor_case_plans.json")}
    courses = read_result("planner/advisor_case_course_plans.csv")
    st.session_state.setdefault("advisor_case", "Case_06")
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
    columns = st.columns(4)
    columns[0].metric("Risk score", f"{signal['Risk_Score']:.1%}")
    columns[1].metric("Threshold", f"{signal['Decision_Threshold']:.0%}")
    columns[2].metric("Early-signal advisor flag", "YES" if signal["Advisor_Review_Flag"] else "NO")
    columns[3].metric("Baseline risk indicators", signal["Baseline_Risk_Count"])
    st.caption("The early-signal flag uses the locked 50% threshold. Every academic plan requires advisor review.")
    if selected == "Case_02":
        st.warning("Near-threshold result — treat with extra uncertainty.")
    show_evidence(signal["Observed_Evidence"])
    st.subheader("Saved recommended academic plan")
    show_validation(plan.get("Validation_Passed"), plan.get("Validation_Issues", []))
    for column, term in zip(st.columns(2), ("T6_Spring", "T7_Fall")):
        with column:
            st.metric(f"{term.replace('_', ' ')} credits", plan[f"{term}_Credits"])
            term_rows = courses[(courses["Case_ID"] == selected) & (courses["Term_ID"] == term)]
            st.dataframe(
                term_rows[["Course_ID", "Credits"]].rename(columns={"Course_ID": "Course"}),
                hide_index=True, width="stretch",
            )
    st.subheader("Cross-term prerequisite relationships")
    relationships = {}
    fall_rows = courses[(courses["Case_ID"] == selected) & (courses["Term_ID"] == "T7_Fall")]
    for course in fall_rows.itertuples(index=False):
        for prerequisite in str(course.Prerequisites).split(";"):
            if prerequisite in plan["T6_Spring"]:
                relationships.setdefault(prerequisite, []).append(course.Course_ID)
    for prerequisite, dependents in relationships.items():
        st.info(f"{prerequisite} · T6 Spring\n\n↓ if successfully completed\n\n{' + '.join(dependents)} · T7 Fall")
    if relationships:
        st.caption("Spring courses can unlock Fall courses only if successfully completed.")
    else:
        st.caption("No Spring-to-Fall prerequisite links in this plan.")
    st.subheader("Planner checks")
    st.markdown(
        "- Prerequisites completed before the dependent term\n"
        "- Courses offered that term\n"
        "- Previously completed courses excluded\n"
        "- Courses support remaining degree requirements\n"
        f"- Credit limit: {plan['Maximum_Recommended_Credits']} per term"
    )
    with st.expander("Planner constraints and assumptions"):
        for note in case["Academic_Plan"]["Constraints_and_Assumptions"]:
            st.write(note)
        st.write(plan["Planning_Assumption"])
        st.write("Same-term prerequisite completion does not count. Successfully completed courses cannot be recommended again.")
        st.dataframe(courses[courses["Case_ID"] == selected], hide_index=True, width="stretch")
        st.caption("Validation is the saved independent backend result; the dashboard does not generate or revalidate plans.")
    st.write(
        "Risk scores are early signals, not diagnoses. Academic plans require advisor review. "
        "Fall recommendations must be revisited if required Spring courses are not successfully completed."
    )
    with st.expander("Full uncertainty and limitations"):
        st.write(case["Uncertainty_and_Oversight"])


def fairness(summary):
    st.header("Fairness / error-pattern audit")
    rates = pd.DataFrame(summary["fairness_error_pattern_audit"]).set_index("Audit_Group")
    chart = rates[["recall", "false_positive_rate", "false_negative_rate"]].rename(columns={
        "recall": "Recall", "false_positive_rate": "False-positive rate",
        "false_negative_rate": "False-negative rate",
    })
    order = ["Recall", "False-positive rate", "False-negative rate"]
    chart_data = chart.reset_index().melt("Audit_Group", var_name="Metric", value_name="Rate")
    st.warning(
        "Main finding: recall is similar across groups, but Audit_C has the highest "
        f"false-positive rate ({rates.loc['Audit_C', 'false_positive_rate']:.2%})."
    )
    st.write("Audit_Group was not used for prediction. These synthetic groups support error-pattern analysis, not claims of real-world demographic fairness.")
    st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Audit_Group:N", title="Audit group"),
        y=alt.Y("Rate:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset=alt.XOffset("Metric:N", sort=order),
        color=alt.Color("Metric:N", sort=order, scale=alt.Scale(domain=order)),
        tooltip=["Audit_Group", "Metric", alt.Tooltip("Rate:Q", format=".2%")],
    ), width="stretch")
    with st.expander("Exact subgroup metrics"):
        st.dataframe(rates, width="stretch")


def failures(summary):
    st.header("Failure analysis")
    st.warning("Academic and engagement indicators are signals, not causes. Even confident predictions can be wrong.")
    for column, failure in zip(st.columns(2), read_result("model/failure_analysis.json")):
        with column:
            st.subheader(failure["Error_Type"])
            st.write(failure["Student_ID"])
            st.metric("Risk score", f"{failure['ML_Risk_Score']:.1%}")
            outcomes = {0: "No intervention needed", 1: "Intervention needed"}
            st.write(f"**Predicted:** {outcomes[failure['ML_Prediction']]}")
            st.write(f"**Actual:** {outcomes[failure['Actual_Outcome']]}")
            features = failure["Observed_Features"]
            st.write(f"- GPA change: {features['GPA_Change']:+.2f}")
            if failure["Error_Type"] == "False Negative":
                st.write(f"- Pass rate: {features['Course_Pass_Rate_Last_2_Terms']:.1%}")
            st.write(f"- Failed courses: {features['Failed_Courses_Last_2_Terms']}")
            if failure["Error_Type"] == "False Positive":
                st.write(f"- Withdrawals: {features['Withdrawals_Last_2_Terms']}")
            st.write(f"- Assignment submission: {features['Assignment_Submission_Rate_4wk']:.1%}")
            if failure["Error_Type"] == "False Negative":
                st.write(f"- Days since last LMS activity: {features['Days_Since_Last_LMS_Activity']}")
                st.write("Risk: a student who needs support could be missed or reviewed late.")
            else:
                st.write(f"- Average assignment score: {features['Average_Assignment_Score_4wk']:.1f}")
                st.write("Risk: unnecessary advisor review or avoidable concern.")
            with st.expander("View all observed features"):
                st.caption(f"Observation term: {failure['Observation_Term']}")
                st.dataframe(pd.DataFrame(features.items(), columns=["Observed feature", "Value"]), hide_index=True, width="stretch")
            with st.expander("Interpretation and improvements"):
                st.write(failure["Interpretation"])
                st.write(failure["Potential_Consequence"])
                st.write(failure["Potential_Improvement"])
    st.info("Lesson: model signals are incomplete, so advisor context remains necessary.")
    with st.expander("Failure example selection"):
        st.write(summary["failure_analysis"]["selection_rule"])


def decision_audit():
    st.header("Decision audit / logging")
    records = {record["case_id"]: record for record in read_result("logs/advisor_case_decision_log.jsonl")}
    selected = st.selectbox("Saved decision record", sorted(records), index=sorted(records).index("Case_06"), format_func=case_label)
    record = records[selected]
    st.write(f"**Model:** {record['model']}")
    columns = st.columns(4)
    columns[0].metric("Risk score", f"{record['risk_score']:.1%}")
    columns[1].metric("Threshold", f"{record['decision_threshold']:.0%}")
    columns[2].metric("Early-signal advisor flag", "YES" if record["advisor_review_flag"] else "NO")
    passed = record.get("plan_validation_passed") is True and not record.get("plan_validation_issues", [])
    columns[3].metric("Plan validation", "PASSED" if passed else "NOT PASSED")
    columns = st.columns(3)
    columns[0].metric("Human approval", "REQUIRED" if record["human_approval_required"] else "NOT REQUIRED")
    columns[1].metric("Autonomous student contact", "ALLOWED" if record["autonomous_student_contact_allowed"] else "NOT ALLOWED")
    columns[2].metric("Autonomous course registration", "ALLOWED" if record["autonomous_course_registration_allowed"] else "NOT ALLOWED")
    with st.expander("Evidence recorded"):
        show_evidence(record["observed_evidence"])
    with st.expander("Plan and constraints recorded"):
        show_validation(record.get("plan_validation_passed"), record.get("plan_validation_issues", []))
        st.json(record["recommended_plan"])
        for note in record["constraints_and_assumptions"]:
            st.write(note)
    with st.expander("Full raw decision log"):
        st.json(record)
    with st.expander("Record metadata and limitations"):
        st.write(f"{record['case_id']} · {record['student_id']} · Saved at: {record['timestamp_utc']}")
        st.caption("The saved record has no distinct model-version field or completed-human-approval field; neither is inferred here.")


def reproducibility(summary):
    st.header("Reproducibility")
    dataset = summary["dataset"]
    columns = st.columns(3)
    columns[0].metric("Training rows", f"{dataset['training_rows']:,}")
    columns[1].metric("Held-out rows", f"{dataset['test_rows']:,}")
    columns[2].metric("Train/test student overlap", dataset["train_test_student_overlap"])
    st.write("Held-out students were separate from training students, and the test set was used only for final evaluation.")
    st.caption("The dashboard reads tracked derived results rather than raw source data.")
    with st.expander("Reproducibility details"):
        st.write("Student_ID is an identifier and Audit_Group is for post-hoc auditing; neither is a predictive feature.")
        st.write("Held-out data was never used to fit models, select hyperparameters, tune thresholds, or engineer rules.")
        st.write("Source CSVs remain excluded from Git. The demo needs neither raw data nor a fitted model.")
        st.caption(f"Results exported at: {summary['generated_at_utc']}")
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
    st.sidebar.caption("Human review required. No autonomous contact or registration.")
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

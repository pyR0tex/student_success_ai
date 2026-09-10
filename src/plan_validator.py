from __future__ import annotations

import pandas as pd

PLANNING_TERMS = ("T6_Spring", "T7_Fall")


def validate_plan(
    student_id: str,
    plan: dict,
    students: pd.DataFrame,
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    prerequisites: pd.DataFrame,
    requirements: pd.DataFrame,
    offerings: pd.DataFrame,
) -> dict:
    """Validate the project-required academic-planning constraints independently."""
    issues: list[str] = []

    student_rows = students[students["Student_ID"] == student_id]
    if student_rows.empty:
        return {"valid": False, "issues": [f"Unknown student {student_id}"]}

    student = student_rows.iloc[0]
    program_id = str(student["Program_ID"])
    max_credits = int(student["Maximum_Recommended_Credits"])

    completed = set(
        history.loc[
            (history["Student_ID"] == student_id) & (history["Passed"] == 1),
            "Course_ID",
        ].astype(str)
    )

    catalog_lookup = catalog.set_index("Course_ID", drop=False)
    prereq_map = (
        prerequisites.groupby("Course_ID")["Prerequisite_Course_ID"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )
    available = {
        (str(row.Course_ID), str(row.Term_ID)): bool(row.Available)
        for row in offerings.itertuples(index=False)
    }

    seen: set[str] = set()
    completed_before_term = set(completed)

    req = requirements[requirements["Program_ID"] == program_id]
    remaining_core = set(
        req.loc[req["Requirement_Type"] == "Required_Course", "Course_ID"].astype(str)
    ) - completed
    elective_rows = req[req["Requirement_Type"] == "Choose_N"]
    elective_courses = set(elective_rows["Course_ID"].astype(str))
    elective_minimum = (
        int(elective_rows["Minimum_Courses_From_Group"].max())
        if not elective_rows.empty
        else 0
    )
    completed_electives = len(completed & elective_courses)
    electives_needed = max(0, elective_minimum - completed_electives)
    planned_electives_that_count = 0

    for term_id in PLANNING_TERMS:
        term_courses = [str(c) for c in plan.get(term_id, [])]
        term_credits = 0

        for course_id in term_courses:
            if course_id in seen:
                issues.append(f"{course_id} is scheduled more than once.")
                continue
            seen.add(course_id)

            if course_id in completed:
                issues.append(f"{course_id} was already successfully completed.")

            if course_id not in catalog_lookup.index:
                issues.append(f"{course_id} is not in the course catalog.")
                continue

            course = catalog_lookup.loc[course_id]
            if str(course["Program_ID"]) != program_id:
                issues.append(f"{course_id} does not belong to {program_id}.")

            term_credits += int(course["Credits"])

            if not available.get((course_id, term_id), False):
                issues.append(f"{course_id} is not available in {term_id}.")

            missing_prereqs = set(prereq_map.get(course_id, set())) - completed_before_term
            if missing_prereqs:
                issues.append(
                    f"{course_id} in {term_id} is missing prerequisites: {sorted(missing_prereqs)}."
                )

            if course_id in remaining_core:
                pass
            elif course_id in elective_courses and planned_electives_that_count < electives_needed:
                planned_electives_that_count += 1
            else:
                issues.append(
                    f"{course_id} does not contribute to a remaining degree requirement."
                )

        if term_credits > max_credits:
            issues.append(
                f"{term_id} has {term_credits} credits, exceeding the maximum of {max_credits}."
            )

        # Planning assumption: successful Spring completion may satisfy Fall prerequisites.
        completed_before_term |= set(term_courses)

    return {"valid": len(issues) == 0, "issues": issues}

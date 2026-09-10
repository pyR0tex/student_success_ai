from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import pandas as pd

from .config import METRICS_DIR, PLANS_DIR
from .data_loader import load_csv
from .plan_validator import validate_plan

PLANNING_TERMS = ("T6_Spring", "T7_Fall")


@dataclass(frozen=True)
class PlanChoice:
    spring: tuple[str, ...]
    fall: tuple[str, ...]
    score: tuple[int, int, int, int]


class AcademicPlanner:
    """Generate deterministic two-term academic plans from explicit constraints."""

    def __init__(self) -> None:
        self.students = load_csv("students.csv")
        self.history = load_csv("student_course_history.csv")
        self.catalog = load_csv("course_catalog.csv")
        self.prerequisites = load_csv("prerequisites.csv")
        self.requirements = load_csv("degree_requirements.csv")
        self.offerings = load_csv("future_course_offerings.csv")

        self._student_lookup = self.students.set_index("Student_ID", drop=False)
        self._catalog_lookup = self.catalog.set_index("Course_ID", drop=False)
        self._prereq_map = (
            self.prerequisites.groupby("Course_ID")["Prerequisite_Course_ID"]
            .apply(lambda s: tuple(sorted(set(s.dropna().astype(str)))))
            .to_dict()
        )
        self._available = {
            (str(row.Course_ID), str(row.Term_ID)): bool(row.Available)
            for row in self.offerings.itertuples(index=False)
        }

    def _student_row(self, student_id: str) -> pd.Series:
        if student_id not in self._student_lookup.index:
            raise KeyError(f"Unknown Student_ID: {student_id}")
        return self._student_lookup.loc[student_id]

    def completed_courses(self, student_id: str) -> set[str]:
        rows = self.history[
            (self.history["Student_ID"] == student_id) & (self.history["Passed"] == 1)
        ]
        return set(rows["Course_ID"].astype(str))

    def requirement_status(self, student_id: str) -> dict:
        student = self._student_row(student_id)
        program_id = str(student["Program_ID"])
        completed = self.completed_courses(student_id)
        req = self.requirements[self.requirements["Program_ID"] == program_id]

        required_core = set(
            req.loc[req["Requirement_Type"] == "Required_Course", "Course_ID"].astype(str)
        )
        elective_rows = req[req["Requirement_Type"] == "Choose_N"]
        elective_courses = set(elective_rows["Course_ID"].astype(str))
        elective_minimum = (
            int(elective_rows["Minimum_Courses_From_Group"].max())
            if not elective_rows.empty
            else 0
        )

        completed_core = completed & required_core
        completed_electives = completed & elective_courses
        elective_remaining = max(0, elective_minimum - len(completed_electives))

        return {
            "student_id": student_id,
            "program_id": program_id,
            "completed": completed,
            "required_core": required_core,
            "remaining_core": required_core - completed,
            "elective_courses": elective_courses,
            "completed_electives": completed_electives,
            "elective_minimum": elective_minimum,
            "elective_remaining": elective_remaining,
            "max_credits": int(student["Maximum_Recommended_Credits"]),
        }

    def _eligible_courses(
        self,
        status: dict,
        term_id: str,
        completed_before_term: set[str],
        already_planned: set[str],
    ) -> list[str]:
        remaining_core = status["remaining_core"] - already_planned

        planned_electives = len(already_planned & status["elective_courses"])
        elective_slots_left = max(0, status["elective_remaining"] - planned_electives)
        elective_candidates = (
            status["elective_courses"] - status["completed_electives"] - already_planned
            if elective_slots_left > 0
            else set()
        )

        candidates = remaining_core | elective_candidates
        eligible: list[str] = []
        for course_id in sorted(candidates):
            if course_id not in self._catalog_lookup.index:
                continue
            if str(self._catalog_lookup.loc[course_id, "Program_ID"]) != status["program_id"]:
                continue
            if not self._available.get((course_id, term_id), False):
                continue
            prereqs = set(self._prereq_map.get(course_id, ()))
            if not prereqs.issubset(completed_before_term):
                continue
            eligible.append(course_id)
        return eligible

    def _course_credits(self, course_id: str) -> int:
        return int(self._catalog_lookup.loc[course_id, "Credits"])

    def _feasible_subsets(
        self,
        course_ids: Iterable[str],
        max_credits: int,
        elective_courses: set[str],
        elective_slots_left: int,
    ) -> list[tuple[str, ...]]:
        courses = tuple(sorted(course_ids))
        feasible: list[tuple[str, ...]] = [tuple()]
        for r in range(1, len(courses) + 1):
            for combo in combinations(courses, r):
                credits = sum(self._course_credits(c) for c in combo)
                if credits > max_credits:
                    continue
                elective_count = sum(c in elective_courses for c in combo)
                if elective_count > elective_slots_left:
                    continue
                feasible.append(combo)
        return feasible

    def _score_plan(self, status: dict, spring: tuple[str, ...], fall: tuple[str, ...]) -> tuple[int, int, int, int]:
        planned = set(spring) | set(fall)
        core_count = len(planned & status["remaining_core"])
        elective_count = min(
            status["elective_remaining"],
            len(planned & status["elective_courses"]),
        )
        requirement_courses = core_count + elective_count
        total_credits = sum(self._course_credits(c) for c in planned)
        # Highest tuple wins: total degree-requirement progress, core progress,
        # elective progress, then credits completed within the supplied ceiling.
        return (requirement_courses, core_count, elective_count, total_credits)

    def generate_plan(self, student_id: str) -> dict:
        status = self.requirement_status(student_id)
        completed = set(status["completed"])
        max_credits = status["max_credits"]

        spring_eligible = self._eligible_courses(
            status,
            "T6_Spring",
            completed_before_term=completed,
            already_planned=set(),
        )
        spring_subsets = self._feasible_subsets(
            spring_eligible,
            max_credits,
            status["elective_courses"],
            status["elective_remaining"],
        )

        best: PlanChoice | None = None
        for spring in spring_subsets:
            spring_set = set(spring)
            completed_before_fall = completed | spring_set
            fall_eligible = self._eligible_courses(
                status,
                "T7_Fall",
                completed_before_term=completed_before_fall,
                already_planned=spring_set,
            )
            elective_slots_left = max(
                0,
                status["elective_remaining"] - len(spring_set & status["elective_courses"]),
            )
            fall_subsets = self._feasible_subsets(
                fall_eligible,
                max_credits,
                status["elective_courses"],
                elective_slots_left,
            )

            for fall in fall_subsets:
                score = self._score_plan(status, spring, fall)
                choice = PlanChoice(spring=spring, fall=fall, score=score)
                if best is None or choice.score > best.score:
                    best = choice
                elif best is not None and choice.score == best.score:
                    # Deterministic tie-break: lexicographically smaller course lists.
                    if (choice.spring, choice.fall) < (best.spring, best.fall):
                        best = choice

        if best is None:
            best = PlanChoice(tuple(), tuple(), (0, 0, 0, 0))

        plan = {
            "Student_ID": student_id,
            "Program_ID": status["program_id"],
            "Maximum_Recommended_Credits": max_credits,
            "Completed_Core_Count": len(status["required_core"] & completed),
            "Remaining_Core_Before_Plan": len(status["remaining_core"]),
            "Completed_Elective_Count": len(status["completed_electives"]),
            "Remaining_Electives_Before_Plan": status["elective_remaining"],
            "T6_Spring": list(best.spring),
            "T7_Fall": list(best.fall),
            "T6_Spring_Credits": sum(self._course_credits(c) for c in best.spring),
            "T7_Fall_Credits": sum(self._course_credits(c) for c in best.fall),
            "Planned_Requirement_Courses": best.score[0],
            "Planned_Core_Courses": best.score[1],
            "Planned_Elective_Courses": best.score[2],
            "Planning_Assumption": (
                "A T6_Spring course is treated as successfully completed before it can satisfy "
                "a prerequisite for T7_Fall; the plan must be revisited if that assumption fails."
            ),
        }

        validation = validate_plan(
            student_id=student_id,
            plan=plan,
            students=self.students,
            history=self.history,
            catalog=self.catalog,
            prerequisites=self.prerequisites,
            requirements=self.requirements,
            offerings=self.offerings,
        )
        plan["Validation_Passed"] = validation["valid"]
        plan["Validation_Issues"] = validation["issues"]
        return plan


def _plan_rows(plan: dict, planner: AcademicPlanner, case_id: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for term_id in PLANNING_TERMS:
        for course_id in plan[term_id]:
            catalog_row = planner._catalog_lookup.loc[course_id]
            prereqs = list(planner._prereq_map.get(course_id, ()))
            rows.append(
                {
                    "Case_ID": case_id,
                    "Student_ID": plan["Student_ID"],
                    "Program_ID": plan["Program_ID"],
                    "Term_ID": term_id,
                    "Course_ID": course_id,
                    "Credits": int(catalog_row["Credits"]),
                    "Requirement_Group": str(catalog_row["Requirement_Group"]),
                    "Required_Flag": int(catalog_row["Required_Flag"]),
                    "Prerequisites": ";".join(prereqs) if prereqs else "None",
                }
            )
    return rows


def generate_advisor_case_plans() -> dict:
    """Generate and validate two-term plans for all required advisor cases."""
    planner = AcademicPlanner()
    cases = load_csv("advisor_cases.csv")

    detailed_plans: list[dict] = []
    plan_rows: list[dict] = []
    summary_rows: list[dict] = []

    for case in cases.itertuples(index=False):
        plan = planner.generate_plan(str(case.Student_ID))
        plan["Case_ID"] = str(case.Case_ID)
        detailed_plans.append(plan)
        plan_rows.extend(_plan_rows(plan, planner, str(case.Case_ID)))
        summary_rows.append(
            {
                "Case_ID": str(case.Case_ID),
                "Student_ID": plan["Student_ID"],
                "Program_ID": plan["Program_ID"],
                "Maximum_Recommended_Credits": plan["Maximum_Recommended_Credits"],
                "T6_Spring_Credits": plan["T6_Spring_Credits"],
                "T7_Fall_Credits": plan["T7_Fall_Credits"],
                "Planned_Requirement_Courses": plan["Planned_Requirement_Courses"],
                "Planned_Core_Courses": plan["Planned_Core_Courses"],
                "Planned_Elective_Courses": plan["Planned_Elective_Courses"],
                "Validation_Passed": plan["Validation_Passed"],
                "Validation_Issue_Count": len(plan["Validation_Issues"]),
            }
        )

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(plan_rows).to_csv(PLANS_DIR / "advisor_case_course_plans.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(PLANS_DIR / "advisor_case_plan_summary.csv", index=False)

    # JSON is convenient for the later end-to-end case-study and demo layer.
    import json

    with open(PLANS_DIR / "advisor_case_plans.json", "w", encoding="utf-8") as f:
        json.dump(detailed_plans, f, indent=2)

    result = {
        "cases": int(len(detailed_plans)),
        "valid_plans": int(sum(bool(p["Validation_Passed"]) for p in detailed_plans)),
        "invalid_plans": int(sum(not bool(p["Validation_Passed"]) for p in detailed_plans)),
        "cases_with_no_recommended_courses": int(
            sum(len(p["T6_Spring"]) + len(p["T7_Fall"]) == 0 for p in detailed_plans)
        ),
        "planning_terms": list(PLANNING_TERMS),
        "method": "Constraint-based two-term search over feasible course subsets",
        "objective": (
            "Maximize courses that satisfy remaining degree requirements; prioritize core courses, "
            "then needed program electives, then degree-progress credits."
        ),
        "planning_assumption": (
            "T6_Spring planned courses are assumed successfully completed before T7_Fall prerequisite checks."
        ),
        "outputs": {
            "course_plans": str(PLANS_DIR / "advisor_case_course_plans.csv"),
            "plan_summary": str(PLANS_DIR / "advisor_case_plan_summary.csv"),
            "plans_json": str(PLANS_DIR / "advisor_case_plans.json"),
        },
    }

    with open(METRICS_DIR / "academic_planner_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

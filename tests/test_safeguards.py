import unittest

import pandas as pd

from src.early_signal_model import MODEL_FEATURES
from src.plan_validator import validate_plan


class PredictiveFeatureSafeguardTests(unittest.TestCase):
    def test_prohibited_columns_are_not_model_features(self):
        prohibited = {"Student_ID", "Audit_Group", "Observation_Term"}
        self.assertTrue(prohibited.isdisjoint(MODEL_FEATURES))


class PlanValidatorSafeguardTests(unittest.TestCase):
    def setUp(self):
        self.students = pd.DataFrame(
            [
                {
                    "Student_ID": "S1",
                    "Program_ID": "P1",
                    "Maximum_Recommended_Credits": 6,
                }
            ]
        )
        self.history = pd.DataFrame(
            [
                {"Student_ID": "S1", "Course_ID": "C0", "Passed": 1},
            ]
        )
        self.catalog = pd.DataFrame(
            [
                {"Course_ID": "C0", "Program_ID": "P1", "Credits": 3},
                {"Course_ID": "C1", "Program_ID": "P1", "Credits": 3},
                {"Course_ID": "C2", "Program_ID": "P1", "Credits": 3},
                {"Course_ID": "C3", "Program_ID": "P1", "Credits": 4},
            ]
        )
        self.prerequisites = pd.DataFrame(
            [
                {
                    "Course_ID": "C2",
                    "Prerequisite_Course_ID": "C1",
                }
            ]
        )
        self.requirements = pd.DataFrame(
            [
                {
                    "Program_ID": "P1",
                    "Requirement_Type": "Required_Course",
                    "Course_ID": "C1",
                    "Minimum_Courses_From_Group": 0,
                },
                {
                    "Program_ID": "P1",
                    "Requirement_Type": "Required_Course",
                    "Course_ID": "C2",
                    "Minimum_Courses_From_Group": 0,
                },
                {
                    "Program_ID": "P1",
                    "Requirement_Type": "Required_Course",
                    "Course_ID": "C3",
                    "Minimum_Courses_From_Group": 0,
                },
            ]
        )
        self.offerings = pd.DataFrame(
            [
                {"Course_ID": "C1", "Term_ID": "T6_Spring", "Available": True},
                {"Course_ID": "C2", "Term_ID": "T6_Spring", "Available": True},
                {"Course_ID": "C2", "Term_ID": "T7_Fall", "Available": True},
                {"Course_ID": "C3", "Term_ID": "T6_Spring", "Available": True},
            ]
        )

    def _validate(self, plan):
        return validate_plan(
            student_id="S1",
            plan=plan,
            students=self.students,
            history=self.history,
            catalog=self.catalog,
            prerequisites=self.prerequisites,
            requirements=self.requirements,
            offerings=self.offerings,
        )

    def test_spring_course_can_unlock_fall_prerequisite(self):
        result = self._validate(
            {"T6_Spring": ["C1"], "T7_Fall": ["C2"]}
        )
        self.assertTrue(result["valid"], result["issues"])

    def test_same_term_prerequisite_does_not_count(self):
        result = self._validate(
            {"T6_Spring": ["C1", "C2"], "T7_Fall": []}
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("missing prerequisites" in issue for issue in result["issues"])
        )

    def test_credit_limit_violation_is_rejected(self):
        result = self._validate(
            {"T6_Spring": ["C1", "C3"], "T7_Fall": []}
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("exceeding the maximum" in issue for issue in result["issues"])
        )

    def test_completed_course_is_rejected(self):
        plan = {"T6_Spring": ["C0"], "T7_Fall": []}
        result = self._validate(plan)
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("already successfully completed" in issue for issue in result["issues"])
        )


if __name__ == "__main__":
    unittest.main()

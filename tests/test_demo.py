import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="student_success_demo_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copy2(PROJECT_ROOT / "demo_app.py", self.root / "demo_app.py")
        shutil.copytree(PROJECT_ROOT / "results", self.root / "results")
        self.app = AppTest.from_file(str(self.root / "demo_app.py"), default_timeout=20)

    def assert_rendered(self):
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(list(self.app.error), [])

    def result_hashes(self):
        return {
            str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.root / "results").rglob("*") if path.is_file()
        }

    def test_sections_and_all_cases_work_without_raw_data_or_writing_results(self):
        before = self.result_hashes()
        self.app.run()
        self.assert_rendered()
        for section in self.app.radio[0].options:
            self.app.radio[0].set_value(section).run()
            self.assert_rendered()
            if section in ("Advisor case explorer", "Decision audit / logging"):
                case_ids = [f"Case_{number:02d}" for number in range(1, 13)]
                self.assertEqual(len(self.app.selectbox[0].options), len(case_ids))
                for case_id in case_ids:
                    self.app.selectbox[0].set_value(case_id).run()
                    self.assert_rendered()
            if section == "Advisor case explorer":
                self.app.button[0].click().run()
                self.assertEqual(self.app.selectbox[0].value, "Case_02")
                links = " ".join(item.value for item in self.app.info)
                self.assertIn("Program_A_C04", links)
                self.assertIn("Program_A_C05", links)
                self.assertIn("if successfully completed", links)
                self.app.button[1].click().run()
                self.assertEqual(self.app.selectbox[0].value, "Case_06")
                self.assertIn("77.6%", [item.value for item in self.app.metric])
        self.assertEqual(before, self.result_hashes())
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "outputs").exists())

    def test_presentation_defaults_and_collapsed_details(self):
        self.app.run()
        self.assertFalse(self.app.expander[0].proto.expanded)
        self.app.radio[0].set_value("Advisor case explorer").run()
        self.assertEqual(self.app.selectbox[0].value, "Case_06")
        values = {item.label: item.value for item in self.app.metric}
        self.assertEqual(values["Risk score"], "77.6%")
        self.assertEqual(values["Early-signal advisor flag"], "YES")
        self.app.button[0].click().run()
        values = {item.label: item.value for item in self.app.metric}
        self.assertEqual(values["Risk score"], "49.5%")
        self.assertEqual(values["Early-signal advisor flag"], "NO")
        self.assertIn("Near-threshold result", self.app.warning[0].value)
        self.app.radio[0].set_value("Decision audit / logging").run()
        self.assertEqual(self.app.selectbox[0].value, "Case_06")
        values = {item.label: item.value for item in self.app.metric}
        self.assertEqual(values["Plan validation"], "PASSED")
        self.assertEqual(values["Human approval"], "REQUIRED")
        self.assertEqual(values["Autonomous student contact"], "NOT ALLOWED")
        self.assertEqual(values["Autonomous course registration"], "NOT ALLOWED")
        for expander in self.app.expander:
            self.assertFalse(expander.proto.expanded, expander.label)
        self.assert_rendered()

    def test_missing_artifact_has_actionable_message(self):
        (self.root / "results" / "presentation_data.json").unlink()
        self.app.run()
        self.assertEqual(list(self.app.exception), [])
        self.assertIn("results/presentation_data.json", self.app.error[0].value)
        self.assertIn("Restore the tracked", self.app.info[-1].value)

    def test_malformed_artifact_has_actionable_message(self):
        path = self.root / "results" / "model" / "failure_analysis.json"
        path.write_text("not json", encoding="utf-8")
        self.app.run()
        self.app.radio[0].set_value("Failure analysis").run()
        self.assertEqual(list(self.app.exception), [])
        self.assertIn("results/model/failure_analysis.json", self.app.error[0].value)

    def test_failed_saved_validation_is_not_presented_as_feasible(self):
        path = self.root / "results" / "planner" / "advisor_case_plans.json"
        plans = json.loads(path.read_text(encoding="utf-8"))
        for plan in plans:
            if plan["Case_ID"] == "Case_02":
                plan["Validation_Passed"] = False
                plan["Validation_Issues"] = ["Example rejected plan"]
        path.write_text(json.dumps(plans), encoding="utf-8")
        self.app.run()
        self.app.radio[0].set_value("Advisor case explorer").run()
        self.assertEqual(list(self.app.exception), [])
        self.app.selectbox[0].set_value("Case_02").run()
        self.assertEqual(list(self.app.success), [])
        self.assertIn("Do not treat this plan as feasible", self.app.error[0].value)


if __name__ == "__main__":
    unittest.main()

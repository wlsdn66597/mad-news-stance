import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_phase1_qualitative.py"
SPEC = importlib.util.spec_from_file_location("qualitative", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QualitativeAnalysisTest(unittest.TestCase):
    def test_parsers_match_phase1_formats(self):
        self.assertEqual(MODULE.parse_answer("mmlu", "Therefore (C)"), "C")
        self.assertEqual(MODULE.parse_answer("gsm8k", "Answer: 1,250"), "1250")

    def test_transition_and_priority(self):
        self.assertEqual(MODULE.transition_name(False, True), "corrected")
        rows = [
            {"item_id": "1", "transition": "stable_correct", "initial_unique_answers": 1, "changed_agents": 0},
            {"item_id": "2", "transition": "corrected", "initial_unique_answers": 2, "changed_agents": 1},
        ]
        self.assertEqual(MODULE.select_cases(rows, 1)[0]["item_id"], "2")


if __name__ == "__main__":
    unittest.main()

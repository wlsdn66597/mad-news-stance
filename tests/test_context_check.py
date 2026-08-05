import unittest

from src.context_check import (
    max_new_tokens_that_fits,
    project_budgets,
    round_requirements,
)


class RoundRequirementsTest(unittest.TestCase):
    def test_round1_carries_own_and_peer_answers(self):
        requirements = round_requirements(
            question_tokens=5000, max_new_tokens=1024, n_agents=3, template_tokens=100
        )
        self.assertEqual(requirements["round0_required"], 6024)
        # 5000 + own 1024 + template 100 + two peers 2048
        self.assertEqual(requirements["round1_input"], 8172)
        self.assertEqual(requirements["round1_required"], 9196)

    def test_more_agents_widen_the_context(self):
        three = round_requirements(1000, 512, n_agents=3)["round1_required"]
        five = round_requirements(1000, 512, n_agents=5)["round1_required"]
        self.assertEqual(five - three, 2 * 512)


class ProjectBudgetsTest(unittest.TestCase):
    def test_reports_fit_and_headroom(self):
        report = project_budgets(
            {"a": 1000, "b": 3000, "c": 2000},
            max_new_tokens=1024,
            context_window=40960,
            n_agents=3,
            template_tokens=100,
        )
        self.assertTrue(report["fits"])
        self.assertEqual(report["over_budget_items"], [])
        self.assertEqual(report["question_tokens"]["max"], 3000)
        # worst item: 3000 + 4*1024 + 100
        self.assertEqual(report["peak_required"]["max"], 7196)
        self.assertEqual(report["headroom_at_worst_item"], 40960 - 7196)

    def test_flags_the_items_that_would_stop_the_run(self):
        report = project_budgets(
            {"small": 500, "huge": 30000},
            max_new_tokens=1024,
            context_window=32768,
            n_agents=3,
        )
        self.assertFalse(report["fits"])
        self.assertEqual([row["item_id"] for row in report["over_budget_items"]], ["huge"])
        self.assertLess(report["headroom_at_worst_item"], 0)

    def test_single_round_run_only_needs_the_question(self):
        report = project_budgets(
            {"a": 30000}, max_new_tokens=1024, context_window=32768, n_agents=3, n_rounds=1
        )
        self.assertTrue(report["fits"])
        self.assertEqual(report["peak_required"]["max"], 31024)

    def test_suggested_budget_actually_fits(self):
        window, question, agents, template = 32768, 30000, 3, 100
        suggested = max_new_tokens_that_fits(question, window, agents, template)
        self.assertLessEqual(
            round_requirements(question, suggested, agents, template)["round1_required"], window
        )
        self.assertGreater(
            round_requirements(question, suggested + 1, agents, template)["round1_required"],
            window,
        )


if __name__ == "__main__":
    unittest.main()

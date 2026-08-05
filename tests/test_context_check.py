import unittest

from src.context_check import (
    max_new_tokens_that_fits,
    peak_requirement,
    project_budgets,
    round_requirements,
)


class RoundRequirementsTest(unittest.TestCase):
    def test_round0_is_just_the_question(self):
        requirements = round_requirements(5000, 1024, n_agents=3, round_index=0)
        self.assertEqual(requirements["input"], 5000)
        self.assertEqual(requirements["required"], 6024)

    def test_round1_carries_own_and_peer_answers(self):
        requirements = round_requirements(
            5000, 1024, n_agents=3, template_tokens=100, round_index=1
        )
        # 5000 + own 1024 + template 100 + two peers 2048
        self.assertEqual(requirements["input"], 8172)
        self.assertEqual(requirements["required"], 9196)

    def test_context_grows_linearly_with_the_round_index(self):
        per_round = [
            round_requirements(5000, 1024, 3, 100, round_index=r)["input"] for r in range(4)
        ]
        deltas = [b - a for a, b in zip(per_round, per_round[1:])]
        self.assertEqual(deltas, [3172, 3172, 3172])

    def test_more_agents_widen_the_context(self):
        three = round_requirements(1000, 512, n_agents=3)["required"]
        five = round_requirements(1000, 512, n_agents=5)["required"]
        self.assertEqual(five - three, 2 * 512)

    def test_peak_is_the_last_round(self):
        self.assertEqual(
            peak_requirement(5000, 1024, 3, 100, n_rounds=4),
            round_requirements(5000, 1024, 3, 100, round_index=3)["required"],
        )
        self.assertEqual(
            peak_requirement(5000, 1024, 3, 100, n_rounds=1),
            round_requirements(5000, 1024, 3, 100, round_index=0)["required"],
        )


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

    def test_four_rounds_need_much_more_than_two(self):
        two = project_budgets({"a": 4625}, 1024, 65536, 3, 100, n_rounds=2)
        four = project_budgets({"a": 4625}, 1024, 65536, 3, 100, n_rounds=4)
        self.assertGreater(four["peak_required"]["max"], two["peak_required"]["max"])
        # three debate rounds instead of one: two extra rounds of 3*1024 + 100
        self.assertEqual(
            four["peak_required"]["max"] - two["peak_required"]["max"], 2 * (3 * 1024 + 100)
        )
        self.assertTrue(four["fits"])

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


class SuggestedBudgetTest(unittest.TestCase):
    def test_suggested_budget_actually_fits(self):
        for n_rounds in (2, 4):
            window, question, agents, template = 32768, 20000, 3, 100
            suggested = max_new_tokens_that_fits(question, window, agents, template, n_rounds)
            self.assertLessEqual(
                peak_requirement(question, suggested, agents, template, n_rounds), window
            )
            self.assertGreater(
                peak_requirement(question, suggested + 1, agents, template, n_rounds), window
            )


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest
from unittest.mock import patch


fake_llm = types.ModuleType("src.llm")
fake_llm.build_messages = lambda user_content, system_prompt=None, history=None: (
    ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    + (list(history) if history else [])
    + [{"role": "user", "content": user_content}]
)
fake_llm.chat = lambda *args, **kwargs: ""
fake_llm.strip_think = lambda text: text
sys.modules.setdefault("src.llm", fake_llm)

from src import role_planner  # noqa: E402


class RolePlannerTest(unittest.TestCase):
    def setUp(self):
        self.item = {
            "id": 9,
            "issue": "정책 도입",
            "headline": "정책 도입 논란",
            "article": "정부와 시민단체의 발언을 함께 인용한 기사.",
        }

    def test_parser_requires_three_known_distinct_roles_and_canonicalizes_order(self):
        raw = (
            '{"selected_roles":["wording","sourcing","issue_alignment"],'
            '"observed_features":["many attributed claims"]}'
        )
        roles, features = role_planner.parse_planner_output(
            raw, role_planner.JOURNALISM_ROLE_LIBRARY
        )
        self.assertEqual(roles, ["issue_alignment", "sourcing", "wording"])
        self.assertEqual(features, ["many attributed claims"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            role_planner.parse_planner_output(
                '{"selected_roles":["sourcing","sourcing","wording"]}',
                role_planner.JOURNALISM_ROLE_LIBRARY,
            )

    def test_invalid_planner_output_falls_back_to_fsw(self):
        with patch.object(role_planner, "chat", return_value="not json"):
            plan = role_planner.select_roles(object(), object(), self.item)
        self.assertTrue(plan["fallback"])
        self.assertEqual(plan["selected_roles"], list(role_planner.FALLBACK_ROLES))

    def test_sw_plus_one_keeps_sourcing_and_wording(self):
        raw = (
            '{"selected_roles":["causal_consequence"],'
            '"observed_features":["the article emphasizes consequences"]}'
        )
        with patch.object(role_planner, "chat", return_value=raw):
            plan = role_planner.select_roles(
                object(), object(), self.item, planner_mode="sw_plus_one"
            )
        self.assertFalse(plan["fallback"])
        self.assertEqual(plan["planner_selected_roles"], ["causal_consequence"])
        self.assertEqual(plan["fixed_roles"], ["sourcing", "wording"])
        self.assertEqual(
            plan["selected_roles"],
            ["causal_consequence", "sourcing", "wording"],
        )

    def test_sw_plus_one_fallback_is_exactly_fsw(self):
        with patch.object(role_planner, "chat", return_value="not json"):
            plan = role_planner.select_roles(
                object(), object(), self.item, planner_mode="sw_plus_one"
            )
        self.assertEqual(
            plan["selected_roles"], ["foregrounding", "sourcing", "wording"]
        )

    def test_four_round_debate_uses_one_planner_plus_twelve_agent_calls(self):
        plan = {
            "selected_roles": ["issue_alignment", "sourcing", "wording"],
            "observed_features": [], "raw": "{}", "prompt": "p", "messages": [],
            "fallback": False, "parse_error": None,
            "stance_label_leakage": False, "role_pool": "journalism",
        }
        debate_result = {
            "pred": "neutral", "raw": [], "preds": ["neutral"] * 3,
            "prompt": "q", "debate_trace": {},
        }
        with patch.object(role_planner, "select_roles", return_value=plan):
            with patch.object(role_planner.methods, "run_debate", return_value=debate_result) as debate:
                result = role_planner.run_planner_debate(
                    object(), object(), object(), self.item, n_rounds=4
                )
        self.assertEqual(result["call_counts"]["total_generation_calls"], 13)
        self.assertEqual(debate.call_args.kwargs["n_rounds"], 4)
        self.assertEqual(debate.call_args.kwargs["n_agents"], 3)
        self.assertEqual(len(debate.call_args.args[4]), 3)


if __name__ == "__main__":
    unittest.main()

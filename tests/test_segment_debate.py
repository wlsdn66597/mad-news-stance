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

from src import segment_debate  # noqa: E402


class FakeTask:
    @staticmethod
    def parse(text):
        lowered = text.lower()
        for label in ("supportive", "oppositional", "neutral"):
            if label in lowered:
                return label
        return None


class SegmentDebateTest(unittest.TestCase):
    def setUp(self):
        self.item = {
            "id": 7,
            "issue": "정책 확대",
            "headline": "정부, 정책 확대 발표",
            "article": (
                "정부는 정책 확대가 필요하다고 밝혔다.\n\n"
                "관계자는 “효과가 클 것”이라고 말했다.\n\n"
                "야당은 반대했지만 정부는 계획을 유지했다."
            ),
        }

    def test_split_uses_four_distinct_journalism_segments(self):
        segments = segment_debate.split_journalism_segments(self.item)
        self.assertEqual(segments["headline"], self.item["headline"])
        self.assertTrue(segments["lead"].startswith("정부는"))
        self.assertTrue(segments["conclusion"].startswith("야당은"))
        self.assertIn("효과가 클 것", segments["quotations"])
        self.assertIn("Attribution context", segments["quotations"])

    def test_no_quotation_is_explicit_not_silently_neutral(self):
        item = dict(self.item, article="인용문이 없는 기사다.")
        segment = segment_debate.split_journalism_segments(item)["quotations"]
        self.assertIn("No direct quotation", segment)

    def test_four_round_majority_uses_16_calls_and_no_aggregator(self):
        fake_trace = {
            "answers_by_round": [
                [
                    "Final stance: supportive",
                    "Final stance: supportive",
                    "Final stance: neutral",
                    "Final stance: oppositional",
                ]
                for _ in range(4)
            ]
        }
        with patch.object(segment_debate, "run_debate_engine", return_value=fake_trace):
            with patch.object(segment_debate, "chat") as aggregator_chat:
                result = segment_debate.run_segment_debate(
                    object(), object(), FakeTask(), self.item, n_rounds=4
                )
        self.assertEqual(result["pred"], "supportive")
        self.assertEqual(result["decision"]["label_counts"]["supportive"], 2)
        self.assertFalse(result["decision"]["tied"])
        self.assertIsNone(result["aggregator"])
        self.assertEqual(result["call_counts"]["total_generation_calls"], 16)
        aggregator_chat.assert_not_called()

    def test_majority_tie_matches_existing_first_valid_rule(self):
        fake_trace = {
            "answers_by_round": [[
                "Final stance: neutral",
                "Final stance: supportive",
                "Final stance: neutral",
                "Final stance: neutral",
            ]]
        }
        with patch.object(segment_debate, "run_debate_engine", return_value=fake_trace):
            result = segment_debate.run_segment_debate(
                object(), object(), FakeTask(), self.item, n_rounds=1
            )
        self.assertEqual(result["pred"], "neutral")
        self.assertFalse(result["decision"]["tied"])

        fake_trace["answers_by_round"][0][-1] = "Final stance: supportive"
        with patch.object(segment_debate, "run_debate_engine", return_value=fake_trace):
            result = segment_debate.run_segment_debate(
                object(), object(), FakeTask(), self.item, n_rounds=1
            )
        self.assertEqual(result["pred"], "neutral")
        self.assertTrue(result["decision"]["tied"])
        self.assertEqual(result["decision"]["reason"], "final_tie_legacy_first_valid")

    def test_aggregator_mode_reproduces_17_call_pipeline(self):
        fake_trace = {
            "answers_by_round": [
                ["Final stance: neutral"] * 4 for _ in range(4)
            ]
        }
        with patch.object(segment_debate, "run_debate_engine", return_value=fake_trace) as engine:
            with patch.object(segment_debate, "chat", return_value="Final stance: supportive"):
                result = segment_debate.run_segment_debate(
                    object(), object(), FakeTask(), self.item, n_rounds=4,
                    decision_rule="aggregator",
                )
        self.assertEqual(result["pred"], "supportive")
        self.assertEqual(result["call_counts"]["total_generation_calls"], 17)
        kwargs = engine.call_args.kwargs
        self.assertEqual(kwargs["n_agents"], 4)
        self.assertEqual(kwargs["n_rounds"], 4)
        self.assertEqual(len(kwargs["agent_questions"]), 4)


if __name__ == "__main__":
    unittest.main()

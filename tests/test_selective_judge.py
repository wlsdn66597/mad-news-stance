import json
import unittest
from unittest.mock import patch

from src.selective_judge import (
    JUDGE_SYSTEM_PROMPT,
    judge_user_payload,
    ordered_candidate_analyses,
    parse_judge_json,
    run_judge,
)


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class SelectiveJudgeTest(unittest.TestCase):
    def setUp(self):
        self.item = {
            "id": "42",
            "issue": "policy reform",
            "headline": "A headline",
            "article": "The original article text.",
        }
        self.rounds = [
            ["agent zero", "agent one", "agent two"],
            ["agent zero final", "agent one final", "agent two final"],
        ]

    def test_candidate_order_is_seeded_and_anonymized(self):
        first, order1 = ordered_candidate_analyses(self.rounds, "42", 7)
        second, order2 = ordered_candidate_analyses(self.rounds, "42", 7)
        self.assertEqual(first, second)
        self.assertEqual(order1, order2)
        self.assertEqual({row["candidate_id"] for row in first}, {"A", "B", "C"})
        self.assertTrue(all("source_agent_index" not in row for row in first))
        mapping = {}
        for row in order1:
            mapping.setdefault(row["source_agent_index"], set()).add(row["candidate_id"])
        self.assertTrue(all(len(candidate_ids) == 1 for candidate_ids in mapping.values()))

    def test_payload_never_contains_gold_or_vote_counts(self):
        candidates, _ = ordered_candidate_analyses(self.rounds, "42", 7)
        payload = judge_user_payload({**self.item, "gold": "neutral"}, candidates)
        encoded = json.dumps(payload)
        self.assertNotIn("gold", encoded)
        self.assertNotIn("confidence", encoded)
        self.assertNotIn("vote_count", encoded)

    def test_strict_json_parser(self):
        parsed = parse_judge_json(
            '{"label":"neutral","evidence_sufficient":true,'
            '"evidence":["article wording"],"rationale":"brief"}'
        )
        self.assertEqual(parsed["label"], "neutral")
        with self.assertRaises(ValueError):
            parse_judge_json('{"label":"neutral"}')

    def test_one_repair_retry_then_success(self):
        outputs = [
            "not json",
            '{"label":"supportive","evidence_sufficient":true,'
            '"evidence":["framing"],"rationale":"brief"}',
        ]
        with patch("src.selective_judge.chat", side_effect=outputs):
            result = run_judge(
                object(), FakeTokenizer(), self.item, self.rounds,
                max_retries=1, temperature=0.0,
            )
        self.assertEqual(result["prediction"], "supportive")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(len(result["attempts"]), 2)

    def test_article_only_omits_candidates(self):
        output = (
            '{"label":"neutral","evidence_sufficient":false,'
            '"evidence":[],"rationale":"insufficient"}'
        )

        def fake_messages(user_content, system_prompt=None, history=None):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        with patch("src.selective_judge.chat", return_value=output), patch(
            "src.selective_judge.build_messages", side_effect=fake_messages
        ) as messages:
            result = run_judge(
                object(), FakeTokenizer(), self.item, self.rounds,
                input_mode="article_only",
            )
        payload = json.loads(messages.call_args.args[0])
        self.assertEqual(payload["candidate_analyses"], [])
        self.assertEqual(result["candidate_order"], [])
        self.assertEqual(messages.call_args.kwargs["system_prompt"], JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

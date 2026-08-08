import json
import unittest
from unittest.mock import patch

from src.selective_advocacy import (
    ABLATIONS,
    independent_case,
    selective_judge_system_prompt,
    judge_payload,
    order_candidates,
    proposed_labels,
    run_selective_judge,
    selective_trigger,
)


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


ITEM = {
    "id": "42",
    "issue": "김포 서울 편입",
    "headline": "편입 논의 본격화",
    "article": "기사 본문입니다.",
    "gold": "neutral",
}


class TriggerTest(unittest.TestCase):
    STABLE = [["supportive", "supportive", "neutral"],
              ["supportive", "supportive", "neutral"]]
    TIED = [["supportive", "oppositional", "neutral"],
            ["supportive", "oppositional", "neutral"]]
    UNANIMOUS = [["supportive"] * 3, ["supportive"] * 3]

    def test_a_two_to_one_split_fires_on_the_default(self):
        triggered, reason, missing = selective_trigger(self.STABLE)
        self.assertTrue(triggered)
        self.assertEqual(missing, ["oppositional"])
        self.assertIn("split_vote", reason)

    def test_instability_alone_ignores_a_stable_split(self):
        triggered, reason, _ = selective_trigger(self.STABLE, "instability")
        self.assertFalse(triggered)
        self.assertEqual(reason, "stable")

    def test_a_tie_fires_even_with_every_label_proposed(self):
        triggered, reason, missing = selective_trigger(self.TIED, "instability")
        self.assertTrue(triggered)
        self.assertEqual(missing, [])
        self.assertEqual(reason, "final_tie")

    def test_a_unanimous_vote_is_left_alone(self):
        """With three agents and three labels a unanimous vote always leaves two
        labels unargued. Measured on the EXAONE test run that is 830 of 1001
        items, so triggering on a missing label overrides the vote almost
        everywhere -- the opposite of what this design is for."""
        triggered, reason, missing = selective_trigger(self.UNANIMOUS)
        self.assertFalse(triggered)
        self.assertEqual(reason, "unanimous_and_stable")
        self.assertEqual(sorted(missing), ["neutral", "oppositional"])
        # the ablation that shows why the missing-label trigger is wrong
        self.assertTrue(selective_trigger(self.UNANIMOUS, "missing_label")[0])

    def test_non_unanimous_covers_both_splits_and_ties(self):
        self.assertTrue(selective_trigger(self.STABLE, "non_unanimous")[0])
        self.assertTrue(selective_trigger(self.TIED, "non_unanimous")[0])
        self.assertFalse(selective_trigger(self.UNANIMOUS, "non_unanimous")[0])

    def test_a_unanimous_final_that_overturned_round0_still_fires(self):
        rounds = [["neutral", "neutral", "supportive"], ["supportive"] * 3]
        self.assertFalse(selective_trigger(rounds, "non_unanimous")[0])
        triggered, reason, _ = selective_trigger(rounds, "unstable_or_split")
        self.assertTrue(triggered)
        self.assertIn("round0_final_disagree", reason)

    def test_scope_any_round_counts_a_label_proposed_earlier(self):
        rounds = [["oppositional", "supportive", "neutral"], ["supportive"] * 3]
        self.assertEqual(selective_trigger(rounds, "missing_label", "last")[2],
                         ["oppositional", "neutral"])
        self.assertEqual(selective_trigger(rounds, "missing_label", "any_round")[2], [])

    def test_unknown_trigger_is_rejected(self):
        with self.assertRaises(ValueError):
            selective_trigger(self.STABLE, "sometimes")

    def test_votes_are_the_signal_being_preserved(self):
        self.assertEqual(dict(proposed_labels(self.STABLE)),
                         {"supportive": 2, "neutral": 1})


class CandidateTest(unittest.TestCase):
    RAW = ["long supportive reasoning here", "short supportive", "neutral reasoning"]
    PARSED = ["supportive", "supportive", "neutral"]

    def test_the_longest_answer_represents_a_label_and_the_count_is_kept(self):
        found = independent_case(self.RAW, self.PARSED, "supportive")
        self.assertEqual(found["independent_supporters"], 2)
        self.assertEqual(found["source_agent_index"], 0)
        self.assertEqual(found["origin"], "independent")

    def test_a_label_nobody_proposed_has_no_independent_case(self):
        self.assertIsNone(independent_case(self.RAW, self.PARSED, "oppositional"))

    def test_order_is_seeded_and_hides_the_agent_identity(self):
        cases = [
            {"stance": "supportive", "origin": "independent",
             "independent_supporters": 2, "source_agent_index": 0, "analysis": "a"},
            {"stance": "neutral", "origin": "independent",
             "independent_supporters": 1, "source_agent_index": 2, "analysis": "b"},
            {"stance": "oppositional", "origin": "commissioned",
             "independent_supporters": 0, "source_agent_index": None, "analysis": "c"},
        ]
        candidates, order = order_candidates(cases, "42", 7)
        self.assertEqual(order, order_candidates(cases, "42", 7)[1])
        self.assertEqual({c["candidate_id"] for c in candidates}, {"A", "B", "C"})
        self.assertNotIn("source_agent_index", candidates[0])

    def test_the_payload_carries_the_vote_and_never_the_gold(self):
        cases = [
            {"stance": "supportive", "origin": "independent",
             "independent_supporters": 2, "analysis": "a"},
            {"stance": "neutral", "origin": "commissioned",
             "independent_supporters": 0, "analysis": "b"},
        ]
        candidates, _ = order_candidates(cases, "42", 7)
        payload = judge_payload(ITEM, candidates, {"supportive": 2, "neutral": 1})
        self.assertEqual(payload["independent_votes"],
                         {"supportive": 2, "oppositional": 0, "neutral": 1})
        self.assertNotIn("gold", json.dumps(payload, ensure_ascii=False))
        self.assertIn(ITEM["article"], json.dumps(payload, ensure_ascii=False))


class SelectiveJudgeTest(unittest.TestCase):
    CASES = [
        {"stance": "supportive", "origin": "independent",
         "independent_supporters": 2, "analysis": "a"},
        {"stance": "oppositional", "origin": "commissioned",
         "independent_supporters": 0, "analysis": "b"},
        {"stance": "neutral", "origin": "independent",
         "independent_supporters": 1, "analysis": "c"},
    ]
    VOTES = {"supportive": 2, "oppositional": 0, "neutral": 1}

    def test_valid_json_is_taken_as_is(self):
        output = ('{"label":"neutral","evidence_sufficient":true,'
                  '"evidence":["both sides"],"rationale":"balanced"}')
        with patch("src.selective_advocacy.chat", return_value=output):
            result = run_selective_judge(
                object(), FakeTokenizer(), ITEM, self.CASES, self.VOTES
            )
        self.assertEqual(result["prediction"], "neutral")
        self.assertEqual(result["retry_count"], 0)
        self.assertFalse(result["label_recovered_from_text"])

    def test_the_retry_resends_the_article_and_the_rationales(self):
        outputs = ["not json at all",
                   '{"label":"supportive","evidence_sufficient":true,'
                   '"evidence":["framing"],"rationale":"clear"}']
        with patch("src.selective_advocacy.chat", side_effect=outputs) as chat:
            result = run_selective_judge(
                object(), FakeTokenizer(), ITEM, self.CASES, self.VOTES
            )
        repair = chat.call_args_list[1].args[2][-1]["content"]
        self.assertIn(ITEM["article"], repair)
        self.assertIn("independent_votes", repair)
        self.assertEqual(result["prediction"], "supportive")

    def test_a_label_is_salvaged_only_after_the_retries(self):
        truncated = '{"label":"oppositional","evidence_sufficient":true,"evidence":["fr'
        with patch("src.selective_advocacy.chat", return_value=truncated) as chat:
            result = run_selective_judge(
                object(), FakeTokenizer(), ITEM, self.CASES, self.VOTES
            )
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(result["label_recovered_from_text"])
        self.assertNotIn("recovered", result["attempts"][0]["parse_error"])

    def test_the_judge_is_told_which_rationales_are_commissioned(self):
        output = "nothing parseable"
        with patch("src.selective_advocacy.chat", return_value=output) as chat:
            run_selective_judge(object(), FakeTokenizer(), ITEM, self.CASES, self.VOTES)
        system_prompt = chat.call_args_list[0].args[2][0]["content"]
        self.assertIn("commissioned", system_prompt)
        self.assertIn("independent", system_prompt)
        user_text = chat.call_args_list[0].args[2][-1]["content"]
        self.assertIn('"origin": "commissioned"', user_text)


class AblationTest(unittest.TestCase):
    """Each ablation removes one thing, and the prompt has to remove it too --
    otherwise the run measures the instruction rather than the input."""

    CASES = [
        {"stance": "supportive", "origin": "independent",
         "independent_supporters": 2, "analysis": "a"},
        {"stance": "oppositional", "origin": "commissioned",
         "independent_supporters": 0, "analysis": "b"},
    ]
    VOTES = {"supportive": 2, "oppositional": 0, "neutral": 0}

    def payload(self, ablation):
        candidates, _ = order_candidates(self.CASES, "42", 7)
        return judge_payload(ITEM, candidates, self.VOTES, ablation)

    def test_full_keeps_everything(self):
        payload = self.payload("full")
        self.assertEqual(len(payload["rationales"]), 2)
        self.assertIn("independent_votes", payload)
        self.assertIn("commissioned", selective_judge_system_prompt("full"))

    def test_no_commissioned_drops_the_counterfactual_and_its_paragraph(self):
        payload = self.payload("no_commissioned")
        self.assertEqual([r["stance"] for r in payload["rationales"]], ["supportive"])
        self.assertIn("independent_votes", payload)
        self.assertNotIn("commissioned", selective_judge_system_prompt("no_commissioned"))

    def test_no_votes_keeps_the_rationales_but_hides_the_count_and_origin(self):
        payload = self.payload("no_votes")
        self.assertEqual(len(payload["rationales"]), 2)
        self.assertNotIn("independent_votes", payload)
        for rationale in payload["rationales"]:
            self.assertNotIn("origin", rationale)
            self.assertNotIn("independent_supporters", rationale)
        prompt = selective_judge_system_prompt("no_votes")
        self.assertNotIn("independent majority", prompt)

    def test_article_only_is_the_stronger_model_control(self):
        payload = self.payload("article_only")
        self.assertNotIn("rationales", payload)
        self.assertNotIn("independent_votes", payload)
        self.assertIn(ITEM["article"], json.dumps(payload, ensure_ascii=False))
        prompt = selective_judge_system_prompt("article_only")
        self.assertIn("article alone", prompt)
        self.assertNotIn("rationale", prompt)

    def test_no_ablation_leaks_the_gold(self):
        for ablation in ABLATIONS:
            encoded = json.dumps(self.payload(ablation), ensure_ascii=False)
            self.assertNotIn("gold", encoded, ablation)

    def test_unknown_ablation_is_rejected(self):
        with self.assertRaises(ValueError):
            selective_judge_system_prompt("none")
        with self.assertRaises(ValueError):
            self.payload("none")

    def test_the_judge_uses_the_matching_prompt(self):
        with patch("src.selective_advocacy.chat", return_value="?") as chat:
            result = run_selective_judge(
                object(), FakeTokenizer(), ITEM, self.CASES, self.VOTES,
                ablation="article_only",
            )
        self.assertEqual(result["ablation"], "article_only")
        self.assertIn("article alone", chat.call_args_list[0].args[2][0]["content"])


if __name__ == "__main__":
    unittest.main()

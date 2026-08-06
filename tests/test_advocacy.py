import json
import unittest
from unittest.mock import patch

from src.advocacy import (
    assigned_stances,
    compliance,
    judge_candidates,
    judge_user_payload,
    parse_support_level,
    peer_order,
    run_advocacy,
    run_advocacy_judge,
    support_fallback,
)
from src.prompts.advocacy import STANCE_LABELS, advocate_question


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


ITEM = {
    "id": "42",
    "issue": "김포 서울 편입",
    "headline": "편입 논의 본격화",
    "article": "기사 본문입니다. 시민 반응이 엇갈린다.",
    "gold": "neutral",
}


def case(stance, support="moderate", analysis=None):
    return {
        "agent_index": STANCE_LABELS.index(stance),
        "stance": stance,
        "analysis": analysis or f"evidence for {stance}\nSupport for the assigned stance: {support}",
        "support": support,
        "stated_label": None,
    }


class AdvocatePromptTest(unittest.TestCase):
    def test_prompt_carries_the_assigned_stance_and_no_gold(self):
        for stance in STANCE_LABELS:
            prompt = advocate_question(ITEM, stance)
            self.assertIn(f"Assigned stance: {stance}", prompt)
            self.assertIn(ITEM["article"], prompt)
            self.assertNotIn("gold", prompt.lower())
            # the other two labels must not be suggested as the answer
            for other in set(STANCE_LABELS) - {stance}:
                self.assertNotIn(f"Assigned stance: {other}", prompt)

    def test_unknown_stance_is_rejected(self):
        with self.assertRaises(ValueError):
            advocate_question(ITEM, "positive")

    def test_every_label_is_argued_exactly_once(self):
        stances = assigned_stances("42", 7)
        self.assertEqual(sorted(stances), sorted(STANCE_LABELS))

    def test_assignment_is_seeded_and_varies_across_items(self):
        self.assertEqual(assigned_stances("42", 7), assigned_stances("42", 7))
        orders = {tuple(assigned_stances(str(i), 7)) for i in range(40)}
        self.assertGreater(len(orders), 1)


class SupportLevelTest(unittest.TestCase):
    def test_parses_the_last_declared_level(self):
        text = "Support for the assigned stance: weak\n...\nSupport for the assigned stance: strong"
        self.assertEqual(parse_support_level(text), "strong")

    def test_tolerates_angle_brackets_and_case(self):
        self.assertEqual(
            parse_support_level("SUPPORT FOR THE ASSIGNED STANCE: <Moderate>"), "moderate"
        )

    def test_missing_level_is_none(self):
        self.assertIsNone(parse_support_level("no such field here"))


class AdvocacyRoundTest(unittest.TestCase):
    def test_one_call_per_label_with_isolated_contexts(self):
        outputs = [f"case {i}\nSupport for the assigned stance: moderate" for i in range(3)]
        with patch("src.advocacy.chat", side_effect=outputs) as chat:
            result = run_advocacy(object(), FakeTokenizer(), ITEM, order_seed=5)
        self.assertEqual(result["calls"], 3)
        self.assertEqual(
            sorted(c["stance"] for c in result["cases"]), sorted(STANCE_LABELS)
        )
        # each advocate starts from a fresh two-message context: no peer leakage
        for call in chat.call_args_list:
            messages = call.args[2]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")

    def test_rebuttal_round_shows_peers_but_not_self(self):
        outputs = [f"case {i}\nSupport for the assigned stance: weak" for i in range(3)]
        rebuttals = [f"rebuttal {i}\nSupport for the assigned stance: strong" for i in range(3)]
        with patch("src.advocacy.chat", side_effect=outputs + rebuttals) as chat:
            result = run_advocacy(
                object(), FakeTokenizer(), ITEM, order_seed=5, rebuttal=True
            )
        self.assertEqual(result["calls"], 6)
        self.assertEqual([c["support"] for c in result["cases"]], ["strong"] * 3)
        self.assertEqual([c["initial_support"] for c in result["cases"]], ["weak"] * 3)
        first_rebuttal = chat.call_args_list[3].args[2][-1]["content"]
        self.assertIn("case 1", first_rebuttal)
        self.assertIn("case 2", first_rebuttal)
        self.assertNotIn("case 0", first_rebuttal)

    def test_peer_order_is_seeded_and_excludes_self(self):
        cases = [case(s) for s in STANCE_LABELS]
        first, order = peer_order(cases, 1, "42", 11)
        second, order2 = peer_order(cases, 1, "42", 11)
        self.assertEqual(first, second)
        self.assertEqual(order, order2)
        self.assertEqual({row["source_agent_index"] for row in order}, {0, 2})


class JudgeInputTest(unittest.TestCase):
    def test_payload_labels_each_analysis_with_the_stance_it_argues(self):
        cases = [case(s) for s in STANCE_LABELS]
        candidates, order = judge_candidates(cases, "42", 3)
        payload = judge_user_payload(ITEM, candidates)
        self.assertEqual(
            sorted(a["stance_argued"] for a in payload["analyses"]), sorted(STANCE_LABELS)
        )
        self.assertEqual({a["candidate_id"] for a in payload["analyses"]}, {"A", "B", "C"})
        self.assertEqual(len(order), 3)

    def test_payload_hides_gold_and_agent_identity(self):
        cases = [case(s) for s in STANCE_LABELS]
        candidates, _ = judge_candidates(cases, "42", 3)
        encoded = json.dumps(judge_user_payload({**ITEM}, candidates), ensure_ascii=False)
        self.assertNotIn("gold", encoded)
        self.assertNotIn("agent_index", encoded)
        self.assertIn(ITEM["article"], encoded)

    def test_candidate_order_is_seeded(self):
        cases = [case(s) for s in STANCE_LABELS]
        self.assertEqual(
            judge_candidates(cases, "42", 3)[1], judge_candidates(cases, "42", 3)[1]
        )

    def test_judge_returns_the_parsed_label(self):
        output = ('{"label":"oppositional","evidence_sufficient":true,'
                  '"evidence":["framing"],"rationale":"brief"}')
        with patch("src.advocacy.chat", return_value=output):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, [case(s) for s in STANCE_LABELS]
            )
        self.assertEqual(result["prediction"], "oppositional")
        self.assertEqual(result["retry_count"], 0)

    def test_one_repair_retry_then_success(self):
        outputs = [
            "not json",
            '{"label":"neutral","evidence_sufficient":true,'
            '"evidence":["x"],"rationale":"y"}',
        ]
        with patch("src.advocacy.chat", side_effect=outputs):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, [case(s) for s in STANCE_LABELS]
            )
        self.assertEqual(result["prediction"], "neutral")
        self.assertEqual(result["retry_count"], 1)

    def test_judge_failure_yields_no_prediction(self):
        with patch("src.advocacy.chat", side_effect=["bad", "still bad"]):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, [case(s) for s in STANCE_LABELS]
            )
        self.assertIsNone(result["prediction"])
        self.assertEqual(len(result["attempts"]), 2)


class FallbackTest(unittest.TestCase):
    def test_strongest_declared_support_wins(self):
        cases = [case("supportive", "weak"), case("oppositional", "strong"),
                 case("neutral", "moderate")]
        label, reason = support_fallback(cases, "42")
        self.assertEqual(label, "oppositional")
        self.assertEqual(reason, "strongest_declared_support")

    def test_tie_is_deterministic(self):
        cases = [case("supportive", "strong"), case("oppositional", "strong"),
                 case("neutral", "weak")]
        first = support_fallback(cases, "42")
        self.assertEqual(first, support_fallback(cases, "42"))
        self.assertIn(first[0], {"supportive", "oppositional"})
        self.assertEqual(first[1], "tied_declared_support_deterministic_fallback")

    def test_no_declared_support_still_returns_a_label(self):
        cases = [dict(case(s), support=None) for s in STANCE_LABELS]
        label, reason = support_fallback(cases, "42")
        self.assertIn(label, STANCE_LABELS)
        self.assertEqual(reason, "no_declared_support_deterministic_fallback")


class ComplianceTest(unittest.TestCase):
    def test_counts_advocates_that_argued_another_side(self):
        cases = [
            dict(case("supportive"), stated_label="supportive"),
            dict(case("oppositional"), stated_label="neutral"),
            dict(case("neutral"), stated_label=None),
        ]
        report = compliance(cases)
        self.assertEqual(report["agents"], 3)
        self.assertEqual(report["stated_label_present"], 2)
        self.assertEqual(report["stated_label_mismatch"], 1)
        self.assertEqual(report["mismatched_stances"], ["oppositional"])


if __name__ == "__main__":
    unittest.main()

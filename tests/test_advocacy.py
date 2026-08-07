import json
import unittest
from unittest.mock import patch

from src.advocacy import (
    assigned_stances,
    parse_final_verdict,
    stated_label,
    compliance,
    judge_candidates,
    judge_failure_fallback,
    judge_user_payload,
    peer_order,
    round_label_trajectory,
    run_advocacy,
    run_advocacy_judge,
)
from src.prompts.advocacy import PROMPT_STYLES, STANCE_LABELS, advocate_question


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


class RecordingChat:
    """chat() receives the agent's live context list and we append to it right
    after, so assertions need a snapshot taken at call time."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, model, tokenizer, messages, **kwargs):
        self.calls.append([dict(message) for message in messages])
        return self.replies.pop(0)


def case(stance, analysis=None, declared=None, source=None):
    return {
        "agent_index": STANCE_LABELS.index(stance),
        "stance": stance,
        "analysis": analysis or f"evidence for {stance}",
        "stated_label": declared,
        "stated_label_source": source,
    }


class AdvocatePromptTest(unittest.TestCase):
    def test_every_style_carries_the_assigned_stance_and_no_gold(self):
        for style in PROMPT_STYLES:
            for stance in STANCE_LABELS:
                prompt = advocate_question(ITEM, stance, style=style)
                self.assertIn(stance, prompt, style)
                self.assertIn(ITEM["article"], prompt)
                self.assertNotIn("gold", prompt.lower())
                for other in set(STANCE_LABELS) - {stance}:
                    self.assertNotIn(other, prompt, f"{style}/{stance} leaked {other}")

    def test_toc_judge_asks_for_discussion_before_the_verdict(self):
        """A closing instruction that reads as "output only this line" made a
        1.2B judge skip the discussion entirely: 841 of 1001 answers were just
        the label."""
        judge = PROMPT_STYLES["toc"]["judge_system"]
        self.assertIn("strengths and weaknesses", judge)
        self.assertIn("final sentence should include", judge)
        self.assertNotIn("should be exactly", judge)

    def test_toc_style_stays_close_to_the_published_prompt_length(self):
        toc, structured = PROMPT_STYLES["toc"], PROMPT_STYLES["structured"]
        # the published Chain-of-Explanation system prompt is 34 words
        self.assertLessEqual(len(toc["advocate_system"].split()), 40)
        # and the contrastive-verification judge prompt is 82
        self.assertLessEqual(len(toc["judge_system"].split()), 90)
        self.assertGreater(
            len(structured["advocate_system"].split()), len(toc["advocate_system"].split())
        )

    def test_no_style_asks_an_advocate_to_rate_its_own_confidence(self):
        for style in PROMPT_STYLES.values():
            for key in ("advocate_system", "advocate_user", "rebuttal"):
                text = style[key].lower()
                self.assertNotIn("support for the assigned stance", text)
                self.assertNotIn("confidence", text)
                self.assertNotIn("weak|moderate|strong", text)

    def test_unknown_style_is_rejected(self):
        with self.assertRaises(ValueError):
            advocate_question(ITEM, "neutral", style="verbose")

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


class AdvocacyRoundTest(unittest.TestCase):
    def test_single_round_is_one_call_per_label_with_isolated_contexts(self):
        chat = RecordingChat([f"case {i}" for i in range(3)])
        with patch("src.advocacy.chat", chat):
            result = run_advocacy(object(), FakeTokenizer(), ITEM, rounds=1, order_seed=5)
        self.assertEqual(result["calls"], 3)
        self.assertEqual(sorted(c["stance"] for c in result["cases"]), sorted(STANCE_LABELS))
        # each advocate starts from a fresh two-message context: no peer leakage
        for messages in chat.calls:
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertNotIn("case ", messages[1]["content"])

    def test_base_configuration_is_two_rounds(self):
        outputs = [f"round0 {i}" for i in range(3)] + [f"round1 {i}" for i in range(3)]
        with patch("src.advocacy.chat", side_effect=outputs):
            result = run_advocacy(object(), FakeTokenizer(), ITEM, order_seed=5)
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["calls"], 6)
        self.assertEqual(len(result["analyses_by_round"]), 2)
        # the judge sees the last round
        self.assertEqual([c["analysis"] for c in result["cases"]], result["analyses_by_round"][1])

    def test_round_count_scales_the_calls_and_history(self):
        outputs = [f"r{r} a{a}" for r in range(4) for a in range(3)]
        with patch("src.advocacy.chat", side_effect=outputs):
            result = run_advocacy(object(), FakeTokenizer(), ITEM, rounds=4, order_seed=5)
        self.assertEqual(result["calls"], 12)
        self.assertEqual(len(result["analyses_by_round"]), 4)
        self.assertEqual(len(result["peer_orders"]), 3 * 3)  # no peers in round 0

    def test_rounds_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            run_advocacy(object(), FakeTokenizer(), ITEM, rounds=0)

    def test_later_rounds_show_peers_but_not_self(self):
        chat = RecordingChat([f"case {i}" for i in range(3)]
                             + [f"rebuttal {i}" for i in range(3)])
        with patch("src.advocacy.chat", chat):
            run_advocacy(object(), FakeTokenizer(), ITEM, rounds=2, order_seed=5)
        first_rebuttal = chat.calls[3][-1]["content"]
        self.assertIn("case 1", first_rebuttal)
        self.assertIn("case 2", first_rebuttal)
        self.assertNotIn("case 0", first_rebuttal)

    def test_each_agent_keeps_its_own_growing_context(self):
        chat = RecordingChat([f"case {i}" for i in range(3)]
                             + [f"rebuttal {i}" for i in range(3)])
        with patch("src.advocacy.chat", chat):
            run_advocacy(object(), FakeTokenizer(), ITEM, rounds=2, order_seed=5)
        second_round = chat.calls[3]
        self.assertEqual(len(second_round), 4)  # system, question, own answer, rebuttal
        self.assertEqual(second_round[2]["content"], "case 0")
        self.assertEqual(second_round[2]["role"], "assistant")

    def test_peer_order_is_seeded_and_excludes_self(self):
        analyses = ["a0", "a1", "a2"]
        stances = list(STANCE_LABELS)
        first, order = peer_order(analyses, stances, 1, "42", 11, 1)
        second, order2 = peer_order(analyses, stances, 1, "42", 11, 1)
        self.assertEqual(first, second)
        self.assertEqual(order, order2)
        self.assertNotIn("a1", [text for _, text in first])
        self.assertEqual({row["source_agent_index"] for row in order}, {0, 2})


class LabelTrajectoryTest(unittest.TestCase):
    def test_reports_whether_each_agent_held_its_assigned_side(self):
        stances = ["supportive", "oppositional", "neutral"]
        analyses_by_round = [
            ["Final stance: supportive", "Final stance: oppositional", "no label"],
            ["Final stance: supportive", "Final stance: neutral",
             "the article reads as supportive to some"],
        ]
        trajectory = round_label_trajectory(analyses_by_round, stances)
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory[0]["held_assigned"], [True, True, True])
        # the oppositional advocate declared neutral in round 1: a real concession
        self.assertEqual(trajectory[1]["held_assigned"], [True, False, True])
        self.assertEqual(trajectory[1]["stated_labels"][1], "neutral")
        # the third only mentions another label in passing, so it still holds
        self.assertEqual(trajectory[1]["stated_label_sources"][2], "mention")


class JudgeInputTest(unittest.TestCase):
    def test_structured_payload_labels_each_analysis_with_the_stance_it_argues(self):
        cases = [case(s) for s in STANCE_LABELS]
        candidates, order = judge_candidates(cases, "42", 3)
        payload = judge_user_payload(ITEM, candidates, "structured")
        self.assertEqual(
            sorted(a["stance_argued"] for a in payload["analyses"]), sorted(STANCE_LABELS)
        )
        self.assertEqual({a["candidate_id"] for a in payload["analyses"]}, {"A", "B", "C"})
        self.assertEqual(len(order), 3)

    def test_toc_payload_is_plain_text_without_a_json_schema(self):
        """ToC's judge is given the article and one rationale per stance value as
        text and answers in prose, so a JSON output schema does not belong."""
        cases = [case(s) for s in STANCE_LABELS]
        candidates, _ = judge_candidates(cases, "42", 3)
        payload = judge_user_payload(ITEM, candidates, "toc")
        self.assertIsInstance(payload, str)
        self.assertNotIn("output_schema", payload)
        self.assertNotIn("candidate_id", payload)
        self.assertIn(ITEM["article"], payload)
        for stance in STANCE_LABELS:
            self.assertIn(f"Stance: {stance} Rationale:", payload)

    def test_neither_payload_leaks_gold_or_agent_identity(self):
        cases = [case(s) for s in STANCE_LABELS]
        candidates, _ = judge_candidates(cases, "42", 3)
        for style in ("toc", "structured"):
            payload = judge_user_payload({**ITEM}, candidates, style)
            encoded = payload if isinstance(payload, str) else json.dumps(
                payload, ensure_ascii=False
            )
            self.assertNotIn("gold", encoded, style)
            self.assertNotIn("agent_index", encoded, style)
            self.assertIn(ITEM["article"], encoded, style)

    def test_candidate_order_is_seeded(self):
        cases = [case(s) for s in STANCE_LABELS]
        self.assertEqual(
            judge_candidates(cases, "42", 3)[1], judge_candidates(cases, "42", 3)[1]
        )


class FinalVerdictTest(unittest.TestCase):
    """ToC puts the stance value in the final sentence, after the discussion."""

    def test_reads_the_closing_sentence_not_the_whole_text(self):
        text = ("Analyst A argues supportive but overstates the framing. "
                "Analyst B argues neutral, which ignores the headline.\n"
                "The article is oppositional.")
        self.assertEqual(parse_final_verdict(text), "oppositional")

    def test_an_explicit_marker_still_wins(self):
        self.assertEqual(
            parse_final_verdict("weighing it up\nFinal stance: neutral"), "neutral"
        )

    def test_no_stance_value_anywhere(self):
        self.assertIsNone(parse_final_verdict("I cannot decide."))
        self.assertIsNone(parse_final_verdict(""))


class JudgeOutputFormatTest(unittest.TestCase):
    """ToC's judge writes prose and names the label last; ours returns JSON."""

    def setUp(self):
        self.cases = [case(s) for s in STANCE_LABELS]

    def test_toc_judge_reads_the_final_line(self):
        prose = ("Analyst A overstates the framing, Analyst B is better grounded.\n"
                 "Final stance: oppositional")
        with patch("src.advocacy.chat", return_value=prose):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, self.cases, prompt_style="toc"
            )
        self.assertEqual(result["prediction"], "oppositional")
        self.assertEqual(result["output_format"], "final_line")
        self.assertEqual(result["retry_count"], 0)
        self.assertIsNone(result["attempts"][0]["parse_error"])

    def test_toc_judge_retries_when_no_label_is_stated(self):
        outputs = ["I cannot tell.",
                   "On balance the framing is critical.\nFinal stance: oppositional"]
        with patch("src.advocacy.chat", side_effect=outputs) as chat:
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, self.cases, prompt_style="toc"
            )
        self.assertEqual(result["prediction"], "oppositional")
        self.assertEqual(result["retry_count"], 1)
        repair = chat.call_args_list[1].args[2][-1]["content"]
        self.assertIn("strengths and weaknesses", repair)

    def test_structured_judge_parses_the_json_schema(self):
        output = ('{"label":"oppositional","evidence_sufficient":true,'
                  '"evidence":["framing"],"rationale":"brief"}')
        with patch("src.advocacy.chat", return_value=output):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, self.cases, prompt_style="structured"
            )
        self.assertEqual(result["prediction"], "oppositional")
        self.assertEqual(result["output_format"], "json")
        self.assertFalse(result["label_recovered_from_text"])

    def test_structured_judge_recovers_a_label_from_broken_json(self):
        truncated = '{"label":"supportive","evidence_sufficient":true,"evidence":["fram'
        with patch("src.advocacy.chat", return_value=truncated):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, self.cases, prompt_style="structured"
            )
        self.assertEqual(result["prediction"], "supportive")
        self.assertTrue(result["label_recovered_from_text"])
        self.assertIn("recovered", result["attempts"][0]["parse_error"])

    def test_no_label_anywhere_still_fails(self):
        with patch("src.advocacy.chat", side_effect=["???", "???"]):
            result = run_advocacy_judge(
                object(), FakeTokenizer(), ITEM, self.cases, prompt_style="toc"
            )
        self.assertIsNone(result["prediction"])

    def test_judge_system_prompt_follows_the_style(self):
        output = "Final stance: neutral"
        seen = {}
        for style, marker in (("toc", "expert linguistic assistant"),
                              ("structured", "advocacy arguments, not evidence")):
            with patch("src.advocacy.chat", return_value=output) as chat:
                run_advocacy_judge(
                    object(), FakeTokenizer(), ITEM, self.cases, prompt_style=style,
                )
            seen[style] = chat.call_args_list[0].args[2][0]["content"]
            self.assertIn(marker, seen[style])
        self.assertNotEqual(seen["toc"], seen["structured"])


class FallbackTest(unittest.TestCase):
    """Not a consensus rule: the advocates disagree by design and the judge
    always decides. This only covers a judge that produced no label at all."""

    def test_returns_a_label_deterministically(self):
        first = judge_failure_fallback("42", 7)
        self.assertEqual(first, judge_failure_fallback("42", 7))
        self.assertIn(first[0], STANCE_LABELS)
        self.assertEqual(first[1], "judge_no_label_deterministic")

    def test_different_items_do_not_all_get_the_same_label(self):
        labels = {judge_failure_fallback(str(i), 7)[0] for i in range(40)}
        self.assertGreater(len(labels), 1)


class StatedLabelTest(unittest.TestCase):
    def test_an_explicit_line_is_a_declaration(self):
        self.assertEqual(
            stated_label("weighing it up\nFinal stance: oppositional"),
            ("oppositional", "marker"),
        )

    def test_discussing_counter_evidence_is_only_a_mention(self):
        """The reason the first smoke reported 26/60 defections: an advocate
        that argues supportive and then names counter-evidence ends on another
        label word, which the fallback parser picks up."""
        text = ("The framing is positive, so a supportive reading fits. "
                "Counter-evidence: the close merely reports both sides, which "
                "some would call neutral.")
        label, source = stated_label(text)
        self.assertEqual(label, "neutral")
        self.assertEqual(source, "mention")

    def test_no_label_at_all(self):
        self.assertEqual(stated_label("nothing relevant here"), (None, None))


class ComplianceTest(unittest.TestCase):
    def test_only_explicit_declarations_count_as_defection(self):
        cases = [
            case("supportive", declared="supportive", source="marker"),
            case("oppositional", declared="neutral", source="marker"),
            case("neutral", declared="supportive", source="mention"),
        ]
        report = compliance(cases)
        self.assertEqual(report["agents"], 3)
        self.assertEqual(report["declared_label"], 2)
        self.assertEqual(report["declared_defection"], 1)
        self.assertEqual(report["defected_stances"], ["oppositional"])
        # the mention-only disagreement is reported but not called a defection
        self.assertEqual(report["last_mention_only"], 1)
        self.assertEqual(report["last_mention_differs"], 1)


if __name__ == "__main__":
    unittest.main()

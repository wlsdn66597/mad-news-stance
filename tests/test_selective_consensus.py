import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.consensus import judge_trigger
from src.consensus_round import (
    RECONSIDERATION_INSTRUCTION,
    consensus_decision,
    consensus_prompt,
    ordered_peer_answers,
    run_consensus_round,
)
from src.selective_consensus import (
    build_summary,
    judge_payload_fingerprint,
    load_judge_cache,
    output_prefix,
    process_record,
    process_records,
)
from src.selective_judge import judge_user_payload

DEBATE_TEMPLATE = (
    "Use the other agents' responses as additional information and reconsider "
    "your previous judgment.\n"
    "Final stance: <supportive|oppositional|neutral>"
    "\n\nThe other agents' judgments are as follows:\n\n{others}"
)


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def make_record(item_id, round0, final, gold="neutral", with_contexts=True):
    """A saved debate item in the Phase 2 ``debate_trace`` schema."""
    question = f"Issue: reform\nHeadline: H{item_id}\nArticle:\nBody of article {item_id}."
    answers = [
        [f"reasoning {index}\nFinal stance: {label}" for index, label in enumerate(round0)],
        [f"revised {index}\nFinal stance: {label}" for index, label in enumerate(final)],
    ]
    trace = {
        "question": question,
        "debate_template": DEBATE_TEMPLATE,
        "system_prompt": None,
        "other_answers_format": "labeled_agents",
        "prompt_profile": "stance_minimal_en",
        "answers_by_round": answers,
    }
    if with_contexts:
        trace["agent_contexts"] = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answers[0][index]},
                {"role": "user", "content": "debate round prompt"},
                {"role": "assistant", "content": answers[1][index]},
            ]
            for index in range(len(round0))
        ]
    return {
        "id": item_id,
        "gold": gold,
        "issue": "reform",
        "headline": f"H{item_id}",
        "article": f"Body of article {item_id}.",
        "prompt": question,
        "debate_trace": trace,
        "round_predictions": [list(round0), list(final)],
        "raw_rounds": answers,
    }


STABLE = make_record("stable", ["neutral"] * 3, ["neutral"] * 3)
UNSTABLE = make_record(
    "unstable", ["supportive"] * 2 + ["neutral"], ["neutral"] * 2 + ["supportive"]
)
TIED = make_record("tied", ["neutral"] * 3, ["neutral", "supportive", "oppositional"])


def consensus_result(labels, mode="plain_extra_round", outputs=None):
    return {
        "consensus_prompt_mode": mode,
        "previous_agent_labels": ["neutral", "neutral", "supportive"],
        "revised_agent_labels": list(labels),
        "revised_agent_outputs": outputs or [f"Final stance: {label}" for label in labels],
        "peer_orders": [],
        "calls": len(labels),
        "latency_seconds": 0.0,
        "token_usage": {"input_tokens": 1, "output_tokens": 1},
        **consensus_decision(labels),
    }


class ConsensusPromptTest(unittest.TestCase):
    def test_plain_mode_reuses_the_saved_debate_prompt_unchanged(self):
        peers = "[Agent 1]\nA\n\n[Agent 2]\nB"
        plain = consensus_prompt(DEBATE_TEMPLATE, peers, "plain_extra_round")
        self.assertEqual(plain, DEBATE_TEMPLATE.format(others=peers, question=""))

    def test_reconsideration_mode_only_appends_the_instruction(self):
        peers = "[Agent 1]\nA\n\n[Agent 2]\nB"
        plain = consensus_prompt(DEBATE_TEMPLATE, peers, "plain_extra_round")
        reconsider = consensus_prompt(DEBATE_TEMPLATE, peers, "independent_reconsideration")
        self.assertEqual(reconsider, f"{plain}\n\n{RECONSIDERATION_INSTRUCTION}")
        self.assertNotIn("agree", reconsider.lower())
        self.assertNotIn("persuade", reconsider.lower())

    def test_prompt_never_reveals_gold_or_vote_counts(self):
        peers = "[Agent 1]\nFinal stance: neutral\n\n[Agent 2]\nFinal stance: neutral"
        for mode in ("plain_extra_round", "independent_reconsideration"):
            prompt = consensus_prompt(DEBATE_TEMPLATE, peers, mode)
            self.assertNotIn("gold", prompt.lower())
            self.assertNotIn("majority label", prompt.lower())
            self.assertNotIn("vote", prompt.lower())

    def test_peer_order_is_seeded_reproducible_and_excludes_self(self):
        answers = ["a0", "a1", "a2"]
        first, order = ordered_peer_answers(answers, 1, "item-3", 11)
        second, order2 = ordered_peer_answers(answers, 1, "item-3", 11)
        self.assertEqual(first, second)
        self.assertEqual(order, order2)
        self.assertNotIn("a1", first)
        self.assertEqual({row["source_agent_index"] for row in order}, {0, 2})


class ConsensusDecisionTest(unittest.TestCase):
    def test_unanimous(self):
        decision = consensus_decision(["neutral"] * 3)
        self.assertTrue(decision["consensus_reached"])
        self.assertEqual(decision["consensus_label"], "neutral")
        self.assertEqual(decision["consensus_reason"], "unanimous")

    def test_two_one_and_one_one_one_are_not_consensus(self):
        two_one = consensus_decision(["neutral", "neutral", "supportive"])
        one_one_one = consensus_decision(["neutral", "supportive", "oppositional"])
        self.assertFalse(two_one["consensus_reached"])
        self.assertEqual(two_one["consensus_reason"], "split_2_1")
        self.assertFalse(one_one_one["consensus_reached"])
        self.assertEqual(one_one_one["consensus_reason"], "split_1_1_1")

    def test_parse_failure_is_not_consensus(self):
        decision = consensus_decision(["neutral", "neutral", None])
        self.assertFalse(decision["consensus_reached"])
        self.assertTrue(decision["parse_failure"])
        self.assertEqual(decision["consensus_reason"], "parse_failure")


class ConsensusRoundTest(unittest.TestCase):
    def test_saved_agent_context_is_extended_with_one_user_turn(self):
        outputs = [f"ok\nFinal stance: {label}" for label in ("neutral", "neutral", "neutral")]
        with patch("src.consensus_round.chat", side_effect=outputs) as chat:
            result = run_consensus_round(
                object(), FakeTokenizer(), UNSTABLE, "plain_extra_round", order_seed=5,
            )
        self.assertEqual(result["revised_agent_labels"], ["neutral"] * 3)
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["agent_context_source"], "saved_agent_contexts")
        messages = chat.call_args_list[0].args[2]
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[:4], UNSTABLE["debate_trace"]["agent_contexts"][0])

    def test_context_is_reconstructed_when_agent_contexts_are_missing(self):
        record = make_record(
            "no-ctx", ["supportive"] * 2 + ["neutral"], ["neutral"] * 3, with_contexts=False
        )
        outputs = ["Final stance: neutral"] * 3
        with patch("src.consensus_round.chat", side_effect=outputs) as chat:
            result = run_consensus_round(object(), FakeTokenizer(), record, "plain_extra_round")
        self.assertEqual(
            result["agent_context_source"], "reconstructed_from_question_and_own_answer"
        )
        messages = chat.call_args_list[2].args[2]
        self.assertEqual(messages[0]["content"], record["debate_trace"]["question"])
        self.assertEqual(messages[1]["content"], record["raw_rounds"][-1][2])

    def test_agents_see_only_anonymized_peer_answers(self):
        outputs = ["Final stance: neutral"] * 3
        with patch("src.consensus_round.chat", side_effect=outputs) as chat:
            run_consensus_round(object(), FakeTokenizer(), UNSTABLE, "independent_reconsideration")
        prompt = chat.call_args_list[0].args[2][-1]["content"]
        self.assertIn(RECONSIDERATION_INSTRUCTION, prompt)
        self.assertIn(UNSTABLE["raw_rounds"][-1][1], prompt)
        self.assertNotIn(UNSTABLE["raw_rounds"][-1][0], prompt)
        self.assertNotIn("gold", prompt.lower())


class RoutingTest(unittest.TestCase):
    def test_trigger_matches_the_existing_selective_judge(self):
        for record in (STABLE, UNSTABLE, TIED):
            expected, reason = judge_trigger(record["round_predictions"], "instability")
            row = process_record(
                record,
                consensus_mode="none",
                judge_fn=lambda r: {"prediction": "neutral", "attempts": [{}]},
            )
            self.assertEqual(row["triggered"], expected, record["id"])
            self.assertEqual(row["trigger_reason"], reason, record["id"])

    def test_stable_items_keep_the_existing_majority_without_any_call(self):
        row = process_record(STABLE, consensus_mode="plain_extra_round")
        self.assertFalse(row["triggered"])
        self.assertEqual(row["final_source"], "baseline")
        self.assertEqual(row["final_prediction"], "neutral")

    def test_unanimous_consensus_skips_the_judge(self):
        calls = []
        row = process_record(
            UNSTABLE,
            consensus_mode="plain_extra_round",
            consensus_fn=lambda record: consensus_result(["oppositional"] * 3),
            judge_fn=lambda record: calls.append(record) or {"prediction": "neutral"},
        )
        self.assertTrue(row["consensus_reached"])
        self.assertFalse(row["judge_required"])
        self.assertEqual(row["judge_source"], "none")
        self.assertEqual(row["final_source"], "consensus")
        self.assertEqual(row["final_prediction"], "oppositional")
        self.assertEqual(calls, [])

    def test_split_and_parse_failure_go_to_the_judge(self):
        for labels, reason in (
            (["neutral", "neutral", "supportive"], "split_2_1"),
            (["neutral", "supportive", "oppositional"], "split_1_1_1"),
            (["neutral", "neutral", None], "parse_failure"),
        ):
            calls = []
            row = process_record(
                UNSTABLE,
                consensus_mode="plain_extra_round",
                consensus_fn=lambda record, labels=labels: consensus_result(labels),
                judge_fn=lambda record: calls.append(record["id"]) or {
                    "prediction": "supportive", "attempts": [{}],
                },
            )
            self.assertEqual(row["consensus_reason"], reason)
            self.assertTrue(row["judge_required"])
            self.assertEqual(row["judge_source"], "model_call")
            self.assertEqual(row["final_source"], "judge")
            self.assertEqual(row["final_prediction"], "supportive")
            self.assertEqual(calls, ["unstable"])
        self.assertTrue(row["consensus_parse_failure"])

    def test_judge_failure_falls_back_to_the_existing_rule(self):
        row = process_record(
            UNSTABLE,
            consensus_mode="none",
            judge_fn=lambda record: {"prediction": None, "attempts": [{}, {}]},
        )
        self.assertTrue(row["fallback_used"])
        self.assertEqual(row["final_source"], "fallback")
        self.assertEqual(row["final_prediction"], "supportive")  # round0 majority
        self.assertEqual(row["fallback_reason"], "round0_unique_majority")

    def test_direct_condition_sends_every_triggered_item_to_the_judge(self):
        rows = [
            process_record(
                record,
                consensus_mode="none",
                judge_fn=lambda record: {"prediction": "neutral", "attempts": [{}]},
            )
            for record in (STABLE, UNSTABLE, TIED)
        ]
        self.assertEqual([row["judge_required"] for row in rows], [False, True, True])
        self.assertEqual([row["revised_agent_labels"] for row in rows], [[], [], []])


def fail_judge():
    raise AssertionError("judge must not be called")


class JudgeInputTest(unittest.TestCase):
    def test_article_only_payload_excludes_consensus_and_debate_text(self):
        payload = judge_user_payload(UNSTABLE, [])
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["candidate_analyses"], [])
        self.assertNotIn("revised", encoded)
        self.assertNotIn("Final stance", encoded)
        self.assertNotIn("gold", encoded)
        self.assertIn(UNSTABLE["article"], encoded)


class JudgeCacheTest(unittest.TestCase):
    expected = {
        "judge_model": "Qwen/Qwen3-8B",
        "judge_input_mode": "article_only",
        "judge_temperature": 0.0,
        "judge_max_new_tokens": 384,
        "judge_max_retries": 1,
    }

    def write_cache(self, directory, rows, config_overrides=None):
        items_path = Path(directory) / "run.items.json"
        items_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        config = {"input_result": "phase2.json", **self.expected}
        config.update(config_overrides or {})
        (Path(directory) / "run.config.json").write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        return str(items_path)

    def cached_row(self, **overrides):
        row = {
            "item_id": "unstable",
            "triggered": True,
            "judge_model": "Qwen/Qwen3-8B",
            "judge_input_mode": "article_only",
            "judge_decoding": {"temperature": 0.0, "max_new_tokens": 384, "max_retries": 1},
            "judge_prediction": "oppositional",
            "final_prediction": "oppositional",
        }
        row.update(overrides)
        return row

    def test_matching_cache_is_reused_without_a_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(directory, [self.cached_row()])
            cache, validation = load_judge_cache(path, self.expected, input_result="phase2.json")
            self.assertEqual(validation["usable_rows"], 1)
            row = process_record(
                UNSTABLE, consensus_mode="none", judge_cache=cache,
                judge_fn=lambda record: fail_judge(),
            )
        self.assertEqual(row["judge_source"], "cache")
        self.assertEqual(row["final_prediction"], "oppositional")
        self.assertEqual(row["final_source"], "judge")

    def test_incompatible_rows_are_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(
                directory, [self.cached_row(judge_input_mode="debate_trace")]
            )
            cache, validation = load_judge_cache(path, self.expected, input_result="phase2.json")
        self.assertEqual(cache, {})
        self.assertEqual(validation["mismatched_rows"], 1)

    def test_incompatible_run_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(
                directory, [self.cached_row()], config_overrides={"judge_temperature": 0.7}
            )
            with self.assertRaises(ValueError):
                load_judge_cache(path, self.expected, input_result="phase2.json")

    def test_different_source_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(directory, [self.cached_row()])
            with self.assertRaises(ValueError):
                load_judge_cache(path, self.expected, input_result="other_phase2.json")

    def test_payload_fingerprint_mismatch_falls_back_to_a_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(
                directory, [self.cached_row(judge_payload_sha256="0" * 64)]
            )
            cache, _ = load_judge_cache(path, self.expected, input_result="phase2.json")
            row = process_record(
                UNSTABLE, consensus_mode="none", judge_cache=cache,
                judge_fn=lambda record: {"prediction": "neutral", "attempts": [{}]},
            )
        self.assertEqual(row["judge_cache_rejected"], "payload_mismatch")
        self.assertEqual(row["judge_source"], "model_call")

    def test_matching_payload_fingerprint_is_accepted(self):
        fingerprint = judge_payload_fingerprint(UNSTABLE)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_cache(
                directory, [self.cached_row(judge_payload_sha256=fingerprint)]
            )
            cache, _ = load_judge_cache(path, self.expected, input_result="phase2.json")
            row = process_record(
                UNSTABLE, consensus_mode="none", judge_cache=cache,
                judge_fn=lambda record: fail_judge(),
            )
        self.assertEqual(row["judge_source"], "cache")

    def test_missing_companion_config_requires_an_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            items_path = Path(directory) / "run.items.json"
            items_path.write_text(json.dumps([self.cached_row()]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_judge_cache(str(items_path), self.expected)
            cache, validation = load_judge_cache(
                str(items_path), self.expected, require_config=False
            )
        self.assertEqual(len(cache), 1)
        self.assertFalse(validation["config_found"])


class ResumeTest(unittest.TestCase):
    def test_saved_rows_are_reused_and_only_missing_items_are_processed(self):
        records = [STABLE, UNSTABLE, TIED]
        calls = []

        def judge(record):
            calls.append(record["id"])
            return {"prediction": "neutral", "attempts": [{}]}

        with tempfile.TemporaryDirectory() as directory:
            items_path = Path(directory) / "run.items.json"

            def save(path, rows):
                path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

            first = process_records(
                records[:2], items_path=items_path, save=save,
                consensus_mode="none", judge_fn=judge,
            )
            self.assertEqual(calls, ["unstable"])
            second = process_records(
                records, items_path=items_path, save=save,
                consensus_mode="none", judge_fn=judge,
            )
        self.assertEqual(calls, ["unstable", "tied"])
        self.assertEqual(len(second), 3)
        self.assertEqual(second[:2], first)


class OutputPrefixTest(unittest.TestCase):
    # The real repeat-run stems from results/phase2.
    STEMS = [
        "stance_qwen_test_n1001_stance_minimal_en_d0_s6000_a3_r2_qwen8_test_full_n1001_s6000",
        "stance_qwen_test_n1001_stance_minimal_en_d0_s6001_a3_r2_qwen8_debate_repeat_n1001_s6001",
    ]

    def prefix(self, stem, mode="independent_reconsideration", **overrides):
        kwargs = {
            "input_result": f"results/phase2/{stem}.json",
            "consensus_mode": mode,
            "trigger": "instability",
            "consensus_model": "Qwen/Qwen3-8B",
            "judge_model": "Qwen/Qwen3-8B",
            "consensus_temperature": 1.0,
            "consensus_seed": 6001,
            "consensus_order_seed": 9001,
            "consensus_max_new_tokens": 1024,
            "judge_input_mode": "article_only",
            "judge_temperature": 0.0,
            "judge_order_seed": 7001,
            "aggregation_method": "current_final_majority",
        }
        kwargs.update(overrides)
        return output_prefix(**kwargs)

    def test_written_filenames_fit_the_filesystem_limit(self):
        for stem in self.STEMS:
            for mode in ("none", "plain_extra_round", "independent_reconsideration"):
                name = self.prefix(stem, mode)
                longest = len(f"{name}.summary.json".encode("utf-8"))
                self.assertLess(longest, 255, f"{mode}: {longest} bytes")

    def test_conditions_and_seeds_get_distinct_prefixes(self):
        names = {
            self.prefix(self.STEMS[0], mode)
            for mode in ("none", "plain_extra_round", "independent_reconsideration")
        }
        names.add(self.prefix(self.STEMS[0], consensus_seed=6002))
        names.add(self.prefix(self.STEMS[1]))
        self.assertEqual(len(names), 5)

    def test_overlong_names_are_hashed_not_truncated_into_collisions(self):
        long_stem = "x" * 400
        first = self.prefix(long_stem)
        second = self.prefix(long_stem, consensus_seed=6002)
        self.assertLessEqual(len(first), 200)
        self.assertNotEqual(first, second)


class SummaryTest(unittest.TestCase):
    def test_summary_counts_consensus_and_judge_stages(self):
        rows = [
            process_record(STABLE, consensus_mode="plain_extra_round"),
            process_record(
                UNSTABLE,
                consensus_mode="plain_extra_round",
                consensus_fn=lambda record: consensus_result(["neutral"] * 3),
            ),
            process_record(
                TIED,
                consensus_mode="plain_extra_round",
                consensus_fn=lambda record: consensus_result(
                    ["neutral", "neutral", "supportive"]
                ),
                judge_fn=lambda record: {
                    "prediction": "supportive",
                    "attempts": [{}],
                    "latency_seconds": 1.5,
                    "token_usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ),
        ]
        summary = build_summary(
            rows,
            {"consensus_prompt_mode": "plain_extra_round"},
            direct_predictions={"stable": "neutral", "unstable": "supportive", "tied": "neutral"},
        )
        stage = summary["consensus_stage"]
        self.assertEqual(stage["triggered_total"], 2)
        self.assertEqual(stage["consensus_attempted"], 2)
        self.assertEqual(stage["consensus_reached"], 1)
        self.assertEqual(stage["consensus_correct"], 1)
        self.assertEqual(stage["wrong_unanimous_consensus"], 0)
        self.assertEqual(stage["judge_required"], 1)
        self.assertEqual(stage["final_source_counts"]["consensus"], 1)
        self.assertEqual(summary["cost"]["judge"]["calls"], 1)
        self.assertEqual(summary["items"], 3)
        self.assertEqual(summary["direct_article_judge_comparison"]["covered_items"], 3)
        self.assertTrue(summary["direct_article_judge_comparison"]["complete_coverage"])


if __name__ == "__main__":
    unittest.main()

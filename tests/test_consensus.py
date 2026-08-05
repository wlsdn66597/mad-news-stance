import copy
import json
import os
import unittest
from pathlib import Path

from src.consensus import (
    aggregate,
    cross_round_pooled_vote,
    current_final_majority,
    free_mad_exact,
    free_mad_scores,
    judge_trigger,
    round0_fallback,
)
from src.consensus_io import transition_records


class ConsensusRulesTest(unittest.TestCase):
    def test_current_majority_preserves_legacy_first_agent_tie(self):
        rounds = [
            ["supportive", "oppositional", "neutral"],
            ["neutral", "supportive", "oppositional"],
        ]
        label, reason = current_final_majority(rounds, "tie", 9)
        self.assertEqual(label, "neutral")
        self.assertEqual(reason, "final_tie_legacy_first_valid")

    def test_round0_fallback_uses_only_unanimous_final(self):
        rounds = [
            ["supportive", "supportive", "neutral"],
            ["oppositional", "oppositional", "supportive"],
        ]
        self.assertEqual(round0_fallback(rounds, 1)[0], "supportive")
        unanimous = [rounds[0], ["oppositional"] * 3]
        self.assertEqual(round0_fallback(unanimous, 1)[0], "oppositional")

    def test_round0_tie_fallback_is_reproducible(self):
        rounds = [
            ["supportive", "oppositional", "neutral"],
            ["supportive", "supportive", "neutral"],
        ]
        first = round0_fallback(rounds, "item-7", 31)
        second = round0_fallback(rounds, "item-7", 31)
        self.assertEqual(first, second)
        self.assertEqual(first[1], "round0_tie_deterministic_fallback")

    def test_cross_round_tie_prefers_round0_majority(self):
        rounds = [
            ["supportive", "supportive", "neutral"],
            ["oppositional", "oppositional", "neutral"],
        ]
        self.assertEqual(cross_round_pooled_vote(rounds, 1)[0], "supportive")

    def test_free_mad_exact_algorithm_one_scores(self):
        rounds = [
            ["supportive", "supportive", "oppositional"],
            ["supportive", "oppositional", "oppositional"],
        ]
        scores = free_mad_scores(rounds)
        self.assertAlmostEqual(scores["supportive"], 37.5)
        self.assertAlmostEqual(scores["oppositional"], 45.0)
        self.assertEqual(free_mad_exact(rounds, 1)[0], "oppositional")

    def test_invalid_predictions_do_not_mutate_source(self):
        rounds = [[None, "garbage", "neutral"], [None, "supportive", "neutral"]]
        original = copy.deepcopy(rounds)
        aggregate("free_mad_exact", rounds, 2)
        self.assertEqual(rounds, original)

    def test_instability_trigger(self):
        stable = [["neutral"] * 3, ["neutral"] * 3]
        changed = [["neutral"] * 2 + ["supportive"], ["supportive"] * 2 + ["neutral"]]
        tied = [["neutral"] * 3, ["neutral", "supportive", "oppositional"]]
        self.assertFalse(judge_trigger(stable, "instability")[0])
        self.assertTrue(judge_trigger(changed, "instability")[0])
        self.assertTrue(judge_trigger(tied, "instability")[0])


class TransitionExportIntegrationTest(unittest.TestCase):
    def test_transition_export_has_expected_104_instability_items(self):
        path = os.environ.get("STANCE_TRANSITION_FILE")
        if not path or not Path(path).exists():
            self.skipTest("set STANCE_TRANSITION_FILE to run the 104-item integration test")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = transition_records(data)
        self.assertEqual(len(rows), 104)
        self.assertEqual(data["summary"]["changed_total"], 104)
        self.assertTrue(all(judge_trigger(row["round_predictions"], "instability")[0] for row in rows))


if __name__ == "__main__":
    unittest.main()

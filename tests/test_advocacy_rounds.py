import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_advocacy_rounds.py"
SPEC = importlib.util.spec_from_file_location("analyze_advocacy_rounds", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def case(stance, declared=None, body="The article reports on the plan."):
    """One advocate's text, optionally ending on an explicit declaration."""
    text = f"{body} The assigned reading is {stance}."
    if declared:
        text += f"\nFinal stance: {declared}"
    return text


def row(item_id, gold, stances, rounds, **extra):
    record = {
        "item_id": item_id,
        "gold": gold,
        "assigned_stances": list(stances),
        "analyses_by_round": [list(texts) for texts in rounds],
    }
    record.update(extra)
    return record


class HeldFlagsTest(unittest.TestCase):
    def test_only_an_explicit_declaration_counts_as_conceding(self):
        stances = ["supportive", "neutral", "oppositional"]
        analyses = [
            case("supportive", declared="neutral"),      # conceded
            case("neutral", declared="neutral"),         # held, declared
            case("oppositional"),                        # held, no marker at all
        ]
        flags = MODULE.held_flags(analyses, stances)
        self.assertEqual([held for held, _ in flags], [False, True, True])
        self.assertEqual([label for _, label in flags],
                         ["neutral", "neutral", "oppositional"])

    def test_a_label_merely_named_in_the_prose_is_not_a_concession(self):
        # an advocate that discusses counter-evidence names other labels all the
        # time; counting those would wildly overstate the rate
        analyses = [case("supportive", body="A neutral reading is possible but weak.")]
        flags = MODULE.held_flags(analyses, ["supportive"])
        self.assertEqual(flags, [(True, "supportive")])


class BallotTest(unittest.TestCase):
    def setUp(self):
        stances = ["supportive", "neutral", "oppositional"]
        self.rows = [
            row(
                "1", "supportive", stances,
                [
                    [case(s, declared=s) for s in stances],
                    # the gold advocate concedes during the rebuttal
                    [case("supportive", declared="neutral"),
                     case("neutral", declared="neutral"),
                     case("oppositional", declared="oppositional")],
                ],
            ),
            row(
                "2", "neutral", stances,
                [
                    [case(s, declared=s) for s in stances],
                    [case(s, declared=s) for s in stances],
                ],
            ),
        ]

    def test_gold_leaves_the_ballot_when_its_advocate_concedes(self):
        first = MODULE.round_report(self.rows, 0, {}, 20)
        last = MODULE.round_report(self.rows, 1, {}, 20)
        self.assertEqual(first["ballot_has_gold"], 2)
        self.assertEqual(last["ballot_has_gold"], 1)
        self.assertEqual(first["gold_advocate_held"], 2)
        self.assertEqual(last["gold_advocate_held"], 1)
        # the other two advocates held throughout, which is what makes the drop
        # directional rather than a general fall in compliance
        self.assertEqual(last["other_advocate_held"], last["other_advocate_n"])

    def test_the_declaration_count_says_whether_the_test_could_run(self):
        # a hold rate of 1.0 means nothing if no case carries a marker, so the
        # denominator has to be reported next to it
        stances = ["supportive", "neutral", "oppositional"]
        silent = [row("1", "neutral", stances, [[case(s) for s in stances]])]
        report = MODULE.round_report(silent, 0, {}, 20)
        self.assertEqual(report["declared_label"], 0)
        self.assertEqual(report["gold_advocate_held"], report["gold_advocate_n"])
        self.assertEqual(report["last_mention_only"], 3)

        loud = MODULE.round_report(self.rows, 0, {}, 20)
        self.assertEqual(loud["declared_label"], loud["cases"])

    def test_paired_comparison_separates_conceding_from_recanting(self):
        paired = MODULE.ballot_paired(self.rows, 0, 1)
        self.assertEqual(paired["items"], 2)
        self.assertEqual(paired["held_first"], 2)
        self.assertEqual(paired["held_last"], 1)
        self.assertEqual(paired["conceded_during_exchange"], 1)
        self.assertEqual(paired["recanted_during_exchange"], 0)


class HomogenisationTest(unittest.TestCase):
    def test_identical_cases_score_full_overlap(self):
        stances = ["supportive", "neutral", "oppositional"]
        shared = "The article quotes the mayor at length about the budget plan."
        rows = [row("1", "neutral", stances, [[shared, shared, shared]])]
        report = MODULE.round_report(rows, 0, {}, 20)
        self.assertAlmostEqual(report["token_overlap"], 1.0)
        self.assertAlmostEqual(report["sentence_reuse"], 1.0)

    def test_reuse_is_symmetric_and_ignores_short_fragments(self):
        left = MODULE.sentences_of("A short one. " + "x" * 40 + ".", 20)
        right = MODULE.sentences_of("Different opener. " + "x" * 40 + ".", 20)
        self.assertEqual(len(left), 1)
        self.assertAlmostEqual(MODULE.reuse(left, right), 1.0)


class TruncationProxyTest(unittest.TestCase):
    def test_a_case_cut_mid_sentence_is_flagged(self):
        self.assertTrue(MODULE.looks_unfinished("the article goes on to say that the"))
        self.assertFalse(MODULE.looks_unfinished("the article says so."))
        self.assertFalse(MODULE.looks_unfinished('he called it "a plan"'))
        self.assertFalse(MODULE.looks_unfinished(""))

    def test_the_closing_declaration_is_not_mistaken_for_a_cut(self):
        # every compliant ToC case ends on an unpunctuated "Final stance:" line
        self.assertFalse(
            MODULE.looks_unfinished("The article says so.\nFinal stance: neutral")
        )
        self.assertTrue(
            MODULE.looks_unfinished("The article goes on to say that the\nFinal stance: neutral")
        )
        # a bare verdict is a judge/advocate compliance failure, not truncation
        self.assertFalse(MODULE.looks_unfinished("Final stance: neutral"))


class GroundingTest(unittest.TestCase):
    def test_a_fabricated_quote_fails_every_strictness(self):
        stances = ["supportive", "neutral", "oppositional"]
        real, fake = "예산안을 통과시켰다", "시장은 사퇴를 예고했다"
        rows = [row("1", "neutral", stances, [[f'"{real}"', f'"{fake}"', "no quotes here"]])]
        articles = {"1": f"의회는 {real} 그리고 회의는 끝났다"}
        report = MODULE.round_report(rows, 0, articles, 20)
        check = report["quote_verification"]
        self.assertEqual(check["quotes"], 2)
        self.assertAlmostEqual(check["loose"], 0.5)
        self.assertAlmostEqual(check["prefix12"], 0.5)
        # the shingle must be short enough to land inside a real quoted passage;
        # at twelve characters this read zero on cases that quoted accurately
        self.assertGreater(report["article_shingle_overlap"], 0.0)


class EndToEndTest(unittest.TestCase):
    def test_analyse_reads_a_saved_items_file(self):
        stances = ["supportive", "neutral", "oppositional"]
        rows = [
            row(
                "1", "supportive", stances,
                [
                    [case(s, declared=s) for s in stances],
                    [case("supportive", declared="neutral"),
                     case("neutral", declared="neutral"),
                     case("oppositional", declared="oppositional")],
                ],
                judged_round=1,
                pred="neutral",
                pred_source="judge",
                candidate_order=[
                    {"candidate_id": "A", "source_agent_index": 1, "stance_argued": "neutral"},
                    {"candidate_id": "B", "source_agent_index": 0, "stance_argued": "supportive"},
                    {"candidate_id": "C", "source_agent_index": 2, "stance_argued": "oppositional"},
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.items.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            report = MODULE.analyse(path, {}, 20)

        self.assertEqual(report["items"], 1)
        self.assertEqual(report["rounds"], 2)
        self.assertEqual(report["judged_round"], {1: 1})
        judge = report["judge"]
        self.assertEqual(judge["judged_items"], 1)
        # the judge read the round where the gold advocate had already conceded
        self.assertEqual(judge["gold_advocate_held"]["conceded"]["items"], 1)
        self.assertEqual(judge["gold_off_the_ballot"]["items"], 1)
        self.assertEqual(judge["winning_position"], {"A": 1})

    def test_mixed_round_counts_are_refused(self):
        stances = ["supportive", "neutral", "oppositional"]
        rows = [
            row("1", "neutral", stances, [[case(s) for s in stances]]),
            row("2", "neutral", stances, [[case(s) for s in stances]] * 2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.items.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(SystemExit):
                MODULE.usable_rows(path)


if __name__ == "__main__":
    unittest.main()

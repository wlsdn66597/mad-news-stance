"""Offline ceiling analysis for saved advocacy cases. No GPU, no model calls.

The advocacy design assigns one agent per label, so the gold label is argued on
every item and a literally perfect judge scores 1.0. That ceiling is vacuous.
The question this script answers is the useful one: **is the case that argues
gold distinguishable from the two that do not?** If nothing in the saved text
separates them, the cases are persuasive noise and no judge can recover the
answer from them.

For every item it ranks the three cases by a set of surface features and reports
how often the gold case comes first. A feature that beats chance (1/3) is signal
a judge could use; if none does, the design is done. It also reports what the
judge that actually ran was tracking, by measuring how often the judge's pick
agrees with each feature's pick and with candidate position.

    python scripts/advocacy_oracle.py results/advocacy/<run>.items.json
    python scripts/advocacy_oracle.py results/advocacy/*.items.json --json out.json
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advocacy import position_bias_chi_square, stated_label  # noqa: E402

QUOTE_PATTERN = re.compile(r"[\"“”'‘’「『]([^\"“”'‘’「』」]{6,200})[\"“”'‘’」』]")
HEDGES = (
    "little support", "does not support", "no explicit", "weak", "not strongly",
    "hardly", "insufficient", "difficult to argue", "does not clearly",
    "cannot be said", "at best", "arguably", "however", "although", "on the other hand",
    "근거가 부족", "보기 어렵", "단정하기", "그러나", "다만",
)
PEER_PATTERN = re.compile(r"analyst", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("items", nargs="+", help="one or more advocacy .items.json")
    parser.add_argument(
        "--round", default="saved",
        help="'saved' uses the cases the judge actually read; an index reads "
             "analyses_by_round[i] instead, so round 0 and round 1 can be "
             "compared without re-running anything.",
    )
    parser.add_argument(
        "--data-path",
        help="the dataset, so quoted passages can be checked against the article. "
             "Adds verified/unverified quote features -- a quote that is not in "
             "the article is a fabricated ground, and the judge cannot tell.",
    )
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def normalize(text):
    return " ".join(str(text or "").split())


def articles_by_id(data_path):
    if not data_path:
        return {}
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return {
        str(row["id"]): normalize(
            f"{row.get('haedline') or row.get('headline', '')} {row.get('article', '')}"
        )
        for row in rows
    }


def features(text, stance, article=None):
    text = text or ""
    lowered = text.lower()
    label, source = stated_label(text)
    quotes = QUOTE_PATTERN.findall(text)
    extra = {}
    if article is not None:
        # Khan et al. (ICML 2024) give their judge quotes tagged verified or
        # unverified against the source, and removing that grounding is what
        # lets an incorrect debater build a persuasive false narrative. Here
        # nothing is verified, so this counts what a verifier would have caught.
        verified = sum(1 for quote in quotes if normalize(quote) in article)
        extra = {
            "verified_quotes": float(verified),
            "unverified_quotes": float(len(quotes) - verified),
            "verified_quote_ratio": float(verified / len(quotes)) if quotes else 0.0,
        }
    return {**extra,
        "chars": float(len(text)),
        "words": float(len(text.split())),
        "quoted_passages": float(len(quotes)),
        "hedges": float(sum(lowered.count(cue) for cue in HEDGES)),
        "peer_mentions": float(len(PEER_PATTERN.findall(text))),
        # an advocate that explicitly declares a label other than the one it was
        # told to argue has conceded it cannot make the case
        "held_assigned": 0.0 if (source == "marker" and label != stance) else 1.0,
    }


def cases_of(row, which_round):
    if which_round == "saved":
        return row["advocate_cases"]
    index = int(which_round)
    analyses = row["analyses_by_round"][index]
    return [
        {"stance": stance, "analysis": analyses[i], "agent_index": i}
        for i, stance in enumerate(row["assigned_stances"])
    ]


def binomial_two_sided(hits, total, p_null=1 / 3):
    """Exact two-sided binomial p-value against picking a case at random."""
    if total == 0:
        return 1.0
    def pmf(k):
        return math.comb(total, k) * p_null ** k * (1 - p_null) ** (total - k)
    observed = pmf(hits)
    return min(1.0, sum(pmf(k) for k in range(total + 1) if pmf(k) <= observed * (1 + 1e-12)))


def rank_pick(scored, reverse):
    """The stance whose case ranks first; None when the top is tied."""
    best = max(scored.values()) if reverse else min(scored.values())
    winners = [stance for stance, value in scored.items() if value == best]
    return winners[0] if len(winners) == 1 else None


def analyse(path, which_round, articles=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = articles or {}
    feature_names = list(features("", "neutral", "" if articles else None))
    hits = {(name, direction): 0 for name in feature_names for direction in ("high", "low")}
    ties = dict.fromkeys(hits, 0)
    judge_agrees = dict.fromkeys(hits, 0)
    gold_conceded = 0
    nongold_conceded = 0
    positions = Counter()
    judged = 0

    for row in rows:
        gold = row["gold"]
        cases = cases_of(row, which_round)
        # an empty string, not None, when the item is missing from the dataset:
        # the feature set must stay the same shape across items
        article = articles.get(str(row["item_id"]), "") if articles else None
        scored = {
            case["stance"]: features(case["analysis"], case["stance"], article)
            for case in cases
        }
        if gold not in scored:
            continue
        judged += 1
        gold_conceded += scored[gold]["held_assigned"] == 0.0
        nongold_conceded += sum(
            values["held_assigned"] == 0.0 for stance, values in scored.items() if stance != gold
        )
        for name in feature_names:
            values = {stance: scored[stance][name] for stance in scored}
            for direction, reverse in (("high", True), ("low", False)):
                pick = rank_pick(values, reverse)
                key = (name, direction)
                if pick is None:
                    ties[key] += 1
                    continue
                hits[key] += pick == gold
                judge_agrees[key] += pick == row["pred"]
        for entry in row.get("candidate_order", []):
            if entry["stance_argued"] == row["pred"]:
                positions[entry["candidate_id"]] += 1

    accuracy = sum(row["pred"] == row["gold"] for row in rows) / max(1, len(rows))
    quote_totals = {}
    if articles:
        every = [
            features(case["analysis"], case["stance"], articles.get(str(row["item_id"])))
            for row in rows for case in cases_of(row, which_round)
            if articles.get(str(row["item_id"])) is not None
        ]
        total_quotes = sum(f["verified_quotes"] + f["unverified_quotes"] for f in every)
        quote_totals = {
            "cases_checked": len(every),
            "quoted_passages": total_quotes,
            "verified": sum(f["verified_quotes"] for f in every),
            "verified_rate": (
                sum(f["verified_quotes"] for f in every) / total_quotes if total_quotes else None
            ),
            "cases_with_no_verified_quote": sum(
                1 for f in every if f["verified_quotes"] == 0
            ),
        }
    predictors = []
    for (name, direction), correct in sorted(hits.items(), key=lambda kv: -kv[1]):
        decided = judged - ties[(name, direction)]
        predictors.append(
            {
                "feature": name,
                "direction": direction,
                "decided_items": decided,
                "gold_ranked_first": correct,
                "accuracy_on_decided": correct / max(1, decided),
                "p_vs_random_case": binomial_two_sided(correct, decided),
                "agrees_with_the_judge": judge_agrees[(name, direction)] / max(1, decided),
            }
        )
    return {
        "file": str(path),
        "round": which_round,
        "items": len(rows),
        "items_with_a_gold_case": judged,
        "saved_judge_accuracy": accuracy,
        "trivial_oracle_accuracy": 1.0 if judged == len(rows) else judged / max(1, len(rows)),
        "gold_case_conceded": gold_conceded,
        "non_gold_cases_conceded": nongold_conceded,
        "best_single_feature": max(predictors, key=lambda p: p["accuracy_on_decided"]),
        "predictors": predictors,
        "judge_position_counts": dict(positions),
        "judge_position_test": position_bias_chi_square(dict(positions)),
        "quote_grounding": quote_totals or None,
    }


def main():
    args = parse_args()
    articles = articles_by_id(args.data_path)
    reports = [analyse(path, args.round, articles) for path in args.items]
    for report in reports:
        print(f"\n===== {Path(report['file']).name}  (round={report['round']}) =====")
        print(f"items={report['items']}  saved judge accuracy={report['saved_judge_accuracy']:.4f}")
        print(f"trivial oracle (gold is argued on every item) = "
              f"{report['trivial_oracle_accuracy']:.4f}  <- vacuous by construction")
        print(f"the case arguing gold explicitly conceded on "
              f"{report['gold_case_conceded']}/{report['items_with_a_gold_case']} items "
              f"({report['non_gold_cases_conceded']} concessions among the "
              f"{2 * report['items_with_a_gold_case']} non-gold cases)")
        grounding = report.get("quote_grounding")
        if grounding and grounding["quoted_passages"]:
            print(f"quoted passages: {int(grounding['quoted_passages'])} across "
                  f"{grounding['cases_checked']} cases · "
                  f"{grounding['verified_rate']:.1%} actually appear in the article · "
                  f"{grounding['cases_with_no_verified_quote']}/{grounding['cases_checked']} "
                  f"cases cite nothing real")
        print("\nranking the three cases by one surface feature; chance = 0.3333")
        print(f"  {'feature':16} {'dir':5} {'decided':>7} {'acc':>7} {'p':>8} {'judge agrees':>13}")
        for row in report["predictors"]:
            print(f"  {row['feature']:16} {row['direction']:5} {row['decided_items']:7d} "
                  f"{row['accuracy_on_decided']:7.4f} {row['p_vs_random_case']:8.4f} "
                  f"{row['agrees_with_the_judge']:13.4f}")
        test = report["judge_position_test"]
        print(f"\njudge winning position {report['judge_position_counts']}"
              + (f"  chi2={test['chi_square']:.2f} df={test['df']} p={test['p_value']:.4f}"
                 if test.get("p_value") is not None else ""))
        best = report["best_single_feature"]
        print(f"reachable from these features alone: {best['accuracy_on_decided']:.4f} "
              f"({best['feature']}/{best['direction']}) vs the judge's "
              f"{report['saved_judge_accuracy']:.4f}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

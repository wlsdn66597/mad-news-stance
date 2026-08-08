"""Neutral as the abstention class, tested offline on a saved run. No GPU.

EXAONE-4.0-1.2B recovers 92% of supportive articles and 21% of neutral ones: it
does not detect neutral, it defaults to supportive. The proposal is to stop
asking a weak model to recognise neutral as a third positive class and instead
let neutral be where an unresolved polar decision falls -- which is also how the
debate literature is actually formulated, two experts arguing two sides.

That is an aggregation rule, so it can be measured on runs that already exist,
before any of it is built. Each rule below replaces the vote with `neutral` on
some subset and leaves the rest alone:

    polar_split      both supportive and oppositional present in the final round
    non_unanimous    the agents did not all agree
    tie              a 1:1:1 final round
    unstable         a tie, or round 0 and the final round disagree
    round0_polar     polar disagreement at round 0 rather than at the end

**These rules are being selected on the same split they are scored on.** A rule
that wins here has been fitted to this test set; confirming it needs another
split or another seed. The per-class table is printed because a rule can raise
accuracy purely by trading supportive recall for neutral recall, which is worth
seeing rather than hiding inside one number.

    python scripts/analyze_neutral_abstention.py --result results/phase2/<run>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import (  # noqa: E402
    LABELS,
    aggregate,
    mcnemar_exact,
    normalize_label,
    transition_metrics,
    unique_majority,
)
from src.consensus_io import load_json, result_records  # noqa: E402
from src.metrics import macro_f1, per_class_prf  # noqa: E402

POLAR = ("supportive", "oppositional")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="saved phase2 result with a debate trace")
    parser.add_argument("--aggregation-method", default="current_final_majority")
    parser.add_argument("--fallback-seed", type=int, default=0)
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def labels_of(values):
    return {normalize_label(v) for v in values} - {None}


def rules(round_predictions):
    """Which rules fire on this item."""
    first, final = round_predictions[0], round_predictions[-1]
    final_labels, first_labels = labels_of(final), labels_of(first)
    vote_first, vote_final = unique_majority(first), unique_majority(final)
    unanimous = len(final_labels) == 1
    return {
        "polar_split": set(POLAR) <= final_labels,
        "non_unanimous": not unanimous,
        "tie": vote_final.tied,
        "unstable": bool(
            vote_final.tied
            or (vote_first.label and vote_final.label and vote_first.label != vote_final.label)
        ),
        "round0_polar": set(POLAR) <= first_labels,
    }


def main():
    args = parse_args()
    records = result_records(load_json(args.result))
    rows = []
    for record in records:
        rounds = record["round_predictions"]
        base = aggregate(args.aggregation_method, rounds, record["id"], args.fallback_seed)
        rows.append({
            "gold": record.get("gold"),
            "base": base["pred"],
            "fires": rules(rounds),
        })
    rows = [row for row in rows if row["gold"] in LABELS]
    gold = [row["gold"] for row in rows]
    base = [row["base"] for row in rows]
    base_correct = sum(p == g for p, g in zip(base, gold))

    print(f"[items] {len(rows)}   gold: "
          + ", ".join(f"{label} {gold.count(label)}" for label in LABELS))
    print(f"[baseline] {args.aggregation_method}: {base_correct}/{len(rows)} = "
          f"{base_correct / len(rows):.4f}  macro-F1 {macro_f1(base, gold, LABELS):.4f}")

    def per_class(preds):
        table = per_class_prf(preds, gold, LABELS)
        return "  ".join(
            f"{label[:4]} {table[label]['recall']:.3f}" for label in LABELS
        )

    print(f"  recall by class: {per_class(base)}")

    report = {"items": len(rows), "baseline_accuracy": base_correct / len(rows), "rules": {}}
    names = list(rows[0]["fires"])
    print(f"\n{'rule':16} {'fires':>6} {'acc':>7} {'macroF1':>8} "
          f"{'w2c':>5} {'c2w':>5} {'net':>5} {'p':>7}   recall by class")
    for name in names:
        preds = ["neutral" if row["fires"][name] else row["base"] for row in rows]
        correct = sum(p == g for p, g in zip(preds, gold))
        fires = sum(row["fires"][name] for row in rows)
        t = transition_metrics(base, preds, gold)
        stats = mcnemar_exact(base, preds, gold)
        print(f"{name:16} {fires:6d} {correct / len(rows):7.4f} "
              f"{macro_f1(preds, gold, LABELS):8.4f} "
              f"{t['wrong_to_correct']:5d} {t['correct_to_wrong']:5d} "
              f"{t['net_improvement']:+5d} {stats['p_value_two_sided']:7.3f}   {per_class(preds)}")
        report["rules"][name] = {
            "fires": fires, "accuracy": correct / len(rows),
            "macro_f1": macro_f1(preds, gold, LABELS), **t, **stats,
        }

    # what abstention could ever buy: neutral wherever the vote is wrong and
    # the gold is neutral, and never otherwise
    oracle = ["neutral" if g == "neutral" else p for p, g in zip(base, gold)]
    oracle_correct = sum(p == g for p, g in zip(oracle, gold))
    print(f"\n[ceiling] perfect abstention (neutral exactly when gold is neutral, "
          f"vote kept otherwise): {oracle_correct}/{len(rows)} = "
          f"{oracle_correct / len(rows):.4f}")
    print("rules are selected and scored on the same split; a winner here is "
          "fitted to it and needs another split or seed to confirm")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

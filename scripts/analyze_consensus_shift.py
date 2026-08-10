"""What happens to the items the extra rounds talk into agreement? No GPU.

Persona debate is more accurate at four rounds than at two, but persona debate
*plus a judge* is better at two. The two facts point at the same mechanism from
opposite sides: extra rounds raise the agents' own accuracy a little while
converging the ensemble a lot, and the judge is only called where the agents
disagree. Rounds three and four therefore buy a few items directly and spend
some of the disagreement signal that would have routed harder items to the
judge.

This isolates the items that pay for it: **non-unanimous at 2R, unanimous at
4R**. For those it reports what the 2R vote said, what the 4R consensus settled
on, and -- when the two judge runs are supplied -- what the judge made of them
back when it still saw them. If the consensus they reached is worse than the
verdict the judge gave, that is the whole explanation.

    python scripts/analyze_consensus_shift.py \
      --two-round results/phase2/<run>_a3_r2_personas.json \
      --four-round results/phase2/<run>_a3_r4_personas.json \
      --judge-two-round results/selective_judge_report/<...r2...>.items.json \
      --judge-four-round results/selective_judge_report/<...r4...>.items.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import (  # noqa: E402
    LABELS,
    mcnemar_exact,
    normalize_label,
    transition_metrics,
    unique_majority,
)
from src.consensus_io import load_json, result_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-round", required=True)
    parser.add_argument("--four-round", required=True)
    parser.add_argument("--judge-two-round")
    parser.add_argument("--judge-four-round")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def final_round(path):
    """item id -> (agent labels at the last round, majority label, gold)."""
    out = {}
    for record in result_records(load_json(path)):
        labels = [normalize_label(v) for v in record["round_predictions"][-1]]
        vote = unique_majority(labels)
        pick = vote.label or next((v for v in labels if v in LABELS), None)
        out[str(record["id"])] = {
            "labels": labels,
            "unanimous": len({v for v in labels if v} ) == 1,
            "majority": pick,
            "gold": record.get("gold"),
        }
    return out


def judge_predictions(path):
    if not path:
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(row["item_id"]): {
            "triggered": bool(row.get("triggered")),
            "final": row.get("final_prediction"),
            "judge": row.get("judge_prediction"),
        }
        for row in rows
    }


def accuracy(keys, pick, gold):
    if not keys:
        return None
    return sum(pick[k] == gold[k] for k in keys) / len(keys)


def main():
    args = parse_args()
    two, four = final_round(args.two_round), final_round(args.four_round)
    j2, j4 = judge_predictions(args.judge_two_round), judge_predictions(args.judge_four_round)

    keys = sorted(set(two) & set(four))
    keys = [k for k in keys if two[k]["gold"] in LABELS]
    if not keys:
        raise SystemExit("the two runs share no items with a usable gold label")
    gold = {k: two[k]["gold"] for k in keys}

    groups = {}
    for k in keys:
        groups.setdefault((two[k]["unanimous"], four[k]["unanimous"]), []).append(k)

    print(f"[items] {len(keys)}")
    print(f"  non-unanimous at 2R: {sum(not two[k]['unanimous'] for k in keys)}")
    print(f"  non-unanimous at 4R: {sum(not four[k]['unanimous'] for k in keys)}")

    print(f"\n[what the extra rounds did]")
    print(f"  {'2R':>14} {'4R':>14} {'n':>5} {'2R acc':>8} {'4R acc':>8} "
          f"{'w2c':>5} {'c2w':>5} {'net':>5} {'p':>7}")
    report = {"items": len(keys), "groups": {}}
    for (u2, u4), group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        before = [two[k]["majority"] for k in group]
        after = [four[k]["majority"] for k in group]
        g = [gold[k] for k in group]
        t = transition_metrics(before, after, g)
        stats = mcnemar_exact(before, after, g)
        name2 = "unanimous" if u2 else "split"
        name4 = "unanimous" if u4 else "split"
        print(f"  {name2:>14} {name4:>14} {len(group):5d} "
              f"{accuracy(group, {k: two[k]['majority'] for k in group}, gold):8.4f} "
              f"{accuracy(group, {k: four[k]['majority'] for k in group}, gold):8.4f} "
              f"{t['wrong_to_correct']:5d} {t['correct_to_wrong']:5d} "
              f"{t['net_improvement']:+5d} {stats['p_value_two_sided']:7.3f}")
        report["groups"][f"2R_{name2}__4R_{name4}"] = {"n": len(group), **t, **stats}

    # the group that pays for the extra rounds
    talked_round = groups.get((False, True), [])
    if not talked_round:
        print("\nno item went from split to unanimous; nothing was talked into agreement")
        return

    print(f"\n[talked into agreement]  {len(talked_round)} items, split at 2R and "
          f"unanimous at 4R")
    consensus = Counter(four[k]["majority"] for k in talked_round)
    right = Counter(four[k]["majority"] for k in talked_round if four[k]["majority"] == gold[k])
    for label in LABELS:
        if consensus[label]:
            print(f"  settled on {label:13} {consensus[label]:4d} times, "
                  f"{right[label]:4d} correct ({right[label] / consensus[label]:.1%})")

    if j2 or j4:
        print(f"\n[and what the judge would have said about them]")
        rows = []
        rows.append(("2R vote", accuracy(talked_round,
                                         {k: two[k]["majority"] for k in talked_round}, gold)))
        if j2:
            fired = [k for k in talked_round if j2.get(k, {}).get("triggered")]
            rows.append((f"2R + judge ({len(fired)} of them were judged)",
                         accuracy(talked_round,
                                  {k: j2.get(k, {}).get("final") for k in talked_round}, gold)))
        rows.append(("4R consensus", accuracy(talked_round,
                                              {k: four[k]["majority"] for k in talked_round}, gold)))
        if j4:
            fired4 = [k for k in talked_round if j4.get(k, {}).get("triggered")]
            rows.append((f"4R + judge ({len(fired4)} still judged)",
                         accuracy(talked_round,
                                  {k: j4.get(k, {}).get("final") for k in talked_round}, gold)))
        for name, value in rows:
            if value is not None:
                print(f"  {name:44} {value:.4f}")
                report.setdefault("talked_into_agreement", {})[name] = value
        print("\nIf '2R + judge' beats '4R consensus' on these items, the extra rounds "
              "cost more by hiding them from the judge than they gained by settling "
              "them.")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

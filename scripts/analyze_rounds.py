"""Round-by-round dynamics of a saved debate run: does another round pay off?

    python scripts/analyze_rounds.py results/phase2/<run>_a3_r4.json
    python scripts/analyze_rounds.py results/phase2/<run>_a3_r4.json \
      --compare results/phase2/<run>_a3_r2.json

Reads only the saved trace, never calls a model. Gold labels are used for
evaluation only. Works for any number of rounds and agents.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import judge_trigger, unique_majority  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--method", default="debate")
    parser.add_argument("--compare", help="another result file to compare final predictions with")
    parser.add_argument("--json", dest="json_out", help="write the full report here")
    return parser.parse_args()


def mcnemar(before, after, gold):
    """Exact two-sided McNemar over paired correctness."""
    b = sum(x == g and y != g for x, y, g in zip(before, after, gold))
    c = sum(x != g and y == g for x, y, g in zip(before, after, gold))
    n = b + c
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {"correct_to_wrong": b, "wrong_to_correct": c, "net": c - b, "p_value": p}


def round_labels(data, method, parse):
    """Per-round agent labels for every item, oldest round first."""
    rows = []
    for item_id, item in data[method].items():
        trace = item.get("debate_trace") or {}
        answers = trace.get("answers_by_round")
        if not answers:
            raise SystemExit(f"item {item_id} has no debate_trace.answers_by_round")
        rows.append(
            {
                "id": item_id,
                "gold": item["gold"],
                "saved_pred": item["pred"],
                "rounds": [[parse(answer) for answer in answers_of] for answers_of in answers],
            }
        )
    return rows


def main():
    args = parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    meta = data.get("_meta", {})
    if args.method not in data:
        raise SystemExit(f"{args.result} has no '{args.method}' results")
    task = Stance(meta.get("data_path", ""), prompt_profile=meta.get("prompt_profile", "legacy_ko"))
    rows = round_labels(data, args.method, task.parse)
    n = len(rows)
    n_rounds = max(len(row["rounds"]) for row in rows)
    n_agents = max(len(values) for row in rows for values in row["rounds"])
    gold = [row["gold"] for row in rows]

    print(f"[run] {Path(args.result).name}")
    print(f"[run] model={meta.get('model_id')} profile={meta.get('prompt_profile')} "
          f"seed={meta.get('run_seed')} items={n} agents={n_agents} rounds={n_rounds}")

    majority = [
        [unique_majority(row["rounds"][r]).label if r < len(row["rounds"]) else None
         for row in rows]
        for r in range(n_rounds)
    ]
    accuracies = [sum(p == g for p, g in zip(preds, gold)) / n for preds in majority]
    saved = [row["saved_pred"] for row in rows]
    print("\n[accuracy] " + "  ".join(
        f"round{r}={accuracy:.4f}" for r, accuracy in enumerate(accuracies)
    ) + f"   (saved pred={sum(p == g for p, g in zip(saved, gold)) / n:.4f})")

    report = {"file": args.result, "items": n, "n_rounds": n_rounds, "n_agents": n_agents,
              "round_accuracy": accuracies, "rounds": [], "meta": meta}

    print("\n[per round] majority label movement")
    for r in range(1, n_rounds):
        before, after = majority[r - 1], majority[r]
        changed = sum(x != y for x, y in zip(before, after))
        stats = mcnemar(before, after, gold)
        flips = Counter()
        for row in rows:
            if r >= len(row["rounds"]):
                continue
            for was, now in zip(row["rounds"][r - 1], row["rounds"][r]):
                if was == now:
                    continue
                flips["flipped"] += 1
                flips["wrong_to_correct" if now == row["gold"] else
                      "correct_to_wrong" if was == row["gold"] else "wrong_to_wrong"] += 1
        answers = sum(len(row["rounds"][r]) for row in rows if r < len(row["rounds"]))
        print(f"  r{r-1}->r{r}  changed={changed:4d}/{n}  "
              f"wrong->correct={stats['wrong_to_correct']:3d}  "
              f"correct->wrong={stats['correct_to_wrong']:3d}  "
              f"net={stats['net']:+d}  McNemar p={stats['p_value']:.3f}")
        print(f"           agents flipped={flips['flipped']:4d}/{answers}  "
              f"w->c={flips['wrong_to_correct']:3d}  c->w={flips['correct_to_wrong']:3d}  "
              f"w->w={flips['wrong_to_wrong']:3d}")
        report["rounds"].append({"from": r - 1, "to": r, "changed": changed,
                                 "majority": stats, "agent_flips": dict(flips)})

    unanimity = [
        sum(1 for row in rows if r < len(row["rounds"]) and len(set(row["rounds"][r])) == 1)
        for r in range(n_rounds)
    ]
    print("\n[unanimity] " + "  ".join(
        f"round{r}={count}/{n} ({count/n:.1%})" for r, count in enumerate(unanimity)
    ))
    report["unanimity"] = unanimity

    # 유형 분석 1: round 0 -> last round, per item
    transitions = Counter()
    for row, before, after in zip(rows, majority[0], majority[-1]):
        was, now = before == row["gold"], after == row["gold"]
        transitions["오답 -> 정답" if (not was and now) else
                    "정답 -> 오답" if (was and not now) else
                    "정답 유지" if was else "오답 유지"] += 1
    print(f"\n[type] round0 -> round{n_rounds - 1}")
    for label in ("오답 -> 정답", "정답 -> 오답", "정답 유지", "오답 유지"):
        print(f"  {label:12} {transitions[label]:5d}")
    report["type_round0_to_final"] = dict(transitions)

    # 유형 분석 2: how the agents split in the final round
    def signature(values):
        counts = Counter(value for value in values if value is not None)
        if not counts:
            return "파싱 실패"
        pattern = ":".join(str(count) for count in sorted(counts.values(), reverse=True))
        unparsed = sum(1 for value in values if value is None)
        return f"{pattern} (+{unparsed} 파싱 실패)" if unparsed else pattern

    splits = Counter(signature(row["rounds"][-1]) for row in rows)
    print(f"\n[agreement] final round agent split")
    for pattern, count in sorted(splits.items(), key=lambda kv: -kv[1]):
        name = "만장일치" if pattern == str(n_agents) else pattern
        print(f"  {name:22} {count:5d}")
    report["final_round_split"] = dict(splits)

    final = majority[-1]
    wrong = [row for row, pred in zip(rows, final) if pred != row["gold"]]
    holds = [row for row in wrong if row["gold"] in row["rounds"][-1]]
    ever = [row for row in wrong
            if any(row["gold"] in values for values in row["rounds"])]
    correct_rows = [row for row, pred in zip(rows, final) if pred == row["gold"]]
    fragile = [row for row in correct_rows if len(set(row["rounds"][-1])) > 1]
    print(f"\n[headroom] wrong at the last round: {len(wrong)}")
    print(f"           some agent still holds the gold label: {len(holds)} "
          f"({len(holds)/max(1,len(wrong)):.1%})  <- the only items more rounds could fix")
    print(f"           gold never proposed in any round:       {len(wrong)-len(ever)}")
    print(f"[headroom] correct at the last round: {len(correct_rows)}, "
          f"of which not unanimous (can still be lost): {len(fragile)}")
    report["headroom"] = {
        "wrong": len(wrong), "some_agent_holds_gold": len(holds),
        "gold_never_proposed": len(wrong) - len(ever),
        "correct": len(correct_rows), "correct_but_split": len(fragile),
    }

    # How much work a selective judge would have on this configuration.
    triggered, reasons, holds_gold, trig_correct = 0, Counter(), 0, 0
    for row in rows:
        fired, why = judge_trigger([row["rounds"][0], row["rounds"][-1]], "instability")
        if not fired:
            continue
        triggered += 1
        reasons[why] += 1
        if row["gold"] in row["rounds"][-1]:
            holds_gold += 1
        if unique_majority(row["rounds"][-1]).label == row["gold"]:
            trig_correct += 1
    print(f"\n[judge trigger] instability fires on {triggered}/{n} ({triggered/n:.1%})  "
          f"{dict(reasons)}")
    print(f"                accuracy there={trig_correct}/{triggered}"
          f"={trig_correct/max(1, triggered):.4f}  "
          f"some agent holds gold={holds_gold}/{triggered}")
    report["judge_trigger"] = {
        "triggered": triggered, "reasons": dict(reasons),
        "accuracy_on_triggered": trig_correct / max(1, triggered),
        "some_agent_holds_gold": holds_gold,
    }

    print("\n[distribution] gold  " + str(dict(Counter(gold))))
    print("[distribution] final " + str(dict(Counter(final))))
    report["distribution"] = {"gold": dict(Counter(gold)), "final": dict(Counter(final))}

    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        if args.method not in other:
            raise SystemExit(f"{args.compare} has no '{args.method}' results")
        shared = sorted(set(data[args.method]) & set(other[args.method]))
        mine = [data[args.method][i]["pred"] for i in shared]
        theirs = [other[args.method][i]["pred"] for i in shared]
        golds = [data[args.method][i]["gold"] for i in shared]
        stats = mcnemar(theirs, mine, golds)
        same = sum(x == y for x, y in zip(mine, theirs))
        print(f"\n[compare] vs {Path(args.compare).name} on {len(shared)} shared items")
        print(f"          accuracy {sum(t == g for t, g in zip(theirs, golds))/len(shared):.4f}"
              f" -> {sum(m == g for m, g in zip(mine, golds))/len(shared):.4f}"
              f"   identical predictions={same}/{len(shared)}")
        print(f"          wrong->correct={stats['wrong_to_correct']}  "
              f"correct->wrong={stats['correct_to_wrong']}  net={stats['net']:+d}  "
              f"McNemar p={stats['p_value']:.3f}")
        report["compare"] = {"file": args.compare, "shared_items": len(shared),
                             "identical_predictions": same, **stats}

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

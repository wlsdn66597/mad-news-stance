"""Paired comparison of saved methods, within or across result files.

    python scripts/compare_methods.py \
      results/phase2/<r2>.json:single \
      results/phase2/<r2>.json:majority \
      results/phase2/<r4>.json:debate=debate_r4 \
      results/selective_judge/<run>.items.json:final_prediction=debate_r4+judge

Each argument is `path:field` with an optional `=label`. For a phase2 result the
field is the method name; for a selective-judge `.items.json` it is a per-item
prediction field (`final_prediction`, `judge_prediction`, `baseline_prediction`).
Every pair is compared on the items they share, with exact McNemar over paired
correctness. Reads only saved predictions; gold is used for evaluation only.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", help="path:method[=label]")
    parser.add_argument(
        "--exclude-parse-errors", action="store_true",
        help="drop every item on which any of the runs hit a judge parse error. "
             "Use when the runs were produced by different judge parsing code: "
             "those are the only items whose prediction the code change could "
             "have moved, so the remainder is still a clean paired comparison.",
    )
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def load(spec):
    body, _, label = spec.partition("=")
    path, _, field = body.rpartition(":")
    if not path or not field:
        raise SystemExit(f"expected path:field[=label], got {spec!r}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        # selective-judge .items.json: one row per item
        missing = [row for row in data if field not in row]
        if missing:
            raise SystemExit(
                f"{path} rows have no '{field}'; expected one of "
                "final_prediction, judge_prediction, baseline_prediction"
            )
        pred = {str(row["item_id"]): row[field] for row in data}
        gold = {str(row["item_id"]): row["gold"] for row in data}
        parse_errors = {
            str(row["item_id"]) for row in data
            if any(a.get("parse_error") for a in (row.get("judge_attempts") or []))
        }
    else:
        if field not in data:
            raise SystemExit(f"{path} has no '{field}' results")
        items = data[field]
        pred = {key: value["pred"] for key, value in items.items()}
        gold = {key: value["gold"] for key, value in items.items()}
        parse_errors = set()
    return {
        "label": label or f"{field}@{Path(path).stem[-12:]}",
        "path": path,
        "method": field,
        "pred": pred,
        "gold": gold,
        "parse_errors": parse_errors,
    }


def mcnemar(before, after, gold):
    b = sum(x == g and y != g for x, y, g in zip(before, after, gold))
    c = sum(x != g and y == g for x, y, g in zip(before, after, gold))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n) if n else 1.0
    return {"correct_to_wrong": b, "wrong_to_correct": c, "net": c - b, "p_value": p}


def main():
    args = parse_args()
    runs = [load(spec) for spec in args.specs]
    shared = set.intersection(*[set(run["pred"]) for run in runs])
    if args.exclude_parse_errors:
        flagged = set.union(*[run["parse_errors"] for run in runs])
        dropped = shared & flagged
        shared -= flagged
        print(f"[filter] dropped {len(dropped)} items with a judge parse error in "
              f"at least one run")
    shared = sorted(shared)
    if not shared:
        raise SystemExit("the runs share no item ids")
    gold = [runs[0]["gold"][key] for key in shared]

    print(f"[items] {len(shared)} shared")
    report = {"shared_items": len(shared), "accuracy": {}, "pairs": []}
    for run in runs:
        correct = sum(run["pred"][key] == g for key, g in zip(shared, gold))
        report["accuracy"][run["label"]] = correct / len(shared)
        print(f"  {run['label']:24} correct={correct:4d}/{len(shared)} "
              f"acc={correct / len(shared):.4f}   ({run['method']} in {Path(run['path']).name})")

    print("\n[pairs] exact McNemar on paired correctness")
    for first, second in itertools.combinations(runs, 2):
        before = [first["pred"][key] for key in shared]
        after = [second["pred"][key] for key in shared]
        stats = mcnemar(before, after, gold)
        same = sum(x == y for x, y in zip(before, after))
        print(f"  {first['label']:20} -> {second['label']:20} "
              f"same={same:4d}/{len(shared)}  "
              f"wrong->correct={stats['wrong_to_correct']:3d}  "
              f"correct->wrong={stats['correct_to_wrong']:3d}  "
              f"net={stats['net']:+d}  p={stats['p_value']:.3f}")
        report["pairs"].append({"from": first["label"], "to": second["label"],
                                "identical_predictions": same, **stats})

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

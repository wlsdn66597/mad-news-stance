"""Paired comparison of saved methods, within or across result files.

    python scripts/compare_methods.py \
      results/phase2/<r2>.json:single \
      results/phase2/<r2>.json:majority \
      results/phase2/<r2>.json:debate \
      results/phase2/<r4>.json:debate=debate_r4

Each argument is `path:method` with an optional `=label`. Every pair is compared
on the items they share, with exact McNemar over paired correctness. Reads only
saved predictions; gold is used for evaluation only.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", help="path:method[=label]")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def load(spec):
    body, _, label = spec.partition("=")
    path, _, method = body.rpartition(":")
    if not path or not method:
        raise SystemExit(f"expected path:method[=label], got {spec!r}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if method not in data:
        raise SystemExit(f"{path} has no '{method}' results")
    items = data[method]
    return {
        "label": label or f"{method}@{Path(path).stem[-12:]}",
        "path": path,
        "method": method,
        "pred": {key: value["pred"] for key, value in items.items()},
        "gold": {key: value["gold"] for key, value in items.items()},
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
    shared = sorted(set.intersection(*[set(run["pred"]) for run in runs]))
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

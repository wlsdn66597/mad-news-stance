"""Offline: is the disagreement trigger a good router? No GPU, no model calls.

Once the ablations show that the rationales contribute nothing, selective
advocacy is a router: keep a weak model's answer on most items and hand the rest
to a strong model. The remaining question is whether the trigger picks the right
items, and two saved full-split runs answer it without generating anything --
substitute the strong run's saved prediction on the triggered items and score.

Three references make the answer readable:

- **random routing** at the same coverage, sampled, with the fraction of draws
  that match or beat the trigger. A trigger that cannot beat picking items at
  random is not a trigger, it is a budget.
- **oracle routing** at the same coverage: route the items the strong model gets
  right and the weak model gets wrong, first. The ceiling for any router.
- **the strong model everywhere**, which is what the extra calls buy at 100%
  coverage.

    python scripts/simulate_routing.py \
      --weak results/phase2/<exaone>.json --weak-method debate \
      --strong results/phase2/<qwen>.json --strong-method majority \
      --trigger unstable_or_split
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus_io import load_json, result_records  # noqa: E402
from src.selective_advocacy import TRIGGERS, selective_trigger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak", required=True, help="saved phase2 result with a debate trace")
    parser.add_argument("--weak-method", default="debate")
    parser.add_argument("--strong", required=True, help="saved phase2 result for the strong model")
    parser.add_argument("--strong-method", default="majority")
    parser.add_argument("--trigger", choices=list(TRIGGERS), default="unstable_or_split")
    parser.add_argument("--trigger-scope", choices=["last", "any_round"], default="last")
    parser.add_argument("--samples", type=int, default=5000, help="random-routing draws")
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def method_predictions(path, method):
    data = load_json(path)
    if method not in data:
        raise SystemExit(f"{path} has no '{method}' results; found {sorted(k for k in data if k != '_meta')}")
    return {str(key): value["pred"] for key, value in data[method].items()}, {
        str(key): value["gold"] for key, value in data[method].items()
    }


def main():
    args = parse_args()
    weak_pred, gold = method_predictions(args.weak, args.weak_method)
    strong_pred, strong_gold = method_predictions(args.strong, args.strong_method)

    triggered = set()
    for record in result_records(load_json(args.weak)):
        fires, _, _ = selective_trigger(
            record["round_predictions"], args.trigger, args.trigger_scope
        )
        if fires:
            triggered.add(str(record["id"]))

    items = sorted(set(weak_pred) & set(strong_pred) & set(gold))
    if not items:
        raise SystemExit("the two runs share no item ids")
    disagreeing_gold = [key for key in items if gold[key] != strong_gold.get(key, gold[key])]
    if disagreeing_gold:
        raise SystemExit(f"the two runs disagree on gold for {len(disagreeing_gold)} items")

    weak_ok = {key: weak_pred[key] == gold[key] for key in items}
    strong_ok = {key: strong_pred[key] == gold[key] for key in items}
    routed = triggered & set(items)
    coverage = len(routed)

    def score(route_set):
        return sum(
            strong_ok[key] if key in route_set else weak_ok[key] for key in items
        )

    base = sum(weak_ok.values())
    everything = sum(strong_ok.values())
    actual = score(routed)

    # random routing at the same coverage
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.samples):
        draws.append(score(set(rng.sample(items, coverage))))
    draws.sort()
    at_least = sum(1 for value in draws if value >= actual) / max(1, len(draws))

    # oracle routing at the same coverage: the flips that help, first
    gains = sorted(
        items, key=lambda key: (strong_ok[key] - weak_ok[key]), reverse=True
    )
    oracle = score(set(gains[:coverage]))

    routed_weak = sum(weak_ok[key] for key in routed)
    routed_strong = sum(strong_ok[key] for key in routed)

    report = {
        "items": len(items),
        "weak": {"file": args.weak, "method": args.weak_method, "correct": base,
                 "accuracy": base / len(items)},
        "strong": {"file": args.strong, "method": args.strong_method, "correct": everything,
                   "accuracy": everything / len(items)},
        "trigger": args.trigger,
        "coverage": coverage,
        "coverage_ratio": coverage / len(items),
        "routed": {"correct": actual, "accuracy": actual / len(items)},
        "on_the_routed_subset": {
            "weak_correct": routed_weak,
            "strong_correct": routed_strong,
            "weak_accuracy": routed_weak / max(1, coverage),
            "strong_accuracy": routed_strong / max(1, coverage),
        },
        "random_routing": {
            "samples": args.samples,
            "mean_correct": sum(draws) / max(1, len(draws)),
            "p05_correct": draws[int(0.05 * len(draws))],
            "p95_correct": draws[min(len(draws) - 1, int(0.95 * len(draws)))],
            "fraction_at_least_as_good_as_the_trigger": at_least,
        },
        "oracle_routing_at_the_same_coverage": {
            "correct": oracle, "accuracy": oracle / len(items)
        },
    }

    n = len(items)
    print(f"[items] {n} shared · trigger {args.trigger} routes {coverage} ({coverage / n:.1%})")
    print(f"\n  {'condition':34} {'correct':>8} {'acc':>8}")
    print(f"  {'weak everywhere (' + args.weak_method + ')':34} {base:8d} {base / n:8.4f}")
    print(f"  {'trigger routing':34} {actual:8d} {actual / n:8.4f}")
    print(f"  {'random routing, same coverage':34} {report['random_routing']['mean_correct']:8.1f} "
          f"{report['random_routing']['mean_correct'] / n:8.4f}"
          f"   [{report['random_routing']['p05_correct']}, "
          f"{report['random_routing']['p95_correct']}] 90%")
    print(f"  {'oracle routing, same coverage':34} {oracle:8d} {oracle / n:8.4f}")
    print(f"  {'strong everywhere (' + args.strong_method + ')':34} {everything:8d} "
          f"{everything / n:8.4f}")
    print(f"\non the {coverage} routed items: weak {routed_weak}/{coverage} "
          f"({routed_weak / max(1, coverage):.4f}) -> strong {routed_strong}/{coverage} "
          f"({routed_strong / max(1, coverage):.4f})")
    print(f"random routing matched or beat the trigger in {at_least:.1%} of "
          f"{args.samples} draws   <- high means the trigger is only buying coverage")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

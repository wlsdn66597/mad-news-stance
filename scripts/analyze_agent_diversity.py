"""How independent are the agents? Offline, over a saved run. No GPU.

An ensemble is worth more than one member only to the extent its members fail
differently. In this repository they do not: `single` and `majority k=3` score
identically for EXAONE-4.0-1.2B (0.4865 both), which is what happens when three
samples of the same prompt are near-copies of each other. Role differentiation
-- giving the agents distinct personas or distinct jobs rather than the same
instruction three times -- is an intervention on exactly this quantity, so it is
worth measuring before it is built.

Reported here:

- **pairwise agreement** between agents, at round 0 and at the end. Debate
  should raise it; the question is how much room there was to begin with.
- **the candidate pool**: how often at least one agent names the gold label.
  This is the hard ceiling on any aggregation rule, and it is what a more
  diverse ensemble would move.
- **observed versus independent all-wrong rate.** If the agents failed
  independently at their measured accuracy, all three would miss far less
  often than they do. The gap is the cost of the correlation.
- **per-agent accuracy**, to check the pool is not just one strong agent
  carrying two weak ones.

    python scripts/analyze_agent_diversity.py --result results/phase2/<run>.json
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import LABELS, normalize_label  # noqa: E402
from src.consensus_io import load_json, result_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="saved phase2 result with a debate trace")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def agreement(rounds):
    """Mean pairwise agreement among the agents in one round."""
    pairs = [
        normalize_label(a) == normalize_label(b)
        for a, b in itertools.combinations(rounds, 2)
    ]
    return sum(pairs) / len(pairs) if pairs else None


def main():
    args = parse_args()
    records = [
        r for r in result_records(load_json(args.result)) if r.get("gold") in LABELS
    ]
    if not records:
        raise SystemExit("no items with a usable gold label")
    n_agents = max(len(r["round_predictions"][0]) for r in records)
    gold = [r["gold"] for r in records]

    first = [[normalize_label(v) for v in r["round_predictions"][0]] for r in records]
    final = [[normalize_label(v) for v in r["round_predictions"][-1]] for r in records]

    print(f"[items] {len(records)}   agents {n_agents}   "
          f"rounds {len(records[0]['round_predictions'])}")

    for name, rounds in (("round 0", first), ("final round", final)):
        agree = sum(agreement(r) for r in rounds) / len(rounds)
        distinct = Counter(len(set(r) - {None}) for r in rounds)
        print(f"\n[{name}] mean pairwise agreement {agree:.4f}")
        print("  distinct labels on the table: "
              + ", ".join(f"{k} on {distinct[k]} items" for k in sorted(distinct)))

    # the pool: what any aggregation rule could possibly reach
    pool_first = sum(g in set(r) for r, g in zip(first, gold))
    pool_final = sum(g in set(r) for r, g in zip(final, gold))
    print(f"\n[candidate pool] at least one agent names the gold label")
    print(f"  round 0     {pool_first}/{len(records)} = {pool_first / len(records):.4f}"
          f"   <- the hard ceiling for majority")
    print(f"  final round {pool_final}/{len(records)} = {pool_final / len(records):.4f}"
          f"   ({pool_final - pool_first:+d} from the exchange)")

    # per-agent accuracy, and what independence would have given
    per_agent = [
        sum(1 for r, g in zip(first, gold) if len(r) > i and r[i] == g) / len(records)
        for i in range(n_agents)
    ]
    print(f"\n[per agent, round 0] "
          + "  ".join(f"a{i} {value:.4f}" for i, value in enumerate(per_agent)))
    independent_all_wrong = 1.0
    for value in per_agent:
        independent_all_wrong *= (1 - value)
    observed_all_wrong = 1 - pool_first / len(records)
    print(f"  all three wrong: observed {observed_all_wrong:.4f}, "
          f"{independent_all_wrong:.4f} if they failed independently")
    print(f"  correlation costs {round((observed_all_wrong - independent_all_wrong) * len(records))} "
          f"items of pool")

    report = {
        "items": len(records),
        "agents": n_agents,
        "agreement_round0": sum(agreement(r) for r in first) / len(records),
        "agreement_final": sum(agreement(r) for r in final) / len(records),
        "pool_round0": pool_first / len(records),
        "pool_final": pool_final / len(records),
        "per_agent_accuracy": per_agent,
        "all_wrong_observed": observed_all_wrong,
        "all_wrong_if_independent": independent_all_wrong,
    }
    print("\nRole differentiation targets the pool line. Personas that do not move "
          "it cannot help any aggregation rule downstream, whatever they do to "
          "the prose.")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

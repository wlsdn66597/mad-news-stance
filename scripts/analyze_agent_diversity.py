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

from src.consensus import (  # noqa: E402
    LABELS,
    normalize_label,
    parse_stance,
    unique_majority,
)
from src.metrics import per_class_prf  # noqa: E402
from src.consensus_io import load_json, result_records  # noqa: E402


def records_from(path):
    """Agent-level predictions, from a debate trace or from round 0 alone.

    A `--methods majority` run has no debate mapping, but the round-0 answers
    every method shares are saved under `_shared_round0`. Round 0 is where the
    candidate pool is set, so that is enough for everything here -- and it is
    the only thing available for a persona check that deliberately skips the
    debate.
    """
    data = load_json(path)
    if isinstance(data.get("debate"), dict):
        return result_records(data)
    shared = data.get("_shared_round0")
    if not shared:
        raise SystemExit(
            f"{path} has neither a debate mapping nor a _shared_round0 block, so "
            "there are no per-agent answers to compare"
        )
    gold = {}
    for name, block in data.items():
        if name.startswith("_") or not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, dict) and "gold" in value:
                gold.setdefault(str(key), value["gold"])
    records = []
    for key, entry in shared.items():
        answers = entry.get("preds") or [parse_stance(a) for a in entry.get("raw", [])]
        records.append({
            "id": key,
            "gold": gold.get(str(key)),
            "round_predictions": [list(answers)],
        })
    return records


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
    records = [r for r in records_from(args.result) if r.get("gold") in LABELS]
    if not records:
        raise SystemExit("no items with a usable gold label")
    n_agents = max(len(r["round_predictions"][0]) for r in records)
    gold = [r["gold"] for r in records]

    first = [[normalize_label(v) for v in r["round_predictions"][0]] for r in records]
    final = [[normalize_label(v) for v in r["round_predictions"][-1]] for r in records]

    print(f"[items] {len(records)}   agents {n_agents}   "
          f"rounds {len(records[0]['round_predictions'])}")

    n_rounds = len(records[0]["round_predictions"])
    single_round = n_rounds == 1

    # One row per round. Debate is a convergence process, so agreement rises and
    # the pool falls; whether accuracy follows is the question, and the table is
    # what makes that visible instead of only the endpoints.
    print(f"\n[per round]")
    print(f"  {'round':>5} {'agree':>7} {'3:0':>6} {'2 lbl':>6} {'3 lbl':>6} "
          f"{'pool':>7} {'majority':>9} {'neut R':>7} {'changed':>8}")
    by_round = []
    previous = None
    for index in range(n_rounds):
        labels = [
            [normalize_label(v) for v in r["round_predictions"][index]] for r in records
        ]
        agree = sum(agreement(r) for r in labels) / len(labels)
        distinct = Counter(len(set(r) - {None}) for r in labels)
        pool = sum(g in set(r) for r, g in zip(labels, gold)) / len(labels)
        votes = [unique_majority(r) for r in labels]
        preds = [
            v.label if v.label else next((x for x in r if x in LABELS), None)
            for v, r in zip(votes, labels)
        ]
        acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
        neutral = per_class_prf(preds, gold, LABELS)["neutral"]["recall"]
        changed = (
            None if previous is None
            else sum(a != b for row_a, row_b in zip(labels, previous)
                     for a, b in zip(row_a, row_b)) / (len(labels) * n_agents)
        )
        print(f"  {index:>5} {agree:7.4f} {distinct.get(1, 0):6d} "
              f"{distinct.get(2, 0):6d} {distinct.get(3, 0):6d} {pool:7.4f} "
              f"{acc:9.4f} {neutral:7.4f} "
              + (f"{changed:8.4f}" if changed is not None else f"{'-':>8}"))
        by_round.append({
            "round": index, "agreement": agree, "unanimous": distinct.get(1, 0),
            "two_labels": distinct.get(2, 0), "three_labels": distinct.get(3, 0),
            "candidate_pool": pool, "majority_accuracy": acc,
            "neutral_recall": neutral, "agent_label_change_rate": changed,
        })
        previous = labels

    # the pool: what any aggregation rule could possibly reach
    pool_first = sum(g in set(r) for r, g in zip(first, gold))
    pool_final = sum(g in set(r) for r, g in zip(final, gold))
    print(f"\n[candidate pool] at least one agent names the gold label")
    print(f"  round 0     {pool_first}/{len(records)} = {pool_first / len(records):.4f}"
          f"   <- the hard ceiling for majority")
    if not single_round:
        print(f"  final round {pool_final}/{len(records)} = {pool_final / len(records):.4f}"
              f"   ({pool_final - pool_first:+d} from the exchange)")
    else:
        print("  (round 0 only in this run; no exchange to report)")

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
        "agreement_final": (
            None if single_round else sum(agreement(r) for r in final) / len(records)
        ),
        "pool_round0": pool_first / len(records),
        "pool_final": None if single_round else pool_final / len(records),
        "by_round": by_round,
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

"""Where the answer is created, lost and chosen. Offline, over saved runs.

Accuracy hides which stage failed. A debate can end more accurate because its
agents proposed the gold label more often, or because the vote picked better
among the same candidates, and those call for opposite fixes. These six numbers
split the pipeline into generation, interaction and selection:

- **candidate coverage**: at least one agent names the gold label. Round 0 is
  what generation produced; the final round is what survived. No aggregation
  rule can exceed it.
- **gold retention**: of the items whose round-0 candidates contained the gold
  label, the share still containing it at the end. What the exchange destroyed.
- **gold discovery**: of the items whose round-0 candidates did not, the share
  that acquired it. What the exchange created.
- **selection efficiency**: of the items whose final candidates contain the
  gold label, the share the run's own saved prediction gets right. How well the
  aggregation rule harvests what it is given.
- **consensus error rate**: of the items the agents finally agree on, the share
  they agree on wrongly. Consensus that is not correctness.
- **communication compliance**: of the messages an agent hands its peers, the
  share carrying anything beyond the label line. Published MAD assumes agents
  read each other's reasoning; this is whether any was sent.

    python scripts/analyze_information_flow.py results/phase2/<run>.json ...
    python scripts/analyze_information_flow.py results/phase2/*_personas*.json --json flow.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import LABELS, parse_stance  # noqa: E402

RATIONALE_CHARS = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="saved phase2 results with debate traces")
    parser.add_argument("--method", default="debate")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def carries_rationale(answer):
    """Anything an agent sent its peers other than the label line."""
    rest = "\n".join(
        line for line in str(answer).splitlines()
        if not line.strip().lower().startswith("final stance")
    )
    return len(rest.strip()) >= RATIONALE_CHARS


def rows_of(path, method):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data.get(method)
    if not isinstance(block, dict) or not block:
        return None, {}
    rows = []
    for item in block.values():
        answers = (item.get("debate_trace") or {}).get("answers_by_round")
        if not answers:
            return None, data.get("_meta", {})
        rows.append({
            "gold": item["gold"],
            "pred": item["pred"],
            "rounds": [[parse_stance(a) for a in answers_of] for answers_of in answers],
            "raw": answers,
        })
    return rows, data.get("_meta", {})


def flow(rows):
    n = len(rows)
    n_rounds = max(len(r["rounds"]) for r in rows)
    coverage = [
        sum(1 for r in rows if r["gold"] in set(r["rounds"][min(i, len(r["rounds"]) - 1)])) / n
        for i in range(n_rounds)
    ]
    had = [r for r in rows if r["gold"] in set(r["rounds"][0])]
    lacked = [r for r in rows if r["gold"] not in set(r["rounds"][0])]
    kept = [r for r in had if r["gold"] in set(r["rounds"][-1])]
    found = [r for r in lacked if r["gold"] in set(r["rounds"][-1])]
    reachable = [r for r in rows if r["gold"] in set(r["rounds"][-1])]
    unanimous = [r for r in rows if len(set(r["rounds"][-1])) == 1]
    compliance = [
        sum(1 for r in rows if i < len(r["rounds"])
            for a in r["raw"][i] if carries_rationale(a))
        / max(1, sum(len(r["raw"][i]) for r in rows if i < len(r["rounds"])))
        for i in range(n_rounds)
    ]
    return {
        "items": n,
        "rounds": n_rounds,
        "coverage_by_round": coverage,
        "coverage_round0": coverage[0],
        "coverage_final": coverage[-1],
        "gold_retention": len(kept) / max(1, len(had)),
        "gold_discovery": len(found) / max(1, len(lacked)),
        "selection_efficiency": (
            sum(1 for r in reachable if r["pred"] == r["gold"]) / max(1, len(reachable))
        ),
        "consensus_rate": len(unanimous) / n,
        "consensus_error_rate": (
            sum(1 for r in unanimous if r["pred"] != r["gold"]) / max(1, len(unanimous))
        ),
        "communication_compliance": compliance,
        "accuracy": sum(1 for r in rows if r["pred"] == r["gold"]) / n,
    }


def main():
    args = parse_args()
    report = {}
    header = (f"  {'run':44} {'acc':>6} {'cov0':>6} {'covF':>6} {'retain':>7} "
              f"{'discov':>7} {'select':>7} {'unan':>6} {'unanErr':>8} {'comply':>7}")
    print("[information flow]  coverage is the ceiling, selection is the harvest")
    print(header)
    for path in args.results:
        rows, meta = rows_of(path, args.method)
        if not rows:
            print(f"  {Path(path).stem[:44]:44} (no '{args.method}' trace)")
            continue
        stats = flow(rows)
        report[path] = {**stats, "meta": meta}
        exchanged = stats["communication_compliance"][1:]
        print(f"  {Path(path).stem[-44:]:44} {stats['accuracy']:6.4f} "
              f"{stats['coverage_round0']:6.4f} {stats['coverage_final']:6.4f} "
              f"{stats['gold_retention']:7.4f} {stats['gold_discovery']:7.4f} "
              f"{stats['selection_efficiency']:7.4f} {stats['consensus_rate']:6.4f} "
              f"{stats['consensus_error_rate']:8.4f} "
              f"{(sum(exchanged) / len(exchanged) if exchanged else 0):7.4f}")

    print("\n  cov0/covF  candidate coverage at round 0 and at the end")
    print("  retain     of items that had the gold candidate, share that kept it")
    print("  discov     of items that lacked it, share that acquired it")
    print("  select     of items whose final candidates contain gold, share predicted right")
    print("  unan/Err   final agreement, and the share of that agreement which is wrong")
    print("  comply     share of exchanged messages carrying more than the label line")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

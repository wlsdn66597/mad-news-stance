"""When does debate beat majority? Offline, over a saved phase2 run. No GPU.

The aggregate number hides the claim worth making. Debate beats majority by 17
items for EXAONE-4.0-1.2B at four rounds and loses by 10 for Qwen3-8B, so
"debate improves accuracy" is not the finding; "debate improves accuracy under
these conditions" is. This splits the paired majority-vs-debate comparison by
the conditions that plausibly govern it:

- **round-0 agreement** -- unanimous, 2:1, or 1:1:1. If the gain sits in the
  split items, debate is repairing the ensemble where the ensemble is weak,
  which is a different claim from repairing it everywhere.
- **whether the gold label was proposed at round 0.** No exchange of opinions
  can reach a label no agent ever named, so this separates "debate found the
  answer" from "debate could not have".
- **gold label** -- neutral is the hard class in this dataset.
- **article length** -- quartiles, since long articles are where a small model's
  attention runs out.

Each bucket gets its own exact McNemar. The buckets are exploratory and are not
corrected for multiplicity; treat them as directions, not as separate claims.

    python scripts/analyze_when_debate_helps.py \
      --result results/phase2/<run>.json \
      --data-path data/k-news-stance_nosegment.json
"""
import argparse
import json
import sys
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
    parser.add_argument("--result", required=True, help="saved phase2 result with a debate trace")
    parser.add_argument("--baseline-method", default="majority")
    parser.add_argument("--candidate-method", default="debate")
    parser.add_argument(
        "--candidate-result",
        help="take the candidate from a different file, e.g. the 4-round run",
    )
    parser.add_argument("--data-path", help="dataset, for the article-length split")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def predictions(path, method):
    data = load_json(path)
    if method not in data:
        raise SystemExit(
            f"{path} has no '{method}'; found "
            f"{sorted(k for k in data if not k.startswith('_'))}"
        )
    return {str(k): v["pred"] for k, v in data[method].items()}, {
        str(k): v["gold"] for k, v in data[method].items()
    }


def agreement_bucket(round0):
    vote = unique_majority(round0)
    counts = sorted(vote.counts.values(), reverse=True)
    if not counts:
        return "no valid label"
    if counts[0] == len(round0):
        return "round0 unanimous (3:0)"
    return "round0 tied (1:1:1)" if vote.tied else "round0 split (2:1)"


def quartiles(values):
    ordered = sorted(values)
    if not ordered:
        return []
    return [ordered[int(q * len(ordered))] for q in (0.25, 0.5, 0.75)]


def main():
    args = parse_args()
    before, gold = predictions(args.result, args.baseline_method)
    after, _ = predictions(args.candidate_result or args.result, args.candidate_method)

    records = {str(r["id"]): r for r in result_records(load_json(args.result), args.data_path)}
    shared = sorted(set(before) & set(after) & set(records))
    if not shared:
        raise SystemExit("the two methods share no item ids")

    lengths = {key: len(records[key].get("article") or "") for key in shared}
    cuts = quartiles([lengths[key] for key in shared if lengths[key]])

    def length_bucket(key):
        value = lengths[key]
        if not value or not cuts:
            return "article length unknown"
        for index, cut in enumerate(cuts):
            if value <= cut:
                return f"article length Q{index + 1}"
        return "article length Q4"

    def gold_proposed(key):
        round0 = records[key]["round_predictions"][0]
        return ("gold proposed at round 0"
                if gold[key] in {normalize_label(v) for v in round0}
                else "gold never proposed at round 0")

    splits = {
        "round-0 agreement": lambda key: agreement_bucket(records[key]["round_predictions"][0]),
        "candidate availability": gold_proposed,
        "gold label": lambda key: gold[key],
        "article length": length_bucket,
    }

    overall = transition_metrics(
        [before[k] for k in shared], [after[k] for k in shared], [gold[k] for k in shared]
    )
    overall_p = mcnemar_exact(
        [before[k] for k in shared], [after[k] for k in shared], [gold[k] for k in shared]
    )
    base_acc = sum(before[k] == gold[k] for k in shared) / len(shared)
    cand_acc = sum(after[k] == gold[k] for k in shared) / len(shared)

    print(f"[items] {len(shared)} shared   {args.baseline_method} {base_acc:.4f} -> "
          f"{args.candidate_method} {cand_acc:.4f}   "
          f"net {overall['net_improvement']:+d}  p={overall_p['p_value_two_sided']:.3f}")

    report = {"items": len(shared), "overall": {**overall, **overall_p}, "splits": {}}
    for title, bucket_of in splits.items():
        buckets = {}
        for key in shared:
            buckets.setdefault(bucket_of(key), []).append(key)
        print(f"\n[{title}]")
        print(f"  {'bucket':30} {'n':>5} {'base':>7} {'cand':>7} "
              f"{'w2c':>5} {'c2w':>5} {'net':>5} {'p':>7}")
        for name in sorted(buckets, key=lambda b: -len(buckets[b])):
            keys = buckets[name]
            b = [before[k] for k in keys]
            a = [after[k] for k in keys]
            g = [gold[k] for k in keys]
            t = transition_metrics(b, a, g)
            stats = mcnemar_exact(b, a, g)
            base = sum(x == y for x, y in zip(b, g)) / len(keys)
            cand = sum(x == y for x, y in zip(a, g)) / len(keys)
            print(f"  {name:30} {len(keys):5d} {base:7.4f} {cand:7.4f} "
                  f"{t['wrong_to_correct']:5d} {t['correct_to_wrong']:5d} "
                  f"{t['net_improvement']:+5d} {stats['p_value_two_sided']:7.3f}")
            report["splits"].setdefault(title, {})[name] = {
                "n": len(keys), "baseline_accuracy": base, "candidate_accuracy": cand,
                **t, **stats,
            }

    print("\nbuckets are exploratory and not corrected for multiplicity; the "
          "overall line is the only confirmatory test here")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

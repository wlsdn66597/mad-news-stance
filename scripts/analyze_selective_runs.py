"""Where the selective-advocacy judge wins and loses, and what would fix it.

Offline over saved `.items.json` runs -- no GPU. The ablations showed that
adding rationales costs accuracy overall; this asks the follow-up questions the
aggregate hides:

- **Which items?** Broken out by why the item fired: a 1:1:1 tie, a 2:1 split, or
  a round-0/final disagreement. A rationale may be worth something exactly where
  the vote carries no information (the tie) and be noise where it does.
- **Does the judge know when it is wrong?** The strict schema returns
  `evidence_sufficient`. Gating on it -- keep the vote when the judge says the
  evidence was thin -- costs nothing and is a real deployable rule.
- **Do the runs agree?** With several ablations saved, a verdict every judge
  reaches is a different object from one only one judge reaches. Accepting only
  agreed verdicts and keeping the vote otherwise is again free.
- **What is the ceiling?** Per item, the best any selection over these runs
  could do.

    python scripts/analyze_selective_runs.py results/selective_advocacy/*.items.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import mcnemar_exact  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("items", nargs="+", help="selective advocacy .items.json files")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def base_label(path):
    name = Path(path).name
    for tag in ("article_only", "no_commissioned", "no_votes", "article_first"):
        if f"abl-{tag}" in name:
            return tag
    return "full"


def labels_for(paths):
    """One readable label per file, disambiguated only where it has to be.

    The article-first ablations differ by quote tagging and by candidate order,
    not by ablation name, so several files legitimately share a base label.
    """
    bases = [base_label(path) for path in paths]
    labels = []
    for path, base in zip(paths, bases):
        if bases.count(base) == 1:
            labels.append(base)
            continue
        name = Path(path).name
        extra = []
        # the prompt-style segment follows, so stop at it rather than swallowing it
        prompt = re.search(r"_jp-([a-z_]+?)_(?:toc|structured)_", name)
        if prompt:
            extra.append(prompt.group(1))
        if "_untagged" in name:
            extra.append("untagged")
        order = re.search(r"_ord(\d+)_", name)
        if order:
            extra.append(f"ord{order.group(1)}")
        labels.append("+".join([base] + extra) if extra else base)
    return labels


def subgroup(reason):
    """Why the item fired, collapsed to the three cases that differ in kind."""
    if not reason:
        return "other"
    if "final_tie" in reason:
        return "tie (1:1:1)"
    if "split_vote" in reason:
        return "split (2:1)"
    if "round0_final_disagree" in reason:
        return "round0 flip, unanimous"
    return "other"


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["item_id"]): row for row in rows}


def accuracy(rows, pick):
    correct = sum(1 for row in rows if pick(row) == row["gold"])
    return correct, correct / max(1, len(rows))


def main():
    args = parse_args()
    labels = labels_for(args.items)
    runs = {label: load(path) for label, path in zip(labels, args.items)}
    if len(runs) != len(args.items):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise SystemExit(
            f"these inputs cannot be told apart by filename: {', '.join(duplicates)}"
        )
    first = next(iter(runs.values()))
    shared = sorted(set.intersection(*[set(rows) for rows in runs.values()]))
    all_rows = [first[key] for key in shared]
    fired = [key for key in shared if first[key]["triggered"]]

    gold = {key: first[key]["gold"] for key in shared}
    baseline = {key: first[key]["baseline_prediction"] for key in shared}
    base_correct = sum(baseline[key] == gold[key] for key in shared)

    print(f"[items] {len(shared)} shared · {len(fired)} triggered "
          f"({len(fired) / max(1, len(shared)):.1%})")
    print(f"[baseline] {first[shared[0]]['baseline_reason']}-style vote: "
          f"{base_correct}/{len(shared)} = {base_correct / len(shared):.4f}")

    report = {"items": len(shared), "triggered": len(fired), "runs": {}}

    # ---- overall, and on the triggered subset only -------------------------
    print(f"\n{'ablation':18} {'overall':>9} {'on triggered':>13} "
          f"{'w2c':>5} {'c2w':>5} {'net':>5}")
    print(f"  {'(the vote alone)':28} {base_correct / len(shared):9.4f} "
          f"{sum(baseline[k] == gold[k] for k in fired) / max(1, len(fired)):13.4f}")
    for label, rows in runs.items():
        correct, acc = accuracy([rows[k] for k in shared], lambda r: r["final_prediction"])
        fired_correct = sum(rows[k]["final_prediction"] == gold[k] for k in fired)
        w2c = sum(1 for k in fired
                  if baseline[k] != gold[k] and rows[k]["final_prediction"] == gold[k])
        c2w = sum(1 for k in fired
                  if baseline[k] == gold[k] and rows[k]["final_prediction"] != gold[k])
        print(f"  {label:28} {acc:9.4f} {fired_correct / max(1, len(fired)):13.4f} "
              f"{w2c:5d} {c2w:5d} {w2c - c2w:+5d}")
        report["runs"][label] = {"overall_accuracy": acc, "wrong_to_correct": w2c,
                                 "correct_to_wrong": c2w}

    # ---- by why the item fired ---------------------------------------------
    groups = {}
    for key in fired:
        groups.setdefault(subgroup(first[key]["trigger_reason"]), []).append(key)
    print(f"\n[by trigger reason]  vote -> each ablation, accuracy on that subgroup")
    header = "  ".join(f"{label:>22}" for label in runs)
    print(f"  {'subgroup':24} {'n':>4} {'vote':>6}  {header}")
    for name, keys in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        vote = sum(baseline[k] == gold[k] for k in keys) / len(keys)
        cells = "  ".join(
            f"{sum(rows[k]['final_prediction'] == gold[k] for k in keys) / len(keys):22.4f}"
            for rows in runs.values()
        )
        print(f"  {name:24} {len(keys):4d} {vote:6.4f}  {cells}")
        report.setdefault("subgroups", {})[name] = {"n": len(keys), "vote_accuracy": vote}

    # ---- does the judge know when its evidence was thin? -------------------
    print(f"\n[evidence_sufficient gate]  keep the vote when the judge says the "
          f"evidence was thin")
    for label, rows in runs.items():
        def gated(row):
            parsed = row.get("judge_parsed_output") or {}
            if row["triggered"] and parsed.get("evidence_sufficient") is False:
                return row["baseline_prediction"]
            return row["final_prediction"]
        correct, acc = accuracy([rows[k] for k in shared], gated)
        thin = sum(1 for k in fired
                   if (rows[k].get("judge_parsed_output") or {}).get(
                       "evidence_sufficient") is False)
        plain = sum(rows[k]["final_prediction"] == gold[k] for k in shared)
        stats = mcnemar_exact(
            [rows[k]["final_prediction"] for k in shared],
            [gated(rows[k]) for k in shared],
            [gold[k] for k in shared],
        )
        print(f"  {label:28} thin={thin:4d}  {plain / len(shared):.4f} -> {acc:.4f}  "
              f"({correct - plain:+d}, p={stats['p_value_two_sided']:.3f})")
        report["runs"][label]["evidence_gate"] = {"thin": thin, "accuracy": acc}

    # ---- agreement between the runs ----------------------------------------
    if len(runs) > 1:
        print(f"\n[agreement gate]  accept a verdict only when at least k of the "
              f"{len(runs)} judges agree, else keep the vote")
        for threshold in range(2, len(runs) + 1):
            picks = {}
            accepted = 0
            for key in shared:
                if not first[key]["triggered"]:
                    picks[key] = baseline[key]
                    continue
                votes = Counter(rows[key]["final_prediction"] for rows in runs.values())
                top, count = votes.most_common(1)[0]
                tied = sum(1 for _, value in votes.items() if value == count) > 1
                if count >= threshold and not tied:
                    picks[key] = top
                    accepted += 1
                else:
                    picks[key] = baseline[key]
            correct = sum(picks[key] == gold[key] for key in shared)
            print(f"  k>={threshold}: accepted {accepted:4d}/{len(fired)}  "
                  f"{correct}/{len(shared)} = {correct / len(shared):.4f}")
            report.setdefault("agreement_gate", {})[f"k>={threshold}"] = {
                "accepted": accepted, "accuracy": correct / len(shared)
            }

    # ---- ceiling over the saved runs ---------------------------------------
    per_item_best = sum(
        1 for key in shared
        if any(rows[key]["final_prediction"] == gold[key] for rows in runs.values())
    )
    print(f"\n[ceiling] some saved run is right on {per_item_best}/{len(shared)} = "
          f"{per_item_best / len(shared):.4f}   <- perfect per-item choice among "
          f"{len(runs)} runs; not reachable, it needs gold")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

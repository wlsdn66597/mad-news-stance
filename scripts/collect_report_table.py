"""Assemble the report's tables from repeated runs. Offline. No GPU.

Every condition is read from whatever `run_report_repeats.sh` produced, scored
the same way, and reported as mean +/- sd over seeds. Accuracy alone is not
enough here: the classes are close to balanced but the models are not, so
macro-F1 and neutral recall are carried alongside it in every row.

Seed variance on this task is 11 to 16 items out of 1001, which is larger than
several of the differences being reported, so a per-seed column is printed too
-- a mean that hides one deviant seed is worth seeing.

    python scripts/collect_report_table.py --seeds 6000 6001 6002
    python scripts/collect_report_table.py --seeds 6000 6001 --markdown report.md
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import LABELS  # noqa: E402
from src.metrics import macro_f1, per_class_prf  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--model", default="exaone")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", default="1001")
    parser.add_argument("--profile", default="stance_minimal_en")
    parser.add_argument("--data-seed", default="0")
    parser.add_argument("--markdown", help="also write the tables to this file")
    return parser.parse_args()


def score(pred, gold):
    correct = sum(p == g for p, g in zip(pred, gold))
    recall = per_class_prf(pred, gold, LABELS)
    return {
        "n": len(gold),
        "correct": correct,
        "accuracy": correct / max(1, len(gold)),
        "macro_f1": macro_f1(pred, gold, LABELS),
        "neutral_recall": recall["neutral"]["recall"],
    }


def from_phase2(path, method):
    """A phase2 method, scored from its saved per-item predictions."""
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data.get(method)
    if not isinstance(block, dict) or not block:
        return None
    pred = [v["pred"] for v in block.values()]
    gold = [v["gold"] for v in block.values()]
    return score(pred, gold)


def from_items(path, field="final_prediction"):
    """A judge run, scored from its .items.json so every row uses one scorer."""
    if not Path(path).exists():
        return None
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    pred = [row[field] for row in rows]
    gold = [row["gold"] for row in rows]
    result = score(pred, gold)
    baseline = [row.get("baseline_prediction") for row in rows]
    fired = [i for i, row in enumerate(rows) if row.get("triggered")]
    if fired and all(b is not None for b in baseline):
        result["triggered"] = len(fired)
        result["trigger_ratio"] = len(fired) / len(rows)
        result["subset_before"] = sum(baseline[i] == gold[i] for i in fired) / len(fired)
        result["subset_after"] = sum(pred[i] == gold[i] for i in fired) / len(fired)
        result["w2c"] = sum(1 for i in fired if baseline[i] != gold[i] and pred[i] == gold[i])
        result["c2w"] = sum(1 for i in fired if baseline[i] == gold[i] and pred[i] != gold[i])
        result["net"] = result["w2c"] - result["c2w"]
    return result


def find_one(pattern):
    matches = sorted(Path(".").glob(pattern))
    return matches[0] if matches else None


def conditions(args, seed):
    """Every row of the report, for one seed, as name -> metrics or None."""
    base = (f"results/phase2/stance_{args.model}_{args.split}_n{args.n}_"
            f"{args.profile}_d{args.data_seed}_s{seed}_a3")
    rows = {
        "single": from_phase2(f"{base}_r2.json", "single"),
        "majority k=3": from_phase2(f"{base}_r2.json", "majority"),
        "debate 2R": from_phase2(f"{base}_r2.json", "debate"),
        "debate 4R": from_phase2(f"{base}_r4.json", "debate"),
        "persona majority": from_phase2(f"{base}_r2_personas.json", "majority"),
        "persona debate 2R": from_phase2(f"{base}_r2_personas.json", "debate"),
        "persona debate 4R": from_phase2(f"{base}_r4_personas.json", "debate"),
        # the control and the two protocols: same rounds, same calls, same
        # personas, so these rows are read against "persona debate 4R" alone
        "persona 4R self-refine": from_phase2(
            f"{base}_r4_personas_selfrefine.json", "debate"),
        "persona 4R evidence-gated": from_phase2(
            f"{base}_r4_personas_evidence_gated.json", "debate"),
        "persona 4R round-specific": from_phase2(
            f"{base}_r4_personas_round_specific.json", "debate"),
        "persona 4R reasoned-exchange": from_phase2(
            f"{base}_r4_personas_reasoned_exchange.json", "debate"),
        "persona 4R reasoned-exchange v2": from_phase2(
            f"{base}_r4_personas_reasoned_exchange_v2.json", "debate"),
        "persona 4R reasoned-exchange long": from_phase2(
            f"{base}_r4_personas_reasoned_exchange_long.json", "debate"),
        "persona 4R evidence-gated v3": from_phase2(
            f"{base}_r4_personas_evidence_gated_v3.json", "debate"),
        "persona 4R round-specific v3": from_phase2(
            f"{base}_r4_personas_round_specific_v3.json", "debate"),
        "persona 4R evidence-gated v2": from_phase2(
            f"{base}_r4_personas_evidence_gated_v2.json", "debate"),
        "persona 4R round-specific v2": from_phase2(
            f"{base}_r4_personas_round_specific_v2.json", "debate"),
    }
    # advocacy: one agent per label, judge on the same backbone
    adv = find_one(f"results/advocacy/advocacy_{args.model}_{args.split}_n{args.n}_"
                   f"{args.profile}_d{args.data_seed}_s{seed}_toc_r2_*.items.json")
    rows["advocacy + judge"] = from_items(adv, "pred") if adv else None
    # selective judge on the persona runs
    for rounds in (2, 4):
        judged = find_one(
            f"results/selective_judge_report/*_s{seed}_a3_r{rounds}_personas_*"
            f"trigger-non_unanimous_*.items.json"
        )
        rows[f"persona {rounds}R + judge"] = from_items(judged) if judged else None
    return rows


def render(name, per_seed, keys):
    values = [row for row in per_seed if row]
    if not values:
        return None
    out = {"condition": name, "seeds": len(values)}
    for key in keys:
        got = [row[key] for row in values if key in row]
        if not got:
            continue
        out[key] = statistics.mean(got)
        out[f"{key}_sd"] = statistics.pstdev(got) if len(got) > 1 else 0.0
    out["per_seed_accuracy"] = [row["accuracy"] for row in values]
    return out


def main():
    args = parse_args()
    collected = {}
    for seed in args.seeds:
        for name, row in conditions(args, seed).items():
            collected.setdefault(name, []).append(row)

    order = ["single", "majority k=3", "debate 2R", "debate 4R", "advocacy + judge",
             "persona majority", "persona debate 2R", "persona debate 4R",
             "persona 4R self-refine", "persona 4R evidence-gated",
             "persona 4R round-specific", "persona 4R reasoned-exchange",
             "persona 4R reasoned-exchange v2",
             "persona 4R reasoned-exchange long",
             "persona 4R evidence-gated v3", "persona 4R round-specific v3",
             "persona 4R evidence-gated v2",
             "persona 4R round-specific v2",
             "persona 2R + judge", "persona 4R + judge"]
    main_keys = ["accuracy", "macro_f1", "neutral_recall"]
    lines = [f"seeds: {', '.join(args.seeds)}", ""]
    lines += ["| condition | seeds | accuracy | macro-F1 | neutral recall | per-seed acc |",
              "|---|---:|---:|---:|---:|---|"]
    for name in order:
        row = render(name, collected.get(name, []), main_keys)
        if not row:
            lines.append(f"| {name} | 0 | - | - | - | not run |")
            continue
        lines.append(
            f"| {name} | {row['seeds']} | "
            f"{row['accuracy']:.4f} +/- {row['accuracy_sd']:.4f} | "
            f"{row['macro_f1']:.4f} +/- {row['macro_f1_sd']:.4f} | "
            f"{row['neutral_recall']:.4f} +/- {row['neutral_recall_sd']:.4f} | "
            + ", ".join(f"{v:.4f}" for v in row["per_seed_accuracy"]) + " |"
        )

    judge_keys = ["trigger_ratio", "subset_before", "subset_after", "w2c", "c2w", "net"]
    lines += ["", "**Judge trigger subset**", "",
              "| condition | trigger | before | after | W2C | C2W | net |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("persona 2R + judge", "persona 4R + judge"):
        row = render(name, collected.get(name, []), judge_keys)
        if not row or "net" not in row:
            lines.append(f"| {name} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {row['trigger_ratio']:.1%} | {row['subset_before']:.4f} | "
            f"{row['subset_after']:.4f} | {row['w2c']:.1f} | {row['c2w']:.1f} | "
            f"{row['net']:+.1f} |"
        )

    lines += ["", "Mean +/- population sd over seeds. Seed variance on this task is "
              "11-16 items in 1001, so read the per-seed column before trusting a "
              "difference smaller than that.",
              "", "Every debate agent is EXAONE-4.0-1.2B. The stage-5 judge is the "
              "only place another model appears; say which one in the caption."]

    text = "\n".join(lines)
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text + "\n", encoding="utf-8")
        print(f"\n[saved] {args.markdown}")


if __name__ == "__main__":
    main()

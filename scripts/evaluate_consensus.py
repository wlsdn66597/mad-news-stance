"""Evaluate offline aggregation rules on saved Phase 2 debate trajectories."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import (  # noqa: E402
    FREE_MAD_CITATION,
    FREE_MAD_DEFAULT_WEIGHTS,
    aggregate,
    bootstrap_accuracy_delta,
    boundary_transition_counts,
    classification_metrics,
    judge_trigger,
    mcnemar_exact,
    transition_metrics,
    unique_majority,
)
from src.consensus_io import load_consensus_records  # noqa: E402


METHODS = [
    "current_final_majority",
    "round0_fallback",
    "cross_round_pooled_vote",
    "free_mad_exact",
    "exploratory_neutral_boundary_fallback",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result")
    parser.add_argument("--transition-file")
    parser.add_argument("--data-path")
    parser.add_argument("--aggregation-method", choices=METHODS, required=True)
    parser.add_argument("--free-mad-weights", default="20,25,30,20")
    parser.add_argument("--fallback-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir", default="results/consensus")
    return parser.parse_args()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_report(summary):
    selected = summary["selected_metrics"]
    baseline = summary["baseline_metrics"]
    transitions = summary["transitions_from_current"]
    lines = [
        "# Offline consensus comparison",
        "",
        f"- Method: `{summary['aggregation_method']}`",
        f"- Items: {selected['items']}",
        f"- Current final majority: accuracy {baseline['accuracy']:.4f}, macro-F1 {baseline['macro_f1']:.4f}",
        f"- Selected method: accuracy {selected['accuracy']:.4f}, macro-F1 {selected['macro_f1']:.4f}",
        f"- Wrong-to-correct: {transitions['wrong_to_correct']}",
        f"- Correct-to-wrong: {transitions['correct_to_wrong']}",
        f"- Net improvement: {transitions['net_improvement']:+d}",
        "",
        "## Trigger diagnostics",
        "",
        f"- Instability count: {summary['instability']['count']} ({summary['instability']['ratio']:.2%})",
        f"- Final 2-1: {summary['final_patterns']['two_to_one']}",
        f"- Final 1-1-1/invalid tie: {summary['final_patterns']['tie']}",
    ]
    if summary["aggregation_method"] == "free_mad_exact":
        lines += [
            "",
            "## Free-MAD provenance",
            "",
            f"- Exact Algorithm 1 trajectory score with weights `{summary['free_mad_weights']}`.",
            f"- Source: {FREE_MAD_CITATION}",
            "- Paper-random tie selection is reproduced with a deterministic per-item seed.",
            "- This is the exact scoring decision mechanism applied offline, not a rerun of the full Free-MAD debate protocol.",
        ]
    if summary.get("transition_file_validation"):
        lines += ["", "## Transition-file validation", "", "```json", json.dumps(summary["transition_file_validation"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    weights = tuple(float(value) for value in args.free_mad_weights.split(","))
    records, source_meta, transition_summary = load_consensus_records(
        args.input_result, args.transition_file, args.data_path
    )
    rows = []
    for record in records:
        rounds = record["round_predictions"]
        item_id = record["id"]
        baseline = aggregate("current_final_majority", rounds, item_id, args.fallback_seed)
        selected = aggregate(
            args.aggregation_method,
            rounds,
            item_id,
            args.fallback_seed,
            free_mad_weights=weights,
        )
        triggered, trigger_reason = judge_trigger(rounds, "instability")
        row = {
            "item_id": item_id,
            "gold": record.get("gold"),
            "round0_agent_preds": rounds[0],
            "final_agent_preds": rounds[-1],
            "round0_majority": unique_majority(rounds[0]).label,
            "final_majority": unique_majority(rounds[-1]).label,
            "baseline_prediction": baseline["pred"],
            "selected_prediction": selected["pred"],
            "aggregation_reason": selected["reason"],
            "aggregation_scores": selected.get("scores"),
            "instability_triggered": triggered,
            "instability_reason": trigger_reason,
            "baseline_correct": baseline["pred"] == record.get("gold"),
            "selected_correct": selected["pred"] == record.get("gold"),
        }
        rows.append(row)

    baseline_preds = [row["baseline_prediction"] for row in rows]
    selected_preds = [row["selected_prediction"] for row in rows]
    golds = [row["gold"] for row in rows]
    triggered_rows = [row for row in rows if row["instability_triggered"]]
    final_patterns = {"unanimous": 0, "two_to_one": 0, "tie": 0}
    for row in rows:
        counts = unique_majority(row["final_agent_preds"]).counts
        maximum = max(counts.values(), default=0)
        if maximum == len(row["final_agent_preds"]):
            final_patterns["unanimous"] += 1
        elif maximum == 2:
            final_patterns["two_to_one"] += 1
        else:
            final_patterns["tie"] += 1
    summary = {
        "aggregation_method": args.aggregation_method,
        "input_result": args.input_result,
        "transition_file": args.transition_file,
        "source_meta": source_meta,
        "free_mad_weights": weights,
        "fallback_seed": args.fallback_seed,
        "baseline_metrics": classification_metrics(baseline_preds, golds),
        "selected_metrics": classification_metrics(selected_preds, golds),
        "transitions_from_current": transition_metrics(baseline_preds, selected_preds, golds),
        "boundary_transitions": boundary_transition_counts(baseline_preds, selected_preds),
        "mcnemar_exact": mcnemar_exact(baseline_preds, selected_preds, golds),
        "bootstrap_accuracy_delta": bootstrap_accuracy_delta(
            baseline_preds, selected_preds, golds, args.fallback_seed, args.bootstrap_samples
        ),
        "instability": {
            "count": len(triggered_rows),
            "ratio": len(triggered_rows) / max(1, len(rows)),
            "baseline_accuracy": sum(row["baseline_correct"] for row in triggered_rows) / max(1, len(triggered_rows)),
            "selected_accuracy": sum(row["selected_correct"] for row in triggered_rows) / max(1, len(triggered_rows)),
        },
        "final_patterns": final_patterns,
        "transition_file_validation": transition_summary,
        "free_mad_reference": FREE_MAD_CITATION if args.aggregation_method == "free_mad_exact" else None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(args.input_result or args.transition_file).stem
    prefix = output_dir / f"{source_stem}_{args.aggregation_method}"
    save_json(prefix.with_suffix(".items.json"), rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    prefix.with_suffix(".md").write_text(markdown_report(summary), encoding="utf-8")
    for suffix, delimiter in ((".csv", ","), (".tsv", "\t")):
        with open(prefix.with_suffix(suffix), "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "item_id", "gold", "round0_majority", "final_majority",
                    "baseline_prediction", "selected_prediction", "aggregation_reason",
                    "instability_triggered", "baseline_correct", "selected_correct",
                ],
                delimiter=delimiter,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {prefix}.*")


if __name__ == "__main__":
    main()

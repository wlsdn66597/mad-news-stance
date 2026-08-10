"""Run a deterministic judge only on unstable saved debate trajectories."""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import (  # noqa: E402
    aggregate,
    bootstrap_accuracy_delta,
    boundary_transition_counts,
    classification_metrics,
    judge_trigger,
    mcnemar_exact,
    stable_seed,
    transition_metrics,
    unique_majority,
)
from src.consensus_io import load_consensus_records  # noqa: E402
from src.llm import load_model  # noqa: E402
from src.selective_judge import (  # noqa: E402
    JUDGE_ROUNDS,
    JUDGE_SYSTEM_PROMPT,
    run_judge,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--transition-file")
    parser.add_argument("--data-path")
    parser.add_argument("--aggregation-method", default="current_final_majority")
    parser.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-trigger", choices=["tie_only", "instability", "non_unanimous", "all"], default="instability")
    parser.add_argument("--judge-input-mode", choices=["article_only", "debate_trace"], default="debate_trace")
    parser.add_argument(
        "--judge-rounds", choices=list(JUDGE_ROUNDS), default="all",
        help="which rounds of the trace the judge reads. 'all' is every "
             "agent at every round -- twelve analyses at four rounds, mostly "
             "converged duplicates that argue with each other. 'last' is the "
             "settled position only, 'first' the independent readings.")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-order-seed", type=int, default=0)
    parser.add_argument("--judge-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-max-retries", type=int, default=1)
    parser.add_argument("--judge-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--mem-fraction", type=float)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="results/selective_judge")
    return parser.parse_args()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_config(meta):
    path = meta.get("config")
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


def safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def report_markdown(summary):
    overall = summary["judge_metrics"]
    subset = summary["trigger_subset"]
    return "\n".join(
        [
            "# Selective judge report",
            "",
            f"- Trigger: `{summary['judge_config']['trigger']}`",
            f"- Input mode: `{summary['judge_config']['input_mode']}`",
            f"- Judge model: `{summary['judge_config']['model_id']}`",
            f"- Triggered: {subset['triggered_total']}/{summary['items']} ({subset['trigger_ratio']:.2%})",
            f"- Overall accuracy: {overall['accuracy']:.4f}",
            f"- Overall macro-F1: {overall['macro_f1']:.4f}",
            f"- Triggered-subset accuracy: {subset['judge_accuracy']:.4f}",
            f"- Wrong-to-correct: {subset['transitions']['wrong_to_correct']}",
            f"- Correct-to-wrong: {subset['transitions']['correct_to_wrong']}",
            f"- Net improvement: {subset['transitions']['net_improvement']:+d}",
            f"- Parse fallbacks: {summary['failures']['fallback_count']}",
            "",
            "Gold labels were used only after inference for evaluation.",
        ]
    ) + "\n"


def main():
    args = parse_args()
    records, meta, transition_summary = load_consensus_records(
        args.input_result, args.transition_file, args.data_path
    )
    config = source_config(meta)
    judge_model_id = args.judge_model or meta.get("model_id")
    if not judge_model_id:
        raise ValueError("judge model is unknown; pass --judge-model")
    load_in_4bit = (
        args.judge_load_in_4bit
        if args.judge_load_in_4bit is not None
        else bool(meta.get("load_in_4bit", config.get("load_in_4bit", False)))
    )
    if args.judge_enabled:
        model, tokenizer = load_model(
            judge_model_id,
            load_in_4bit=load_in_4bit,
            gpu_index=args.gpu_index,
            mem_fraction=args.mem_fraction,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = tokenizer = None

    profile = meta.get("prompt_profile", "unknown")
    source_stem = Path(args.input_result).stem
    prefix_name = (
        f"{source_stem}_judge-{safe_name(judge_model_id)}_trigger-{args.judge_trigger}_"
        f"input-{args.judge_input_mode}_"
        f"{'' if args.judge_rounds == 'all' else f'rounds-{args.judge_rounds}_'}"
        f"seed-{args.judge_order_seed}_"
        f"temp-{args.judge_temperature:g}_tok-{args.judge_max_new_tokens}_retry-{args.judge_max_retries}_"
        f"q4-{int(load_in_4bit)}_agg-{safe_name(args.aggregation_method)}_"
        f"a{meta.get('n_agents', 'x')}_r{meta.get('n_rounds', 'x')}"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / prefix_name
    items_path = prefix.with_suffix(".items.json")
    existing = {}
    if items_path.exists():
        existing = {str(row["item_id"]): row for row in json.loads(items_path.read_text(encoding="utf-8"))}

    output_rows = []
    failures = []
    for index, record in enumerate(records, start=1):
        item_id = record["id"]
        key = str(item_id)
        rounds = record["round_predictions"]
        baseline = aggregate(args.aggregation_method, rounds, item_id, args.judge_order_seed)
        triggered, trigger_reason = judge_trigger(rounds, args.judge_trigger)
        if key in existing:
            output_rows.append(existing[key])
            continue
        row = {
            "item_id": item_id,
            "gold": record.get("gold"),
            "round0_agent_preds": rounds[0],
            "final_agent_preds": rounds[-1],
            "round0_majority": unique_majority(rounds[0]).label,
            "final_majority": unique_majority(rounds[-1]).label,
            "trigger_mode": args.judge_trigger,
            "trigger_reason": trigger_reason,
            "triggered": triggered,
            "judge_input_mode": args.judge_input_mode,
            "judge_model": judge_model_id,
            "judge_quantization": "4bit-nf4" if load_in_4bit else "none",
            "judge_seed": stable_seed(args.judge_order_seed, item_id, "judge_generation"),
            "judge_decoding": {
                "temperature": args.judge_temperature,
                "do_sample": bool(args.judge_temperature > 0),
                "max_new_tokens": args.judge_max_new_tokens,
                "max_retries": args.judge_max_retries,
            },
            "judge_prediction": None,
            "judge_raw_output": None,
            "judge_parsed_output": None,
            "candidate_order": [],
            "fallback_used": False,
            "fallback_reason": None,
            "baseline_prediction": baseline["pred"],
            "baseline_correct": baseline["pred"] == record.get("gold"),
            "latency_seconds": 0.0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }
        final_prediction = baseline["pred"]
        if triggered and args.judge_enabled:
            for field in ("issue", "headline", "article"):
                if not record.get(field):
                    raise ValueError(
                        f"item {item_id} lacks {field}; pass --data-path for the original dataset"
                    )
            set_seed(int(row["judge_seed"]) % (2**32))
            result = run_judge(
                model,
                tokenizer,
                {**record, "id": item_id},
                record["raw_rounds"],
                input_mode=args.judge_input_mode,
                order_seed=args.judge_order_seed,
                max_new_tokens=args.judge_max_new_tokens,
                max_retries=args.judge_max_retries,
                temperature=args.judge_temperature,
                enable_thinking=False,
                judge_rounds=args.judge_rounds,
            )
            row.update(
                {
                    "judge_prediction": result["prediction"],
                    "judge_raw_output": result["raw_output"],
                    "judge_parsed_output": result["parsed_output"],
                    "judge_attempts": result["attempts"],
                    "candidate_order": result["candidate_order"],
                    "latency_seconds": result["latency_seconds"],
                    "token_usage": result["token_usage"],
                }
            )
            if result["prediction"]:
                final_prediction = result["prediction"]
            else:
                fallback = aggregate("round0_fallback", rounds, item_id, args.judge_order_seed)
                final_prediction = fallback["pred"]
                row["fallback_used"] = True
                row["fallback_reason"] = fallback["reason"]
                failures.append({"item_id": item_id, "attempts": result["attempts"], "fallback": fallback})
        row["final_prediction"] = final_prediction
        row["judge_correct"] = final_prediction == record.get("gold")
        output_rows.append(row)
        save_json(items_path, output_rows)
        if triggered:
            print(f"[judge] {index}/{len(records)} item={item_id} pred={final_prediction}", flush=True)

    baseline_preds = [row["baseline_prediction"] for row in output_rows]
    judge_preds = [row["final_prediction"] for row in output_rows]
    golds = [row["gold"] for row in output_rows]
    triggered_rows = [row for row in output_rows if row["triggered"]]
    triggered_baseline = [row["baseline_prediction"] for row in triggered_rows]
    triggered_judge = [row["final_prediction"] for row in triggered_rows]
    triggered_golds = [row["gold"] for row in triggered_rows]
    subset_transitions = transition_metrics(triggered_baseline, triggered_judge, triggered_golds)
    summary = {
        "items": len(output_rows),
        "source_meta": meta,
        "transition_file_validation": transition_summary,
        "judge_config": {
            "enabled": args.judge_enabled,
            "model_id": judge_model_id,
            "quantization": "4bit-nf4" if load_in_4bit else "none",
            "trigger": args.judge_trigger,
            "input_mode": args.judge_input_mode,
            "judge_rounds": args.judge_rounds,
            "temperature": args.judge_temperature,
            "do_sample": bool(args.judge_temperature > 0),
            "order_seed": args.judge_order_seed,
            "max_new_tokens": args.judge_max_new_tokens,
            "max_retries": args.judge_max_retries,
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "gold_available_to_judge": False,
        },
        "baseline_metrics": classification_metrics(baseline_preds, golds),
        "judge_metrics": classification_metrics(judge_preds, golds),
        "overall_transitions": transition_metrics(baseline_preds, judge_preds, golds),
        "boundary_transitions": boundary_transition_counts(baseline_preds, judge_preds),
        "mcnemar_exact": mcnemar_exact(baseline_preds, judge_preds, golds),
        "bootstrap_accuracy_delta": bootstrap_accuracy_delta(
            baseline_preds, judge_preds, golds, args.judge_order_seed
        ),
        "trigger_subset": {
            "triggered_total": len(triggered_rows),
            "trigger_ratio": len(triggered_rows) / max(1, len(output_rows)),
            "baseline_accuracy": sum(row["baseline_correct"] for row in triggered_rows) / max(1, len(triggered_rows)),
            "judge_accuracy": sum(row["judge_correct"] for row in triggered_rows) / max(1, len(triggered_rows)),
            "transitions": subset_transitions,
            "threshold_to_beat_round0_fallback": 48 if len(triggered_rows) == 104 else None,
        },
        "cost": {
            "judge_calls_including_retries": sum(len(row.get("judge_attempts", [])) for row in output_rows),
            "latency_seconds": sum(row["latency_seconds"] for row in output_rows),
            "input_tokens": sum(row["token_usage"]["input_tokens"] for row in output_rows),
            "output_tokens": sum(row["token_usage"]["output_tokens"] for row in output_rows),
        },
        "failures": {"fallback_count": sum(row["fallback_used"] for row in output_rows), "items": failures},
    }
    save_json(items_path, output_rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    save_json(prefix.with_suffix(".failures.json"), failures)
    prefix.with_suffix(".md").write_text(report_markdown(summary), encoding="utf-8")
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "item_id", "gold", "round0_majority", "final_majority", "triggered",
            "trigger_reason", "baseline_prediction", "judge_prediction", "final_prediction",
            "fallback_used", "baseline_correct", "judge_correct", "latency_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {prefix}.*")


if __name__ == "__main__":
    main()

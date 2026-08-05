"""Selective consensus round in front of the existing article-only judge.

Unstable saved items are first re-examined once by the original debate agents.
A unanimous revised label is accepted directly; everything else goes to the
existing article-only judge, and a failed judge keeps the existing fallback.

Three comparable conditions on the same instability subset:

    --consensus-prompt-mode none                       # direct_article_judge
    --consensus-prompt-mode plain_extra_round          # extra_round_then_judge
    --consensus-prompt-mode independent_reconsideration  # reconsideration_then_judge
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import stable_seed  # noqa: E402
from src.consensus_io import load_consensus_records  # noqa: E402
from src.consensus_round import (  # noqa: E402
    CONSENSUS_PROMPT_MODES,
    NO_CONSENSUS_MODE,
    RECONSIDERATION_INSTRUCTION,
    run_consensus_round,
)
from src.llm import load_model  # noqa: E402
from src.selective_consensus import (  # noqa: E402
    JUDGE_CACHE_CONFIG_FIELDS,
    build_summary,
    direct_judge_predictions,
    load_existing_rows,
    load_judge_cache,
    output_prefix,
    process_records,
)
from src.selective_judge import JUDGE_SYSTEM_PROMPT, run_judge  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result", required=True)
    parser.add_argument("--transition-file")
    parser.add_argument("--data-path")
    parser.add_argument("--aggregation-method", default="current_final_majority")
    parser.add_argument("--judge-trigger", choices=["tie_only", "instability", "non_unanimous", "all"], default="instability")
    parser.add_argument(
        "--consensus-prompt-mode",
        choices=[NO_CONSENSUS_MODE, *CONSENSUS_PROMPT_MODES],
        default="plain_extra_round",
    )
    parser.add_argument("--consensus-model")
    parser.add_argument("--consensus-temperature", type=float)
    parser.add_argument("--consensus-max-new-tokens", type=int)
    parser.add_argument("--consensus-seed", type=int)
    parser.add_argument("--consensus-order-seed", type=int, default=9001)
    parser.add_argument("--consensus-debate-template")
    parser.add_argument("--consensus-enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--consensus-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-input-mode", choices=["article_only", "debate_trace"], default="article_only")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-order-seed", type=int, default=7001)
    parser.add_argument("--judge-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-max-retries", type=int, default=1)
    parser.add_argument("--judge-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--judge-result", help="previous selective-judge .items.json to reuse")
    parser.add_argument("--judge-cache-require-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--mem-fraction", type=float)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, help="process only the first N items (smoke runs)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/selective_consensus")
    return parser.parse_args()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_config(meta):
    path = meta.get("config")
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {}


class ModelPool:
    """Load each (model, quantization) pair once; reuse it for both stages."""

    def __init__(self, gpu_index, mem_fraction, trust_remote_code):
        self.gpu_index = gpu_index
        self.mem_fraction = mem_fraction
        self.trust_remote_code = trust_remote_code
        self._loaded = {}

    def get(self, model_id, load_in_4bit):
        key = (model_id, bool(load_in_4bit))
        if key not in self._loaded:
            print(f"[load] {model_id} load_in_4bit={bool(load_in_4bit)}", flush=True)
            self._loaded[key] = load_model(
                model_id,
                load_in_4bit=bool(load_in_4bit),
                gpu_index=self.gpu_index,
                mem_fraction=self.mem_fraction,
                trust_remote_code=self.trust_remote_code,
            )
        return self._loaded[key]


def report_markdown(summary):
    stage = summary["consensus_stage"]
    overall = summary["final_metrics"]
    subset = summary["trigger_subset"]
    lines = [
        "# Selective consensus report",
        "",
        f"- Consensus prompt mode: `{summary['config']['consensus_prompt_mode']}`",
        f"- Trigger: `{summary['config']['judge_trigger']}`",
        f"- Consensus model: `{summary['config']['consensus_model']}` "
        f"(temperature {summary['config']['consensus_temperature']}, "
        f"max_new_tokens {summary['config']['consensus_max_new_tokens']})",
        f"- Judge: `{summary['config']['judge_model']}` "
        f"(input `{summary['config']['judge_input_mode']}`, "
        f"temperature {summary['config']['judge_temperature']})",
        "",
        f"- Triggered: {stage['triggered_total']}/{summary['items']} "
        f"({subset['trigger_ratio']:.2%})",
        f"- Consensus reached: {stage['consensus_reached']}/{stage['consensus_attempted']} "
        f"({stage['consensus_reached_ratio']:.2%}), accuracy {stage['consensus_accuracy']:.4f}",
        f"- Wrong unanimous consensus: {stage['wrong_unanimous_consensus']}",
        f"- Sent to judge: {stage['judge_required']} ({stage['judge_required_ratio']:.2%})",
        f"- Judge sources: {stage['judge_source_counts']}",
        f"- Fallbacks used: {stage['fallback_used']}",
        "",
        f"- Overall accuracy: {overall['accuracy']:.4f} "
        f"(baseline {summary['baseline_metrics']['accuracy']:.4f})",
        f"- Overall macro-F1: {overall['macro_f1']:.4f} "
        f"(baseline {summary['baseline_metrics']['macro_f1']:.4f})",
        f"- Wrong-to-correct: {summary['overall_transitions']['wrong_to_correct']}",
        f"- Correct-to-wrong: {summary['overall_transitions']['correct_to_wrong']}",
        f"- Net improvement: {summary['overall_transitions']['net_improvement']:+d}",
    ]
    comparison = summary.get("direct_article_judge_comparison")
    if comparison:
        lines += [
            "",
            "## vs direct article-only judge",
            f"- Covered items: {comparison['covered_items']} "
            f"(complete: {comparison['complete_coverage']})",
            f"- Accuracy delta: {comparison['accuracy_delta']:+.4f}",
            f"- Macro-F1 delta: {comparison['macro_f1_delta']:+.4f}",
            f"- Wrong-to-correct vs direct: {comparison['transitions_vs_direct']['wrong_to_correct']}",
            f"- Correct-to-wrong vs direct: {comparison['transitions_vs_direct']['correct_to_wrong']}",
        ]
    lines += [
        "",
        "Gold labels were used only after inference, for evaluation.",
        "This is `selective independent reconsideration with unanimity-based acceptance`,",
        "not a reproduction of a published consensus protocol.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    records, meta, transition_summary = load_consensus_records(
        args.input_result, args.transition_file, args.data_path
    )
    if args.limit:
        records = records[: args.limit]
    config = source_config(meta)

    consensus_model_id = args.consensus_model or meta.get("model_id")
    judge_model_id = args.judge_model or meta.get("model_id")
    consensus_temperature = (
        args.consensus_temperature
        if args.consensus_temperature is not None
        else meta.get("temperature", config.get("sampling", {}).get("temperature", 1.0))
    )
    consensus_max_new_tokens = (
        args.consensus_max_new_tokens
        or meta.get("max_new_tokens")
        or config.get("max_new_tokens")
        or 1024
    )
    consensus_seed = (
        args.consensus_seed if args.consensus_seed is not None else meta.get("run_seed", 0)
    )
    source_load_in_4bit = bool(meta.get("load_in_4bit", config.get("load_in_4bit", False)))
    consensus_4bit = (
        args.consensus_load_in_4bit if args.consensus_load_in_4bit is not None else source_load_in_4bit
    )
    judge_4bit = (
        args.judge_load_in_4bit if args.judge_load_in_4bit is not None else source_load_in_4bit
    )
    consensus_enabled = args.consensus_prompt_mode != NO_CONSENSUS_MODE
    if consensus_enabled and not consensus_model_id:
        raise ValueError("consensus model is unknown; pass --consensus-model")
    if args.judge_enabled and not judge_model_id:
        raise ValueError("judge model is unknown; pass --judge-model")

    judge_cache, judge_cache_validation, direct_predictions = None, None, None
    if args.judge_result:
        expected = {
            "judge_model": judge_model_id,
            "judge_input_mode": args.judge_input_mode,
            "judge_temperature": args.judge_temperature,
            "judge_max_new_tokens": args.judge_max_new_tokens,
            "judge_max_retries": args.judge_max_retries,
        }
        judge_cache, judge_cache_validation = load_judge_cache(
            args.judge_result,
            expected,
            input_result=args.input_result,
            require_config=args.judge_cache_require_config,
        )
        direct_predictions = direct_judge_predictions(args.judge_result)
        print(
            f"[judge-cache] usable={judge_cache_validation['usable_rows']} "
            f"mismatched={judge_cache_validation['mismatched_rows']}",
            flush=True,
        )

    pool = ModelPool(args.gpu_index, args.mem_fraction, args.trust_remote_code)

    def consensus_fn(record):
        model, tokenizer = pool.get(consensus_model_id, consensus_4bit)
        return run_consensus_round(
            model,
            tokenizer,
            record,
            args.consensus_prompt_mode,
            meta=meta,
            debate_template=args.consensus_debate_template,
            order_seed=args.consensus_order_seed,
            generation_seed=consensus_seed,
            temperature=consensus_temperature,
            max_new_tokens=consensus_max_new_tokens,
            enable_thinking=args.consensus_enable_thinking,
            seed_hook=lambda seed: set_seed(int(seed) % (2**32)),
        )

    def judge_fn(record):
        for field in ("issue", "headline", "article"):
            if not record.get(field):
                raise ValueError(
                    f"item {record['id']} lacks {field}; pass --data-path for the original dataset"
                )
        model, tokenizer = pool.get(judge_model_id, judge_4bit)
        set_seed(int(stable_seed(args.judge_order_seed, record["id"], "judge_generation")) % (2**32))
        return run_judge(
            model,
            tokenizer,
            record,
            record["raw_rounds"],
            input_mode=args.judge_input_mode,
            order_seed=args.judge_order_seed,
            max_new_tokens=args.judge_max_new_tokens,
            max_retries=args.judge_max_retries,
            temperature=args.judge_temperature,
            enable_thinking=False,
        )

    run_config = {
        "input_result": args.input_result,
        "transition_file": args.transition_file,
        "data_path": args.data_path,
        "aggregation_method": args.aggregation_method,
        "judge_trigger": args.judge_trigger,
        "consensus_prompt_mode": args.consensus_prompt_mode,
        "consensus_model": consensus_model_id if consensus_enabled else None,
        "consensus_temperature": consensus_temperature if consensus_enabled else None,
        "consensus_max_new_tokens": consensus_max_new_tokens if consensus_enabled else None,
        "consensus_seed": consensus_seed if consensus_enabled else None,
        "consensus_order_seed": args.consensus_order_seed if consensus_enabled else None,
        "consensus_enable_thinking": args.consensus_enable_thinking,
        "consensus_quantization": "4bit-nf4" if consensus_4bit else "none",
        "consensus_added_instruction": (
            RECONSIDERATION_INSTRUCTION
            if args.consensus_prompt_mode == "independent_reconsideration"
            else None
        ),
        "judge_enabled": args.judge_enabled,
        "judge_model": judge_model_id,
        "judge_input_mode": args.judge_input_mode,
        "judge_temperature": args.judge_temperature,
        "judge_order_seed": args.judge_order_seed,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "judge_max_retries": args.judge_max_retries,
        "judge_quantization": "4bit-nf4" if judge_4bit else "none",
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "judge_result_cache": args.judge_result,
        "judge_cache_fields": list(JUDGE_CACHE_CONFIG_FIELDS),
        "gold_available_to_agents_or_judge": False,
        "limit": args.limit,
        "method": "selective independent reconsideration with unanimity-based acceptance",
    }

    prefix_name = output_prefix(
        input_result=args.input_result,
        consensus_mode=args.consensus_prompt_mode,
        trigger=args.judge_trigger,
        consensus_model=consensus_model_id,
        judge_model=judge_model_id,
        consensus_temperature=consensus_temperature,
        consensus_seed=consensus_seed,
        consensus_order_seed=args.consensus_order_seed,
        consensus_max_new_tokens=consensus_max_new_tokens,
        judge_input_mode=args.judge_input_mode,
        judge_temperature=args.judge_temperature,
        judge_order_seed=args.judge_order_seed,
        aggregation_method=args.aggregation_method,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / prefix_name
    items_path = prefix.with_suffix(".items.json")

    if args.resume:
        reused = len(load_existing_rows(items_path))
        if reused:
            print(f"[resume] reusing {reused} saved rows from {items_path}", flush=True)

    def on_row(index, total, row, was_reused):
        if row["triggered"] and not was_reused:
            print(
                f"[consensus] {index}/{total} item={row['item_id']} "
                f"reached={row['consensus_reached']} judge={row['judge_source']} "
                f"final={row['final_prediction']} ({row['final_source']})",
                flush=True,
            )

    rows = process_records(
        records,
        items_path=items_path,
        save=save_json,
        resume=args.resume,
        on_row=on_row,
        aggregation_method=args.aggregation_method,
        trigger_mode=args.judge_trigger,
        fallback_seed=args.judge_order_seed,
        consensus_mode=args.consensus_prompt_mode,
        consensus_fn=consensus_fn,
        judge_fn=judge_fn,
        judge_cache=judge_cache,
        judge_enabled=args.judge_enabled,
    )

    summary = build_summary(
        rows,
        run_config,
        source_meta={**meta, "transition_file_validation": transition_summary},
        judge_cache_validation=judge_cache_validation,
        direct_predictions=direct_predictions,
        bootstrap_seed=args.judge_order_seed,
    )
    save_json(items_path, rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    prefix.with_suffix(".md").write_text(report_markdown(summary), encoding="utf-8")
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "item_id", "gold", "triggered", "trigger_reason", "baseline_prediction",
            "consensus_prompt_mode", "previous_agent_labels", "revised_agent_labels",
            "consensus_reached", "consensus_label", "consensus_reason", "judge_required",
            "judge_source", "judge_prediction", "final_prediction", "final_source",
            "fallback_used", "baseline_correct", "final_correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fields
                }
            )
    print(json.dumps(summary["consensus_stage"], ensure_ascii=False, indent=2))
    print(f"[saved] {prefix}.*")


if __name__ == "__main__":
    main()

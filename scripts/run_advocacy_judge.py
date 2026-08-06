"""Stance-advocacy debate over the whole split, then a contrastive judge.

One agent argues for each of the three labels, so the gold label is always among
the candidates; the judge reads the article plus the three anonymized, shuffled
cases and decides. Costs n_labels * rounds + 1 generations per item: 7 in the
base two-round configuration, against 6 for a two-round debate and 12 for four.

    python scripts/run_advocacy_judge.py \
      --config config/phase2_exaone_stance_minimal_en.yaml --model exaone \
      --split test --n 1001 --data-seed 0 --run-seed 6000

Pass --baseline-result to also report the paired comparison against a saved
phase2 method (majority is the strongest baseline this repository has measured).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advocacy import (  # noqa: E402
    compliance,
    judge_failure_fallback,
    round_label_trajectory,
    run_advocacy,
    run_advocacy_judge,
)
from src.consensus import (  # noqa: E402
    classification_metrics,
    mcnemar_exact,
    stable_seed,
    transition_metrics,
)
from src.llm import load_model, model_context_window  # noqa: E402
from src.prompts.advocacy import (  # noqa: E402
    PROMPT_STYLES,
    STANCE_LABELS,
    get_prompt_style,
)
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2_qwen8_stance_minimal_en.yaml")
    parser.add_argument("--model", default="qwen", choices=["qwen", "exaone"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--advocate-temperature", type=float, default=None)
    parser.add_argument("--advocate-max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="advocacy rounds: 1 is the independent cases only, 2 adds one "
             "rebuttal round (the base configuration), higher raises the limit.",
    )
    parser.add_argument(
        "--prompt-style", choices=sorted(PROMPT_STYLES), default="toc",
        help="'toc' keeps the published prompt lengths and prose output; "
             "'structured' spells out the evidence fields and uses an "
             "advocacy-aware judge with the strict JSON schema.",
    )
    parser.add_argument("--order-seed", type=int, default=8001)
    parser.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-new-tokens", type=int, default=384)
    parser.add_argument("--judge-max-retries", type=int, default=1)
    parser.add_argument("--baseline-result", help="saved phase2 result for a paired comparison")
    parser.add_argument("--baseline-method", default="majority")
    parser.add_argument("--limit", type=int, help="process only the first N items (smoke runs)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/advocacy")
    return parser.parse_args()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["models"][args.model]

    split = args.split or cfg["split"]
    n = args.n or cfg["n"]
    data_seed = cfg["seed"] if args.data_seed is None else args.data_seed
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    temperature = (
        args.advocate_temperature
        if args.advocate_temperature is not None
        else cfg["sampling"]["temperature"]
    )
    max_new_tokens = args.advocate_max_new_tokens or cfg["max_new_tokens"]
    profile = cfg.get("prompt_profile", "legacy_ko")
    judge_model_id = args.judge_model or model_cfg["id"]
    enable_thinking = False if args.model == "qwen" else None

    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=n, seed=data_seed)
    if args.limit:
        items = items[: args.limit]

    print(
        f"[cfg] model={model_cfg['id']} split={split} n={len(items)} "
        f"data_seed={data_seed} run_seed={run_seed} temperature={temperature} "
        f"prompt_style={args.prompt_style} rounds={args.rounds}"
    )
    set_seed(run_seed)
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if judge_model_id != model_cfg["id"]:
        judge_model, judge_tokenizer = load_model(
            judge_model_id,
            load_in_4bit=cfg["load_in_4bit"],
            gpu_index=cfg["gpu_index"],
            mem_fraction=cfg["mem_fraction"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
    else:
        judge_model, judge_tokenizer = model, tokenizer
    print(f"[tokens] max_new_tokens={max_new_tokens} "
          f"context_window={model_context_window(model, tokenizer)}")

    prefix_name = (
        f"advocacy_{args.model}_{split}_n{len(items)}_{profile}_d{data_seed}_s{run_seed}"
        f"_{args.prompt_style}_r{args.rounds}_ord{args.order_seed}"
        f"_judge-{safe_name(judge_model_id.split('/')[-1])}_jt{args.judge_temperature:g}"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / prefix_name
    items_path = prefix.with_suffix(".items.json")

    existing = {}
    if args.resume and items_path.exists():
        existing = {
            str(row["item_id"]): row
            for row in json.loads(items_path.read_text(encoding="utf-8"))
        }
        print(f"[resume] reusing {len(existing)} saved rows", flush=True)

    rows = []
    for index, item in enumerate(items, start=1):
        key = str(item["id"])
        if key in existing:
            rows.append(existing[key])
            continue
        record = {**item, "id": key}
        advocacy = run_advocacy(
            model,
            tokenizer,
            record,
            order_seed=args.order_seed,
            generation_seed=run_seed,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            rounds=args.rounds,
            prompt_style=args.prompt_style,
            seed_hook=lambda seed: set_seed(int(seed) % (2**32)),
        )
        cases = advocacy["cases"]
        row = {
            "item_id": key,
            "gold": item["gold"],
            "issue": item["issue"],
            "rounds": advocacy["rounds"],
            "assigned_stances": advocacy["assigned_stances"],
            "advocate_cases": cases,
            "analyses_by_round": advocacy["analyses_by_round"],
            "label_trajectory": round_label_trajectory(
                advocacy["analyses_by_round"], advocacy["assigned_stances"]
            ),
            "peer_orders": advocacy["peer_orders"],
            "compliance": compliance(cases),
            "advocacy_cost": {
                "calls": advocacy["calls"],
                "latency_seconds": advocacy["latency_seconds"],
                "token_usage": advocacy["token_usage"],
            },
            "judge_cost": {"calls": 0, "latency_seconds": 0.0,
                           "token_usage": {"input_tokens": 0, "output_tokens": 0}},
            "judge_prediction": None,
            "fallback_used": False,
            "fallback_reason": None,
        }
        if args.judge_enabled:
            set_seed(int(stable_seed(args.order_seed, key, "advocacy_judge")) % (2**32))
            judged = run_advocacy_judge(
                judge_model,
                judge_tokenizer,
                record,
                cases,
                order_seed=args.order_seed,
                max_new_tokens=args.judge_max_new_tokens,
                max_retries=args.judge_max_retries,
                temperature=args.judge_temperature,
                enable_thinking=False,
                prompt_style=args.prompt_style,
            )
            row.update(
                {
                    "judge_prediction": judged["prediction"],
                    "judge_raw_output": judged["raw_output"],
                    "judge_parsed_output": judged["parsed_output"],
                    "judge_attempts": judged["attempts"],
                    "candidate_order": judged["candidate_order"],
                    "judge_cost": {
                        "calls": len(judged["attempts"]),
                        "latency_seconds": judged["latency_seconds"],
                        "token_usage": judged["token_usage"],
                    },
                }
            )
        if row["judge_prediction"]:
            row["pred"] = row["judge_prediction"]
            row["pred_source"] = "judge"
        else:
            label, reason = judge_failure_fallback(key, args.order_seed)
            row["pred"] = label
            row["pred_source"] = "fallback"
            row["fallback_used"] = True
            row["fallback_reason"] = reason
        row["correct"] = row["pred"] == item["gold"]
        rows.append(row)
        save_json(items_path, rows)
        print(f"[advocacy] {index}/{len(items)} item={key} pred={row['pred']} "
              f"({row['pred_source']})", flush=True)

    preds = [row["pred"] for row in rows]
    golds = [row["gold"] for row in rows]
    summary = {
        "items": len(rows),
        "config": {
            "model_id": model_cfg["id"],
            "judge_model_id": judge_model_id,
            "prompt_profile": profile,
            "split": split,
            "n": len(rows),
            "data_seed": data_seed,
            "run_seed": run_seed,
            "advocate_temperature": temperature,
            "advocate_max_new_tokens": max_new_tokens,
            "rounds": args.rounds,
            "order_seed": args.order_seed,
            "judge_enabled": args.judge_enabled,
            "judge_temperature": args.judge_temperature,
            "judge_max_new_tokens": args.judge_max_new_tokens,
            "judge_max_retries": args.judge_max_retries,
            "labels": list(STANCE_LABELS),
            "prompt_style": args.prompt_style,
            "advocate_system_prompt": get_prompt_style(args.prompt_style)["advocate_system"],
            "advocate_user_template": get_prompt_style(args.prompt_style)["advocate_user"],
            "judge_system_prompt": get_prompt_style(args.prompt_style)["judge_system"],
            "gold_available_to_agents_or_judge": False,
        },
        "metrics": classification_metrics(preds, golds),
        "prediction_source_counts": dict(Counter(row["pred_source"] for row in rows)),
        "fallback_used": sum(row["fallback_used"] for row in rows),
        "compliance": {
            "agent_answers": sum(r["compliance"]["agents"] for r in rows),
            "declared_label": sum(r["compliance"]["declared_label"] for r in rows),
            "declared_defection": sum(r["compliance"]["declared_defection"] for r in rows),
            "defected_by_stance": dict(
                Counter(s for r in rows for s in r["compliance"]["defected_stances"])
            ),
            "last_mention_only": sum(r["compliance"]["last_mention_only"] for r in rows),
            "last_mention_differs": sum(r["compliance"]["last_mention_differs"] for r in rows),
            "held_assigned_by_round": [
                sum(sum(t["held_assigned"]) for r in rows for t in r["label_trajectory"]
                    if t["round"] == round_index)
                for round_index in range(args.rounds)
            ],
        },
        "judged_label_counts": dict(Counter(preds)),
        "gold_label_counts": dict(Counter(golds)),
        "cost": {
            "advocacy": {
                "calls": sum(r["advocacy_cost"]["calls"] for r in rows),
                "latency_seconds": sum(r["advocacy_cost"]["latency_seconds"] for r in rows),
                "input_tokens": sum(
                    r["advocacy_cost"]["token_usage"]["input_tokens"] for r in rows),
                "output_tokens": sum(
                    r["advocacy_cost"]["token_usage"]["output_tokens"] for r in rows),
            },
            "judge": {
                "calls": sum(r["judge_cost"]["calls"] for r in rows),
                "latency_seconds": sum(r["judge_cost"]["latency_seconds"] for r in rows),
                "input_tokens": sum(
                    r["judge_cost"]["token_usage"]["input_tokens"] for r in rows),
                "output_tokens": sum(
                    r["judge_cost"]["token_usage"]["output_tokens"] for r in rows),
            },
        },
    }

    if args.baseline_result:
        baseline = json.loads(Path(args.baseline_result).read_text(encoding="utf-8"))
        method = baseline.get(args.baseline_method) or {}
        shared = [row for row in rows if str(row["item_id"]) in method]
        if shared:
            before = [method[str(row["item_id"])]["pred"] for row in shared]
            after = [row["pred"] for row in shared]
            covered_golds = [row["gold"] for row in shared]
            summary["baseline_comparison"] = {
                "file": args.baseline_result,
                "method": args.baseline_method,
                "shared_items": len(shared),
                "baseline_metrics": classification_metrics(before, covered_golds),
                "transitions": transition_metrics(before, after, covered_golds),
                "mcnemar_exact": mcnemar_exact(before, after, covered_golds),
            }

    save_json(items_path, rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as handle:
        fields = ["item_id", "gold", "pred", "pred_source", "correct",
                  "rounds", "assigned_stances", "fallback_used", "fallback_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(row.get(key), ensure_ascii=False)
                 if isinstance(row.get(key), (list, dict)) else row.get(key)
                 for key in fields}
            )

    print("\n===== ADVOCACY + JUDGE =====")
    print(f"accuracy={summary['metrics']['accuracy']:.4f}  "
          f"macro_f1={summary['metrics']['macro_f1']:.4f}  items={len(rows)}")
    c = summary["compliance"]
    print(f"fallbacks={summary['fallback_used']}  "
          f"advocates that declared a different stance="
          f"{c['declared_defection']}/{c['declared_label']} declared "
          f"(of {c['agent_answers']} answers; {c['last_mention_differs']}"
          f"/{c['last_mention_only']} only differ in the last label mentioned)")
    comparison = summary.get("baseline_comparison")
    if comparison:
        t = comparison["transitions"]
        print(f"vs {comparison['method']}: "
              f"{comparison['baseline_metrics']['accuracy']:.4f} -> "
              f"{summary['metrics']['accuracy']:.4f}  "
              f"w2c={t['wrong_to_correct']} c2w={t['correct_to_wrong']} "
              f"net={t['net_improvement']:+d} "
              f"p={comparison['mcnemar_exact']['p_value_two_sided']:.3f}")
    print(f"[saved] {prefix}.*")


if __name__ == "__main__":
    main()

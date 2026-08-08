"""Selective advocacy over a saved debate run: commission only what is missing.

Reads a saved phase2 result, keeps its free-form majority as the prediction on
every stable item, and on the rest commissions an advocate case for each label
no agent proposed, then adjudicates with a judge that is given the independent
vote count. Nothing is regenerated for the agents that already answered.

    python scripts/run_selective_advocacy.py \
      --input-result results/phase2/<run>.json \
      --data-path data/k-news-stance_nosegment.json \
      --config config/phase2_qwen8_stance_minimal_en.yaml --model qwen \
      --advocate-model LGAI-EXAONE/EXAONE-4.0-1.2B \
      --trigger unstable_or_split

The advocate and the judge can be different models: the advocate writes the
missing counterfactual in the same voice as the debaters that produced the run,
while the judge should be the strongest model available.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advocacy import run_single_advocate  # noqa: E402
from src.consensus import (  # noqa: E402
    LABELS,
    aggregate,
    bootstrap_accuracy_delta,
    boundary_transition_counts,
    classification_metrics,
    mcnemar_exact,
    stable_seed,
    transition_metrics,
    unique_majority,
)
from src.consensus_io import load_consensus_records  # noqa: E402
from src.llm import load_model  # noqa: E402
from src.prompts.advocacy import PROMPT_STYLES  # noqa: E402
from src.selective_advocacy import (  # noqa: E402
    SELECTIVE_JUDGE_SYSTEM_PROMPT,
    TRIGGERS,
    independent_case,
    proposed_labels,
    run_selective_judge,
    selective_trigger,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result", required=True, help="saved phase2 result JSON")
    parser.add_argument("--data-path", help="dataset, needed for the article text")
    parser.add_argument("--config", default="config/phase2_qwen8_stance_minimal_en.yaml")
    parser.add_argument("--model", default="qwen", choices=["qwen", "exaone"],
                        help="which entry of the yaml config supplies the judge")
    parser.add_argument("--advocate-model", help="model that writes the commissioned "
                                                 "cases; defaults to the judge model")
    parser.add_argument("--judge-model", help="defaults to the config model id")
    parser.add_argument("--aggregation-method", default="current_final_majority")
    parser.add_argument("--trigger", choices=list(TRIGGERS), default="unstable_or_split",
                        help="where to spend the extra calls. The default is a "
                             "tie, a 2:1 split or a round0/final disagreement; "
                             "'missing_label' fires on ~99%% of items and is kept "
                             "only as the ablation that shows why.")
    parser.add_argument("--trigger-scope", choices=["last", "any_round"], default="last",
                        help="'last' counts only the final round's labels as proposed; "
                             "'any_round' counts a label proposed in any round")
    parser.add_argument("--prompt-style", choices=sorted(PROMPT_STYLES), default="toc",
                        help="prompt style for the commissioned advocate case")
    parser.add_argument("--advocate-temperature", type=float, default=1.0)
    parser.add_argument("--advocate-max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-retries", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=8001)
    parser.add_argument("--run-seed", type=int, default=6000)
    parser.add_argument("--judge-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, help="process only the first N items")
    parser.add_argument("--dry-run", action="store_true",
                        help="report trigger counts and the exact call budget, load no model")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/selective_advocacy")
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
    judge_model_id = args.judge_model or model_cfg["id"]
    advocate_model_id = args.advocate_model or judge_model_id

    records, meta, _ = load_consensus_records(args.input_result, None, args.data_path)
    if args.limit:
        records = records[: args.limit]

    # --- what would run, before anything is loaded ---------------------------
    plan = []
    for record in records:
        rounds = record["round_predictions"]
        triggered, reason, missing = selective_trigger(rounds, args.trigger, args.trigger_scope)
        plan.append((record, triggered, reason, missing))
    triggered_count = sum(1 for _, triggered, _, _ in plan if triggered)
    advocate_calls = sum(len(missing) for _, triggered, _, missing in plan if triggered)
    print(f"[plan] {len(records)} items · triggered {triggered_count} "
          f"({triggered_count / max(1, len(records)):.1%}) · "
          f"{advocate_calls} commissioned advocate calls + {triggered_count} judge calls "
          f"= {advocate_calls + triggered_count} generations "
          f"({(advocate_calls + triggered_count) / max(1, len(records)):.2f}/item; "
          f"full advocacy at 2 rounds is 7.00/item)", flush=True)
    print(f"[plan] trigger reasons: "
          f"{dict(Counter(reason for _, triggered, reason, _ in plan if triggered))}", flush=True)
    # a trigger that fires on nearly everything is not selective, and the whole
    # argument for this design is that the unanimous items keep their vote
    if triggered_count > 0.5 * len(records):
        print(f"[plan][warn] this trigger fires on {triggered_count / len(records):.0%} of "
              f"items; it has degenerated towards --trigger all, and the vote signal "
              f"it was meant to preserve is being overridden on most items", flush=True)
    if args.dry_run:
        return

    prefix_name = (
        f"seladv_{Path(args.input_result).stem}_{args.trigger}-{args.trigger_scope}"
        f"_{args.prompt_style}_ord{args.order_seed}_s{args.run_seed}"
        f"_adv-{safe_name(advocate_model_id.split('/')[-1])}"
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

    set_seed(args.run_seed)
    # nothing is loaded when every triggered item is already saved
    outstanding = [
        record for record, triggered, _, _ in plan
        if triggered and str(record["id"]) not in existing
    ]
    needs_models = args.judge_enabled and bool(outstanding)
    judge_model = judge_tokenizer = advocate_model = advocate_tokenizer = None
    if needs_models:
        judge_model, judge_tokenizer = load_model(
            judge_model_id,
            load_in_4bit=cfg["load_in_4bit"],
            gpu_index=cfg["gpu_index"],
            mem_fraction=cfg["mem_fraction"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        if advocate_model_id == judge_model_id:
            advocate_model, advocate_tokenizer = judge_model, judge_tokenizer
        else:
            advocate_model, advocate_tokenizer = load_model(
                advocate_model_id,
                load_in_4bit=cfg["load_in_4bit"],
                gpu_index=cfg["gpu_index"],
                mem_fraction=cfg["mem_fraction"],
                trust_remote_code=model_cfg.get("trust_remote_code", False),
            )

    rows = []
    wall_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    generated = 0
    for index, (record, triggered, reason, missing) in enumerate(plan, start=1):
        key = str(record["id"])
        if key in existing:
            rows.append(existing[key])
            continue
        rounds = record["round_predictions"]
        raw_rounds = record["raw_rounds"]
        baseline = aggregate(args.aggregation_method, rounds, record["id"], args.order_seed)
        votes = proposed_labels(rounds, args.trigger_scope)
        row = {
            "item_id": key,
            "gold": record.get("gold"),
            "issue": record.get("issue"),
            "headline": record.get("headline"),
            "round0_agent_preds": rounds[0],
            "final_agent_preds": rounds[-1],
            "independent_votes": {label: int(votes.get(label, 0)) for label in LABELS},
            "round0_majority": unique_majority(rounds[0]).label,
            "final_majority": unique_majority(rounds[-1]).label,
            "trigger_mode": args.trigger,
            "trigger_reason": reason,
            "triggered": triggered,
            "missing_labels": missing,
            "baseline_prediction": baseline["pred"],
            "baseline_reason": baseline["reason"],
            "commissioned_cases": [],
            "candidate_order": [],
            "judge_prediction": None,
            "judge_raw_output": None,
            "judge_parsed_output": None,
            "advocate_cost": {"calls": 0, "latency_seconds": 0.0,
                              "token_usage": {"input_tokens": 0, "output_tokens": 0}},
            "judge_cost": {"calls": 0, "latency_seconds": 0.0,
                           "token_usage": {"input_tokens": 0, "output_tokens": 0}},
        }
        final_prediction = baseline["pred"]

        if triggered and args.judge_enabled:
            for field in ("issue", "headline", "article"):
                if not record.get(field):
                    raise SystemExit(
                        f"item {key} lacks {field}; pass --data-path for the original dataset"
                    )
            item = {**record, "id": key}
            cases = []
            for label in LABELS:
                found = independent_case(raw_rounds[-1], rounds[-1], label)
                if found is not None:
                    cases.append(found)
                    continue
                commissioned = run_single_advocate(
                    advocate_model,
                    advocate_tokenizer,
                    item,
                    label,
                    temperature=args.advocate_temperature,
                    max_new_tokens=args.advocate_max_new_tokens,
                    enable_thinking=False,
                    prompt_style=args.prompt_style,
                    generation_seed=args.run_seed,
                    seed_hook=lambda seed: set_seed(int(seed) % (2**32)),
                )
                cases.append(
                    {
                        "stance": label,
                        "origin": "commissioned",
                        "independent_supporters": 0,
                        "source_agent_index": None,
                        "analysis": commissioned["analysis"],
                        "stated_label": commissioned["stated_label"],
                        "stated_label_source": commissioned["stated_label_source"],
                    }
                )
                row["commissioned_cases"].append(
                    {k: v for k, v in commissioned.items() if k != "analysis"}
                )
                row["advocate_cost"]["calls"] += 1
                row["advocate_cost"]["latency_seconds"] += commissioned["latency_seconds"]
                for field, value in commissioned["token_usage"].items():
                    row["advocate_cost"]["token_usage"][field] += value

            set_seed(int(stable_seed(args.order_seed, key, "selective_advocacy_judge")) % (2**32))
            judged = run_selective_judge(
                judge_model,
                judge_tokenizer,
                item,
                cases,
                votes,
                order_seed=args.order_seed,
                max_new_tokens=args.judge_max_new_tokens,
                max_retries=args.judge_max_retries,
                temperature=args.judge_temperature,
                enable_thinking=False,
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
            if judged["prediction"]:
                final_prediction = judged["prediction"]
            else:
                # no judge label at all: keep the vote rather than invent one
                row["fallback_to_baseline"] = True

        row["final_prediction"] = final_prediction
        row["pred"] = final_prediction
        row["baseline_correct"] = row["baseline_prediction"] == record.get("gold")
        row["correct"] = final_prediction == record.get("gold")
        rows.append(row)
        save_json(items_path, rows)
        generated += 1
        if triggered:
            elapsed = time.perf_counter() - wall_start
            print(f"[seladv] {index}/{len(plan)} item={key} {reason} "
                  f"{row['baseline_prediction']} -> {final_prediction}  "
                  f"elapsed={elapsed / 60:.1f}m", flush=True)

    baseline_preds = [row["baseline_prediction"] for row in rows]
    final_preds = [row["final_prediction"] for row in rows]
    golds = [row["gold"] for row in rows]
    fired = [row for row in rows if row["triggered"]]
    summary = {
        "items": len(rows),
        "source_meta": meta,
        "config": {
            "input_result": args.input_result,
            "judge_model_id": judge_model_id,
            "advocate_model_id": advocate_model_id,
            "aggregation_method": args.aggregation_method,
            "trigger": args.trigger,
            "trigger_scope": args.trigger_scope,
            "prompt_style": args.prompt_style,
            "advocate_temperature": args.advocate_temperature,
            "judge_temperature": args.judge_temperature,
            "order_seed": args.order_seed,
            "run_seed": args.run_seed,
            "judge_system_prompt": SELECTIVE_JUDGE_SYSTEM_PROMPT,
            "gold_available_to_agents_or_judge": False,
        },
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_clock_seconds": time.perf_counter() - wall_start,
            "items_generated_this_run": generated,
        },
        "baseline_metrics": classification_metrics(baseline_preds, golds),
        "metrics": classification_metrics(final_preds, golds),
        "overall_transitions": transition_metrics(baseline_preds, final_preds, golds),
        "mcnemar_exact": mcnemar_exact(baseline_preds, final_preds, golds),
        "bootstrap_accuracy_delta": bootstrap_accuracy_delta(
            baseline_preds, final_preds, golds, args.order_seed
        ),
        "boundary_transitions": boundary_transition_counts(baseline_preds, final_preds),
        "trigger_subset": {
            "triggered_total": len(fired),
            "trigger_ratio": len(fired) / max(1, len(rows)),
            "reasons": dict(Counter(row["trigger_reason"] for row in fired)),
            "missing_label_counts": dict(
                Counter(label for row in fired for label in row["missing_labels"])
            ),
            "baseline_accuracy": (
                sum(row["baseline_correct"] for row in fired) / max(1, len(fired))
            ),
            "final_accuracy": sum(row["correct"] for row in fired) / max(1, len(fired)),
            "transitions": transition_metrics(
                [row["baseline_prediction"] for row in fired],
                [row["final_prediction"] for row in fired],
                [row["gold"] for row in fired],
            ),
        },
        "judge_diagnostics": {
            "winning_candidate_position": dict(
                Counter(
                    next(
                        (c["candidate_id"] for c in row.get("candidate_order", [])
                         if c["stance_argued"] == row["final_prediction"]),
                        None,
                    )
                    for row in fired if row.get("candidate_order")
                )
            ),
            # did the judge follow the commissioned counterfactual or the vote?
            "picked_commissioned": sum(
                1 for row in fired
                for c in row.get("candidate_order", [])
                if c["stance_argued"] == row["final_prediction"] and c["origin"] == "commissioned"
            ),
            "picked_commissioned_correct": sum(
                1 for row in fired if row["correct"]
                for c in row.get("candidate_order", [])
                if c["stance_argued"] == row["final_prediction"] and c["origin"] == "commissioned"
            ),
            "labels_recovered_from_text": sum(
                1 for row in rows
                if "recovered" in ((row.get("judge_attempts") or [{}])[-1].get("parse_error") or "")
            ),
        },
        "cost": {
            "advocate_calls": sum(row["advocate_cost"]["calls"] for row in rows),
            "judge_calls": sum(row["judge_cost"]["calls"] for row in rows),
            "generations_per_item": (
                sum(row["advocate_cost"]["calls"] + row["judge_cost"]["calls"] for row in rows)
                / max(1, len(rows))
            ),
            "latency_seconds": sum(
                row["advocate_cost"]["latency_seconds"] + row["judge_cost"]["latency_seconds"]
                for row in rows
            ),
        },
    }

    save_json(items_path, rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as handle:
        fields = ["item_id", "gold", "triggered", "trigger_reason", "missing_labels",
                  "baseline_prediction", "judge_prediction", "final_prediction",
                  "baseline_correct", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(row.get(key), ensure_ascii=False)
                 if isinstance(row.get(key), (list, dict)) else row.get(key)
                 for key in fields}
            )

    print("\n===== SELECTIVE ADVOCACY =====")
    subset = summary["trigger_subset"]
    t = summary["overall_transitions"]
    print(f"{args.aggregation_method} {summary['baseline_metrics']['accuracy']:.4f} -> "
          f"selective advocacy {summary['metrics']['accuracy']:.4f}  "
          f"w2c={t['wrong_to_correct']} c2w={t['correct_to_wrong']} "
          f"net={t['net_improvement']:+d} "
          f"p={summary['mcnemar_exact']['p_value_two_sided']:.3f}")
    print(f"triggered {subset['triggered_total']}/{len(rows)} "
          f"({subset['trigger_ratio']:.1%}); on those items "
          f"{subset['baseline_accuracy']:.4f} -> {subset['final_accuracy']:.4f}")
    print(f"kept {t['correct_preservation_rate']:.1%} of the baseline's correct answers, "
          f"fixed {t['error_correction_rate']:.1%} of its errors")
    print(f"cost {summary['cost']['generations_per_item']:.2f} generations/item "
          f"({summary['cost']['advocate_calls']} advocate + "
          f"{summary['cost']['judge_calls']} judge)")
    diag = summary["judge_diagnostics"]
    print(f"judge chose a commissioned counterfactual on {diag['picked_commissioned']} items "
          f"({diag['picked_commissioned_correct']} of them correct)")
    print(f"[saved] {prefix}.*")


if __name__ == "__main__":
    main()

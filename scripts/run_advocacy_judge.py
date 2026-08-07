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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.advocacy import (  # noqa: E402
    cases_from_round,
    compliance,
    judge_failure_fallback,
    position_bias_chi_square,
    resolve_round_index,
    round_label_trajectory,
    run_advocacy,
    run_advocacy_judge,
)
from src.consensus import (  # noqa: E402
    bootstrap_accuracy_delta,
    boundary_transition_counts,
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
    parser.add_argument(
        "--judge-load-in-4bit", action=argparse.BooleanOptionalAction, default=None,
        help="quantization for the judge; defaults to the config's load_in_4bit. "
             "A judge-only re-run should not inherit the advocate's setting by accident.",
    )
    parser.add_argument(
        "--judge-round", default="last",
        help="which advocacy round the judge reads: 'last' (default) or a round "
             "index. --judge-round 0 gives the judge the independent cases, "
             "before the agents start addressing each other.",
    )
    parser.add_argument(
        "--judge-order-seed", type=int, default=None,
        help="seed for the judge's candidate shuffle only; defaults to "
             "--order-seed. Change it alone to re-test position bias on reused "
             "cases without touching the label assignment they were built with.",
    )
    parser.add_argument(
        "--reuse-advocacy",
        help="a previous .items.json to take the advocate cases from, so only "
             "the judge is re-run. Isolates a judge change at a fraction of the "
             "cost; the advocacy settings must match.",
    )
    parser.add_argument(
        "--allow-reuse-mismatch", action="store_true",
        help="downgrade the --reuse-advocacy config check to a warning",
    )
    parser.add_argument("--baseline-result", help="saved phase2 result for a paired comparison")
    parser.add_argument("--baseline-method", default="majority")
    parser.add_argument("--limit", type=int, help="process only the first N items (smoke runs)")
    parser.add_argument(
        "--qualitative-per-cell", type=int, default=3,
        help="samples written to the qualitative report per (gold, prediction) cell",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/advocacy")
    return parser.parse_args()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


# What has to match for the two runs to be the same experiment on the same
# items: the split, the item sampling, the label assignment and the round count.
REUSE_MUST_MATCH = (
    "split", "n", "data_seed", "run_seed", "order_seed", "rounds", "limit",
)

# Reported, never enforced. In a judge-only pass no advocate runs, so --config,
# --model and the advocate decoding settings describe *this run's judge*, not the
# advocates that wrote the saved cases -- judging EXAONE cases with a Qwen config
# is the intended use, not a mismatch. --prompt-style is likewise the thing being
# ablated. All of them are still printed, because the advocates' identity is part
# of what a result means.
REUSE_PROVENANCE = (
    "config", "model", "prompt_style", "advocate_temperature",
    "advocate_max_new_tokens", "judge_model",
)


def sibling_config_path(items_path: Path) -> Path:
    name = items_path.name
    if name.endswith(".items.json"):
        name = name[: -len(".items.json")] + ".config.json"
        return items_path.with_name(name)
    return items_path.with_suffix(".config.json")


def validate_reuse_config(args, source_path: Path):
    """Check the saved cases were produced by the run we think they were.

    Matching the round count and the presence of cases -- the previous check --
    lets a different model, prompt style, order seed or run seed pass silently,
    which makes a judge-only comparison unpaired without saying so.
    """
    config_path = sibling_config_path(source_path)
    if not config_path.exists():
        message = f"{source_path} has no sibling {config_path.name}; cannot verify the advocacy settings"
        if not args.allow_reuse_mismatch:
            raise SystemExit(message + " (pass --allow-reuse-mismatch to proceed anyway)")
        print(f"[reuse][warn] {message}", flush=True)
        return {}
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    mismatched, unchecked = [], []
    for key in REUSE_MUST_MATCH:
        mine, theirs = getattr(args, key, None), saved.get(key)
        if mine is None or theirs is None:
            if mine != theirs:
                unchecked.append(f"{key}: this run {mine!r} vs saved {theirs!r}")
            continue
        if mine != theirs:
            mismatched.append(f"{key}: this run {mine!r} vs saved {theirs!r}")
    if unchecked:
        print("[reuse][warn] resolved from the yaml config, not compared: "
              + "; ".join(unchecked), flush=True)
    if mismatched:
        message = ("the saved advocacy covers different items than this run:\n  "
                   + "\n  ".join(mismatched))
        if not args.allow_reuse_mismatch:
            raise SystemExit(message + "\nre-run advocacy, or pass --allow-reuse-mismatch")
        print(f"[reuse][warn] {message}", flush=True)
    changed = [
        f"{key}: {saved.get(key)!r} -> {getattr(args, key, None)!r}"
        for key in REUSE_PROVENANCE if saved.get(key) != getattr(args, key, None)
    ]
    if changed:
        print("[reuse] the cases were written under, and this pass judges them with: "
              + "; ".join(changed), flush=True)
    if saved.get("reuse_advocacy"):
        print(f"[reuse] source was itself a re-judge of {saved['reuse_advocacy']}", flush=True)
    return saved


def advocate_model_from(saved_config):
    """The model that actually wrote the reused cases.

    In a judge-only pass the yaml on the command line describes the judge, so
    without this the summary would name the judge as the advocate.
    """
    try:
        with open(saved_config["config"], encoding="utf-8") as handle:
            return yaml.safe_load(handle)["models"][saved_config["model"]]["id"]
    except (KeyError, TypeError, OSError):
        return None


def clip(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + " …"


def qualitative_report(rows, labels, per_cell, summary):
    """Samples for error analysis, grouped by (gold, prediction).

    Each sample carries the headline, which stance each agent was told to argue,
    the judge's own reasoning, and the opening of the case that argued the gold
    label -- enough to see whether the judge ignored a good argument or the
    advocate never made one.
    """
    lines = [
        "# Advocacy + judge: qualitative samples",
        "",
        f"- accuracy {summary['metrics']['accuracy']:.4f} · "
        f"macro-F1 {summary['metrics']['macro_f1']:.4f} · {summary['items']} items",
        f"- up to {per_cell} samples per (gold → predicted) cell, correct cells first",
        "",
    ]
    cells = {}
    for row in rows:
        cells.setdefault((row["gold"], row["pred"]), []).append(row)
    ordered = sorted(cells, key=lambda key: (key[0] != key[1], key[0], key[1]))
    for gold, pred in ordered:
        bucket = cells[(gold, pred)]
        mark = "correct" if gold == pred else "error"
        lines += [f"## gold={gold} → pred={pred}  ({len(bucket)} items, {mark})", ""]
        for row in bucket[:per_cell]:
            gold_case = next(
                (case for case in row["advocate_cases"] if case["stance"] == gold), None
            )
            parsed = row.get("judge_parsed_output") or {}
            judge_text = parsed.get("rationale") or row.get("judge_raw_output") or ""
            lines += [
                f"### item {row['item_id']}",
                f"- issue: {clip(row.get('issue'), 90)}",
                f"- headline: {clip(row.get('headline'), 110)}",
                f"- assigned: {row['assigned_stances']}",
                f"- judge said: {clip(judge_text, 400)}",
                f"- the case for the gold label ({gold}): "
                f"{clip(gold_case['analysis'] if gold_case else '', 400)}",
                "",
            ]
    return "\n".join(lines) + "\n"


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

    judge_order_seed = (
        args.order_seed if args.judge_order_seed is None else args.judge_order_seed
    )
    judge_4bit = (
        cfg["load_in_4bit"] if args.judge_load_in_4bit is None else args.judge_load_in_4bit
    )

    print(
        f"[cfg] model={model_cfg['id']} split={split} n={len(items)} "
        f"data_seed={data_seed} run_seed={run_seed} temperature={temperature} "
        f"prompt_style={args.prompt_style} rounds={args.rounds} "
        f"judge_round={args.judge_round} judge_order_seed={judge_order_seed}"
    )

    # --- reuse is resolved before anything touches the GPU -------------------
    reused_advocacy = {}
    reuse_source_config = {}
    if args.reuse_advocacy:
        source_path = Path(args.reuse_advocacy)
        reuse_source_config = validate_reuse_config(args, source_path)
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        reused_advocacy = {str(row["item_id"]): row for row in source_rows}
        bad = [
            key for key, row in reused_advocacy.items()
            if row.get("rounds") != args.rounds
            or not row.get("advocate_cases")
            or len(row.get("analyses_by_round") or []) != args.rounds
        ]
        if bad:
            raise SystemExit(
                f"{args.reuse_advocacy} has {len(bad)} items whose advocacy "
                f"does not match --rounds {args.rounds}; re-run advocacy instead"
            )
        missing = [str(item["id"]) for item in items if str(item["id"]) not in reused_advocacy]
        if missing:
            raise SystemExit(
                f"{args.reuse_advocacy} is missing {len(missing)} of the {len(items)} "
                f"items in this split (first: {missing[:3]}); it cannot be reused here"
            )
        print(f"[reuse] taking advocate cases for {len(reused_advocacy)} items from "
              f"{args.reuse_advocacy}; the advocate model will not be loaded", flush=True)

    set_seed(run_seed)
    # A judge-only pass has no advocate to run, so loading the advocate model
    # only occupies VRAM and forces the judge to inherit its quantization.
    if reused_advocacy:
        model = tokenizer = None
    else:
        model, tokenizer = load_model(
            model_cfg["id"],
            load_in_4bit=cfg["load_in_4bit"],
            gpu_index=cfg["gpu_index"],
            mem_fraction=cfg["mem_fraction"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
    if not args.judge_enabled:
        judge_model, judge_tokenizer = None, None
    elif model is not None and judge_model_id == model_cfg["id"] and judge_4bit == cfg["load_in_4bit"]:
        judge_model, judge_tokenizer = model, tokenizer
    else:
        judge_model, judge_tokenizer = load_model(
            judge_model_id,
            load_in_4bit=judge_4bit,
            gpu_index=cfg["gpu_index"],
            mem_fraction=cfg["mem_fraction"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
    reference_model = model if model is not None else judge_model
    reference_tokenizer = tokenizer if tokenizer is not None else judge_tokenizer
    if reference_model is not None:
        print(f"[tokens] max_new_tokens={max_new_tokens} "
              f"context_window={model_context_window(reference_model, reference_tokenizer)}")

    judge_round_tag = "" if str(args.judge_round) == "last" else f"_jr{args.judge_round}"
    judge_seed_tag = "" if judge_order_seed == args.order_seed else f"_jord{judge_order_seed}"
    prefix_name = (
        f"advocacy_{args.model}_{split}_n{len(items)}_{profile}_d{data_seed}_s{run_seed}"
        f"_{args.prompt_style}_r{args.rounds}_ord{args.order_seed}"
        f"{'_rejudge' if args.reuse_advocacy else ''}{judge_round_tag}{judge_seed_tag}"
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
    wall_start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    generated = 0
    for index, item in enumerate(items, start=1):
        key = str(item["id"])
        if key in existing:
            rows.append(existing[key])
            continue
        record = {**item, "id": key}
        source = reused_advocacy.get(key)
        if source is not None:
            advocacy = {
                "rounds": source["rounds"],
                "assigned_stances": source["assigned_stances"],
                "analyses_by_round": source["analyses_by_round"],
                "cases": source["advocate_cases"],
                "peer_orders": source.get("peer_orders", []),
                "cost_by_round": source["advocacy_cost"].get("by_round", []),
                "calls": 0,
                "latency_seconds": 0.0,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }
        elif reused_advocacy:
            raise SystemExit(f"item {key} is missing from {args.reuse_advocacy}")
        else:
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
        # the judge reads one chosen round, not necessarily the last one
        judged_round = resolve_round_index(advocacy["analyses_by_round"], args.judge_round)
        cases = cases_from_round(
            advocacy["analyses_by_round"], advocacy["assigned_stances"], judged_round
        )
        row = {
            "item_id": key,
            "gold": item["gold"],
            "issue": item["issue"],
            "headline": item["headline"],
            "rounds": advocacy["rounds"],
            "judged_round": judged_round,
            "advocacy_source": "reused" if source is not None else "generated",
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
                "by_round": advocacy["cost_by_round"],
            },
            "judge_cost": {"calls": 0, "latency_seconds": 0.0,
                           "token_usage": {"input_tokens": 0, "output_tokens": 0}},
            "judge_prediction": None,
            "fallback_used": False,
            "fallback_reason": None,
        }
        if args.judge_enabled:
            set_seed(int(stable_seed(judge_order_seed, key, "advocacy_judge")) % (2**32))
            judged = run_advocacy_judge(
                judge_model,
                judge_tokenizer,
                record,
                cases,
                order_seed=judge_order_seed,
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
        generated += 1
        elapsed = time.perf_counter() - wall_start
        rate = elapsed / generated
        remaining = (len(items) - index) * rate
        print(f"[advocacy] {index}/{len(items)} item={key} pred={row['pred']} "
              f"({row['pred_source']})  {rate:.1f}s/item  "
              f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m", flush=True)

    preds = [row["pred"] for row in rows]
    golds = [row["gold"] for row in rows]
    winning_positions = dict(
        Counter(
            next(
                (c["candidate_id"] for c in row.get("candidate_order", [])
                 if c["stance_argued"] == row["pred"]),
                None,
            )
            for row in rows if row.get("candidate_order")
        )
    )
    summary = {
        "items": len(rows),
        "config": {
            "model_id": model_cfg["id"],
            # in a rejudge the yaml describes the judge, so name the advocates
            "advocate_model_id": (
                advocate_model_from(reuse_source_config) or model_cfg["id"]
                if reuse_source_config else model_cfg["id"]
            ),
            "judge_model_id": judge_model_id,
            "prompt_profile": profile,
            "split": split,
            "n": len(rows),
            "data_seed": data_seed,
            "run_seed": run_seed,
            "advocate_temperature": temperature,
            "advocate_max_new_tokens": max_new_tokens,
            "rounds": args.rounds,
            "judge_round": args.judge_round,
            "judged_round_index": rows[0].get("judged_round") if rows else None,
            "reuse_advocacy": args.reuse_advocacy,
            "reuse_source_config": reuse_source_config or None,
            "order_seed": args.order_seed,
            "judge_order_seed": judge_order_seed,
            "judge_load_in_4bit": judge_4bit,
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
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_clock_seconds": time.perf_counter() - wall_start,
            "items_generated_this_run": generated,
            "items_reused_from_resume": len(rows) - generated,
            "seconds_per_generated_item": (
                (time.perf_counter() - wall_start) / generated if generated else None
            ),
        },
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
        "judge_diagnostics": {
            # if the judge tracked candidate position rather than argument
            # quality, the winning candidate_id would not be uniform
            "winning_candidate_position": winning_positions,
            "winning_candidate_position_test": position_bias_chi_square(winning_positions),
            # a label read out of malformed output is not a parsed verdict; the
            # previous count called every prose (toc) answer a recovery
            "label_recovered_from_text": sum(
                1 for row in rows
                if "recovered" in ((row.get("judge_attempts") or [{}])[-1].get("parse_error") or "")
            ),
            "judge_parse_errors": sum(
                1 for row in rows
                for attempt in (row.get("judge_attempts") or [])
                if attempt.get("parse_error")
            ),
            "judge_retries": sum(
                max(0, len(row.get("judge_attempts") or []) - 1) for row in rows
            ),
            # a judge that answers with the bare verdict skipped the contrastive
            # discussion the method depends on
            "judge_output_chars": {
                "median": sorted(len(row.get("judge_raw_output") or "") for row in rows)[
                    len(rows) // 2
                ] if rows else 0,
                "under_60_chars": sum(
                    1 for row in rows if len(row.get("judge_raw_output") or "") < 60
                ),
            },
        },
        "cost": {
            "advocacy": {
                # a judge-only pass generates nothing; by_round then describes the
                # run the cases came from, not this one
                "inherited_from_reuse": bool(args.reuse_advocacy),
                "calls": sum(r["advocacy_cost"]["calls"] for r in rows),
                "latency_seconds": sum(r["advocacy_cost"]["latency_seconds"] for r in rows),
                "input_tokens": sum(
                    r["advocacy_cost"]["token_usage"]["input_tokens"] for r in rows),
                "output_tokens": sum(
                    r["advocacy_cost"]["token_usage"]["output_tokens"] for r in rows),
                "by_round": [
                    {
                        "round": round_index,
                        "calls": sum(b["calls"] for r in rows
                                     for b in r["advocacy_cost"].get("by_round", [])
                                     if b["round"] == round_index),
                        "latency_seconds": sum(b["latency_seconds"] for r in rows
                                               for b in r["advocacy_cost"].get("by_round", [])
                                               if b["round"] == round_index),
                        "output_tokens": sum(b["output_tokens"] for r in rows
                                             for b in r["advocacy_cost"].get("by_round", [])
                                             if b["round"] == round_index),
                    }
                    for round_index in range(args.rounds)
                ],
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
                "bootstrap_accuracy_delta": bootstrap_accuracy_delta(
                    before, after, covered_golds, args.order_seed
                ),
                "boundary_transitions": boundary_transition_counts(before, after),
            }

    save_json(items_path, rows)
    save_json(prefix.with_suffix(".summary.json"), summary)
    save_json(prefix.with_suffix(".config.json"), vars(args))
    prefix.with_suffix(".qualitative.md").write_text(
        qualitative_report(rows, STANCE_LABELS, args.qualitative_per_cell, summary),
        encoding="utf-8",
    )
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
    runtime = summary["runtime"]
    print(f"wall clock={runtime['wall_clock_seconds'] / 3600:.2f}h  "
          f"generated={runtime['items_generated_this_run']}  "
          f"reused={runtime['items_reused_from_resume']}")
    if args.reuse_advocacy:
        print("  advocacy per-round costs below are inherited from the reused run; "
              "this pass generated none of them")
    for block in summary["cost"]["advocacy"]["by_round"]:
        print(f"  round {block['round']}: {block['calls']} calls  "
              f"{block['latency_seconds'] / 60:.1f}m  "
              f"{block['output_tokens']} output tokens")
    print(f"  judge: {summary['cost']['judge']['calls']} calls  "
          f"{summary['cost']['judge']['latency_seconds'] / 60:.1f}m  "
          f"retries={summary['judge_diagnostics']['judge_retries']}")
    diag = summary["judge_diagnostics"]
    test = diag["winning_candidate_position_test"]
    print(f"judge picked candidate position: {diag['winning_candidate_position']}"
          + (f"  chi2={test['chi_square']:.2f} df={test['df']} p={test['p_value']:.4f}"
             if test.get("p_value") is not None else ""))
    print(f"judge parse errors={diag['judge_parse_errors']}  "
          f"labels salvaged from malformed output={diag['label_recovered_from_text']}")
    print(f"judge output: median {diag['judge_output_chars']['median']} chars  "
          f"bare verdicts (<60 chars) {diag['judge_output_chars']['under_60_chars']}/{len(rows)}"
          f"   <- high means the judge skipped the discussion")
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
        # a repair step keeps what the baseline got right and fixes some of what
        # it got wrong; these two rates being equal means it is replacing the
        # baseline rather than adjudicating it, and inverted means anti-correlated
        print(f"   kept {t['correct_preservation_rate']:.1%} of "
              f"{comparison['method']}'s correct answers, "
              f"fixed {t['error_correction_rate']:.1%} of its errors")
    print(f"[saved] {prefix}.*  (items / summary / config / csv / qualitative.md)")


if __name__ == "__main__":
    main()

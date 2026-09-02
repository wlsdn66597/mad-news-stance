"""Run four journalism-segment agents and combine their final labels."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import load_model, model_context_window  # noqa: E402
from src.metrics import LABELS_DEFAULT, format_report  # noqa: E402
from src.segment_debate import SEGMENT_ORDER, run_segment_debate  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2_exaone_stance_minimal_en.yaml")
    parser.add_argument("--model", default="exaone", choices=["qwen", "exaone"])
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-rounds", type=int, default=4)
    parser.add_argument(
        "--decision-rule", choices=["majority", "aggregator"], default="majority",
        help="majority uses no final model call; aggregator reproduces the earlier experiment",
    )
    parser.add_argument(
        "--aggregator-temperature", type=float, default=None,
        help="defaults to --temperature, keeping the stance-generating call comparable",
    )
    parser.add_argument("--aggregator-max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def generation_seed(run_seed, namespace, item_id):
    token = f"{run_seed}\0stance\0{namespace}\0{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def save(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["models"][args.model]
    n = args.n or cfg["n"]
    split = args.split or cfg["split"]
    profile = cfg.get("prompt_profile", "stance_minimal_en")
    data_seed = cfg["seed"] if args.data_seed is None else args.data_seed
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    temperature = (
        cfg["sampling"]["temperature"]
        if args.temperature is None else args.temperature
    )
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    enable_thinking = {"on": True, "off": False}.get(
        args.enable_thinking, False if args.model == "qwen" else None
    )
    aggregator_temperature = (
        temperature
        if args.aggregator_temperature is None else args.aggregator_temperature
    )

    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=n, seed=data_seed)
    output = Path(args.output) if args.output else Path(
        f"results/phase2/stance_{args.model}_{split}_n{n}_{profile}_"
        f"d{data_seed}_s{run_seed}_segment-agents-{args.decision_rule}_"
        f"a4_r{args.n_rounds}"
        f"{f'_{args.tag}' if args.tag else ''}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}
    done = results.setdefault("segment_debate", {})
    aggregator_calls = int(args.decision_rule == "aggregator")
    calls_per_item = len(SEGMENT_ORDER) * args.n_rounds + aggregator_calls
    results["_meta"] = {
        "method": f"segment_debate_{args.decision_rule}",
        "config": args.config,
        "model_id": model_cfg["id"],
        "load_in_4bit": bool(cfg["load_in_4bit"]),
        "prompt_profile": profile,
        "split": split,
        "n": n,
        "data_seed": data_seed,
        "run_seed": run_seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "n_agents": len(SEGMENT_ORDER),
        "segment_order": list(SEGMENT_ORDER),
        "n_rounds": args.n_rounds,
        "decision_rule": args.decision_rule,
        "tie_rule": "legacy_first_valid" if args.decision_rule == "majority" else None,
        "tie_seed": run_seed if args.decision_rule == "majority" else None,
        "aggregator_temperature": (
            aggregator_temperature if args.decision_rule == "aggregator" else None
        ),
        "aggregator_max_new_tokens": (
            args.aggregator_max_new_tokens if args.decision_rule == "aggregator" else None
        ),
        "enable_thinking": enable_thinking,
        "calls_per_item": calls_per_item,
    }

    print(
        f"[data] split={split} n={len(items)} [method] segment agents "
        f"a={len(SEGMENT_ORDER)} r={args.n_rounds} decision={args.decision_rule} "
        f"calls/item={calls_per_item}", flush=True
    )
    set_seed(run_seed)
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    results["_meta"]["context_window"] = model_context_window(model, tokenizer)
    start = time.time()
    for index, item in enumerate(items, 1):
        key = str(item["id"])
        if key in done:
            continue
        item_seed = generation_seed(run_seed, "segment_debate", key)
        set_seed(item_seed)
        result = run_segment_debate(
            model,
            tokenizer,
            task,
            item,
            max_new_tokens=max_new_tokens,
            n_rounds=args.n_rounds,
            temperature=temperature,
            aggregator_temperature=aggregator_temperature,
            aggregator_max_new_tokens=args.aggregator_max_new_tokens,
            enable_thinking=enable_thinking,
            decision_rule=args.decision_rule,
            tie_seed=run_seed,
        )
        result.update({
            "gold": item["gold"],
            "correct": bool(task.correct(result["pred"], item)),
            "generation_seed": item_seed,
            "issue": item["issue"],
            "headline": item.get("headline"),
            "genre": item.get("genre"),
        })
        done[key] = result
        save(output, results)
        print(f"[{index}/{len(items)}] item={key} pred={result['pred']} gold={item['gold']}", flush=True)

    preds = [value["pred"] for value in done.values()]
    golds = [value["gold"] for value in done.values()]
    print(format_report(preds, golds, LABELS_DEFAULT))
    print(f"[done] {len(done)} items in {time.time() - start:.0f}s")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()

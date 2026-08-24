"""Choose three article-specific roles, then run the four-round debate."""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import load_model, model_context_window  # noqa: E402
from src.metrics import LABELS_DEFAULT, format_report  # noqa: E402
from src.role_planner import PLANNER_MODES, ROLE_LIBRARIES, run_planner_debate  # noqa: E402
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
    parser.add_argument("--role-pool", choices=sorted(ROLE_LIBRARIES), default="journalism")
    parser.add_argument("--planner-mode", choices=PLANNER_MODES, default="free3")
    parser.add_argument("--planner-temperature", type=float, default=0.0)
    parser.add_argument("--planner-max-new-tokens", type=int, default=128)
    parser.add_argument("--debate-protocol", default="reasoned_exchange_full")
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
    temperature = cfg["sampling"]["temperature"] if args.temperature is None else args.temperature
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    enable_thinking = {"on": True, "off": False}.get(
        args.enable_thinking, False if args.model == "qwen" else None
    )

    task = Stance(cfg["data_path"], prompt_profile=profile)
    if args.debate_protocol not in task.debate_protocols:
        choices = ", ".join(sorted(task.debate_protocols))
        raise ValueError(f"profile {profile} offers these protocols: {choices}")
    templates = task.debate_protocols[args.debate_protocol]
    if len(templates) not in (1, max(1, args.n_rounds - 1)):
        raise ValueError(
            f"protocol {args.debate_protocol} has {len(templates)} templates; "
            f"{args.n_rounds} rounds need one or {args.n_rounds - 1}"
        )
    items = task.load(split=split, n=n, seed=data_seed)
    planner_mode_tag = "" if args.planner_mode == "free3" else f"-{args.planner_mode}"
    output = Path(args.output) if args.output else Path(
        f"results/phase2/stance_{args.model}_{split}_n{n}_{profile}_"
        f"d{data_seed}_s{run_seed}_planner-{args.role_pool}{planner_mode_tag}_"
        f"a3_r{args.n_rounds}_"
        f"{args.debate_protocol}{f'_{args.tag}' if args.tag else ''}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}
    done = results.setdefault("planner_debate", {})
    results["_meta"] = {
        "method": "planner_debate",
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
        "n_agents": 3,
        "n_rounds": args.n_rounds,
        "role_pool": args.role_pool,
        "planner_mode": args.planner_mode,
        "available_roles": list(ROLE_LIBRARIES[args.role_pool]),
        "planner_temperature": args.planner_temperature,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "debate_protocol": args.debate_protocol,
        "enable_thinking": enable_thinking,
        "calls_per_item": 1 + 3 * args.n_rounds,
    }

    print(
        f"[data] split={split} n={len(items)} [method] planner={args.role_pool}/"
        f"{args.planner_mode} "
        f"a=3 r={args.n_rounds} calls/item={1 + 3 * args.n_rounds}", flush=True
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
        item_seed = generation_seed(
            run_seed, f"planner_{args.role_pool}_{args.planner_mode}", key
        )
        set_seed(item_seed)
        result = run_planner_debate(
            model,
            tokenizer,
            task,
            item,
            max_new_tokens=max_new_tokens,
            n_rounds=args.n_rounds,
            temperature=temperature,
            planner_temperature=args.planner_temperature,
            planner_max_new_tokens=args.planner_max_new_tokens,
            role_pool=args.role_pool,
            planner_mode=args.planner_mode,
            debate_protocol=args.debate_protocol,
            enable_thinking=enable_thinking,
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
        print(
            f"[{index}/{len(items)}] item={key} roles="
            f"{','.join(result['selected_roles'])} pred={result['pred']} gold={item['gold']}",
            flush=True,
        )

    preds = [value["pred"] for value in done.values()]
    golds = [value["gold"] for value in done.values()]
    combinations = Counter(tuple(value["selected_roles"]) for value in done.values())
    fallbacks = sum(bool(value["planner"]["fallback"]) for value in done.values())
    print(format_report(preds, golds, LABELS_DEFAULT))
    print(f"[planner] fallback={fallbacks}/{len(done)} role_combinations={dict(combinations)}")
    print(f"[done] {len(done)} items in {time.time() - start:.0f}s")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()

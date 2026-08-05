"""Run K-News-Stance baselines and debate variants with versioned prompts.

With shared Round 0 enabled, majority, direct debate, and memory debate reuse
the exact same initial answers for paired communication comparisons.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import methods  # noqa: E402
from src.llm import load_model, model_context_window  # noqa: E402
from src.metrics import LABELS_DEFAULT, format_report  # noqa: E402
from src.prompts.stance import PROFILES  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402

METHOD_FNS = {
    "single": methods.run_single,
    "vanilla": methods.run_vanilla,
    "cot": methods.run_cot,
    "majority": methods.run_majority,
    "debate": methods.run_debate,
    "debate_memory": methods.run_debate_memory,
}
SHARED_METHODS = {"single", "majority", "debate", "debate_memory"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2.yaml")
    parser.add_argument("--model", default="qwen", choices=["qwen", "exaone"])
    parser.add_argument("--methods", default="vanilla,cot,majority,debate")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--prompt-profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--memory-max-new-tokens", type=int, default=None)
    parser.add_argument("--memory-temperature", type=float, default=None)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def choose(argument, config, key, default):
    return argument if argument is not None else config.get(key, default)


def generation_seed(run_seed, task_name, namespace, item_id):
    token = f"{run_seed}\0{task_name}\0{namespace}\0{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def method_kwargs(method, cfg, args, temperature, initial_answers=None):
    kwargs = {"temperature": temperature}
    n_agents = args.n_agents or cfg["debate"]["n_agents"]
    if method == "single" and initial_answers is not None:
        kwargs["initial_answers"] = initial_answers
    if method == "majority":
        kwargs["k"] = args.k or cfg["majority"]["k"]
        if initial_answers is not None:
            kwargs["initial_answers"] = initial_answers
    if method in {"debate", "debate_memory"}:
        kwargs["n_agents"] = n_agents
        kwargs["n_rounds"] = args.n_rounds or cfg["debate"]["n_rounds"]
        if initial_answers is not None:
            kwargs["initial_answers"] = initial_answers
    if method == "debate_memory":
        memory_cfg = cfg.get("memory", {})
        kwargs["memory_max_new_tokens"] = choose(
            args.memory_max_new_tokens, memory_cfg, "max_new_tokens", 384
        )
        kwargs["memory_temperature"] = choose(
            args.memory_temperature, memory_cfg, "temperature", 0.0
        )
    return kwargs


def save(path, results):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["models"][args.model]

    method_list = [
        method.strip() for method in args.methods.split(",") if method.strip()
    ]
    unknown = sorted(set(method_list) - set(METHOD_FNS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")

    n = args.n or cfg["n"]
    split = args.split or cfg["split"]
    profile = args.prompt_profile or cfg.get("prompt_profile", "legacy_ko")
    data_seed = cfg["seed"] if args.data_seed is None else args.data_seed
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg["sampling"]["temperature"]
    )
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    n_agents = args.n_agents or cfg["debate"]["n_agents"]
    n_rounds = args.n_rounds or cfg["debate"]["n_rounds"]
    share_round0 = bool(cfg.get("share_round0", False))
    if share_round0 and (args.k or cfg["majority"]["k"]) != n_agents:
        raise ValueError("shared Round 0 requires majority.k == debate.n_agents")

    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=n, seed=data_seed)
    system_prompt = model_cfg.get("system_prompt")
    enable_thinking = False if args.model == "qwen" else None

    print(
        f"[cfg] profile={profile} data_seed={data_seed} run_seed={run_seed} "
        f"temperature={temperature} agents={n_agents} "
        f"load_in_4bit={bool(cfg['load_in_4bit'])}"
    )
    print(f"[data] stance split={split} n={len(items)}")

    set_seed(run_seed)
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    context_window = model_context_window(model, tokenizer)
    print(
        f"[tokens] max_new_tokens={max_new_tokens} context_window={context_window}"
    )

    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(
        f"results/phase2/stance_{args.model}_{split}_n{n}_{profile}_"
        f"d{data_seed}_s{run_seed}_a{n_agents}_r{n_rounds}{tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}

    results["_meta"] = {
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
        "context_window": context_window,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "share_round0": share_round0,
    }

    shared_needed = share_round0 and bool(set(method_list) & SHARED_METHODS)
    shared = results.setdefault("_shared_round0", {})
    if shared_needed:
        print("[shared-round0] generating/reusing paired initial answers")
        for item in items:
            key = str(item["id"])
            if key in shared:
                continue
            item_seed = generation_seed(run_seed, task.name, "shared_round0", key)
            set_seed(item_seed)
            initial = methods.generate_initial_answers(
                model, tokenizer, task, item, system_prompt, max_new_tokens,
                n_agents=n_agents, temperature=temperature,
                enable_thinking=enable_thinking,
            )
            initial["generation_seed"] = item_seed
            shared[key] = initial
            save(out_path, results)

    for method in method_list:
        function = METHOD_FNS[method]
        done = results.setdefault(method, {})
        start = time.time()
        for item in items:
            key = str(item["id"])
            if key in done:
                continue
            initial_answers = None
            if shared_needed and method in SHARED_METHODS:
                initial_answers = shared[key]["raw"]

            namespace = (
                "paired_debate_update"
                if shared_needed and method in {"debate", "debate_memory"}
                else method
            )
            item_seed = generation_seed(run_seed, task.name, namespace, key)
            set_seed(item_seed)
            result = function(
                model, tokenizer, task, item, system_prompt, max_new_tokens,
                enable_thinking=enable_thinking,
                **method_kwargs(
                    method, cfg, args, temperature,
                    initial_answers=initial_answers,
                ),
            )
            result.update(
                {
                    "gold": item["gold"],
                    "correct": bool(task.correct(result["pred"], item)),
                    "generation_seed": item_seed,
                    "issue": item["issue"],
                    "genre": item.get("genre"),
                    "prompt_profile": profile,
                    "round0_source": (
                        "shared" if initial_answers is not None else "generated"
                    ),
                }
            )
            done[key] = result
            save(out_path, results)
        print(f"[{method:14s}] done {len(done)} items {time.time() - start:.0f}s")

    print("\n===== PHASE 2 REPORT =====")
    print(f"stance model={args.model} split={split} n={n} profile={profile}")
    for method in method_list:
        values = results.get(method, {})
        if not values:
            continue
        preds = [value["pred"] for value in values.values()]
        golds = [value["gold"] for value in values.values()]
        print(f"\n### {method}")
        print(format_report(preds, golds, LABELS_DEFAULT))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

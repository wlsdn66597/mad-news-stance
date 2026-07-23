"""Phase 1 재현 실행기.

한 번 실행으로 (task, model) 에 대해 여러 method 를 같은 서브셋에 돌려 정확도를 비교한다.
결과는 results/phase1/{task}_{model}_n{n}.json 에 저장되고, 재실행 시 이미 끝난 항목은 건너뛴다.

예:
  python scripts/run_phase1.py --task gsm8k
  python scripts/run_phase1.py --task mmlu --n 10 --methods vanilla,debate
  python scripts/run_phase1.py --task gsm8k --methods debate,debate_memory --n-agents 5
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
from src.llm import load_model          # noqa: E402
from src import methods                 # noqa: E402
from src.tasks.gsm8k import GSM8K        # noqa: E402
from src.tasks.mmlu import MMLU          # noqa: E402

TASKS = {"gsm8k": GSM8K, "mmlu": MMLU}
PAPER_METHODS = {
    "gsm8k": ["single", "reflection", "majority", "debate"],
    "mmlu": ["single", "reflection", "debate"],
}
METHOD_FNS = {
    "vanilla": methods.run_vanilla,
    "cot": methods.run_cot,
    "single": methods.run_single,
    "reflection": methods.run_reflection,
    "majority": methods.run_majority,
    "debate": methods.run_debate,
    "debate_memory": methods.run_debate_memory,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1.yaml")
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--model", default="qwen", choices=["qwen", "qwen4", "qwen8", "exaone"])
    parser.add_argument(
        "--methods",
        default="paper",
        help=(
            "'paper' selects the published comparison set per task: "
            "GSM8K=single,reflection,majority,debate; "
            "MMLU=single,reflection,debate."
        ),
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--sampling-protocol",
        choices=["paper", "uniform"],
        default=None,
        help="paper reproduces the published task sampling algorithm; uniform preserves the legacy loader.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="문제 서브셋 추출 seed. 반복 간 같은 문제를 비교하려면 고정한다.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="LLM 생성 seed. 반복실험에서는 실행마다 다르게 지정한다.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--memory-max-new-tokens",
        type=int,
        default=None,
        help="shared memory 요약의 생성 토큰 상한",
    )
    parser.add_argument(
        "--memory-temperature",
        type=float,
        default=None,
        help="shared memory 요약 생성 온도",
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _value_or_config(argument_value, config, key, default):
    return argument_value if argument_value is not None else config.get(key, default)


def method_kwargs(method, cfg, args, temperature):
    kwargs = {"temperature": temperature}
    if method == "majority":
        kwargs["k"] = args.k or cfg["majority"]["k"]
        kwargs["initial_style"] = "paper"
    if method == "debate":
        kwargs["initial_style"] = "paper"
    if method in {"debate", "debate_memory"}:
        kwargs["n_agents"] = args.n_agents or cfg["debate"]["n_agents"]
        kwargs["n_rounds"] = args.n_rounds or cfg["debate"]["n_rounds"]
    if method == "debate_memory":
        memory_cfg = cfg.get("memory", {})
        kwargs["memory_max_new_tokens"] = _value_or_config(
            args.memory_max_new_tokens, memory_cfg, "max_new_tokens", 256
        )
        kwargs["memory_temperature"] = _value_or_config(
            args.memory_temperature, memory_cfg, "temperature", 0.0
        )
    return kwargs


def generation_seed(run_seed, task_name, method, item_id):
    """재개 여부나 method 실행 순서에 영향받지 않는 문항별 seed를 만든다."""
    token = f"{run_seed}\0{task_name}\0{method}\0{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["models"][args.model]
    n = args.n or cfg["n"]
    split = args.split or cfg["split"]
    sampling_protocol = args.sampling_protocol or cfg.get("sampling_protocol", "uniform")
    requested_methods = [
        method.strip() for method in args.methods.split(",") if method.strip()
    ]
    if requested_methods == ["paper"]:
        method_list = PAPER_METHODS[args.task]
    else:
        if "paper" in requested_methods:
            raise ValueError("'paper' cannot be combined with explicit method names")
        method_list = requested_methods
    unknown = sorted(set(method_list) - set(METHOD_FNS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")

    data_seed = cfg["seed"] if args.data_seed is None else args.data_seed
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    enable_thinking = False if args.model.startswith("qwen") else None
    system_prompt = model_cfg.get("system_prompt")
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg["sampling"]["temperature"]
    )
    print(
        f"[cfg] data_seed={data_seed} run_seed={run_seed} "
        f"sampling_protocol={sampling_protocol} temperature={temperature} max_new_tokens={max_new_tokens}"
    )

    task = TASKS[args.task]()
    items = task.load(split=split, n=n, seed=data_seed, sampling_protocol=sampling_protocol)
    print(f"[data] {task.name} split={split} n={len(items)} sampling={sampling_protocol}")

    set_seed(run_seed)
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(f"results/phase1/{task.name}_{args.model}_n{n}{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}

    for method in method_list:
        function = METHOD_FNS[method]
        kwargs = method_kwargs(method, cfg, args, temperature)
        done = results.setdefault(method, {})
        start = time.time()
        for item in items:
            key = str(item["id"])
            if key in done:
                continue

            item_seed = generation_seed(run_seed, task.name, method, key)
            set_seed(item_seed)
            result = function(
                model,
                tokenizer,
                task,
                item,
                system_prompt,
                max_new_tokens,
                enable_thinking=enable_thinking,
                **kwargs,
            )
            result["gold"] = item["gold"]
            result["item"] = {
                field: item[field]
                for field in ("id", "draw_index", "source_index", "subject")
                if field in item
            }
            result["correct"] = bool(task.correct(result["pred"], item))
            result["generation_seed"] = item_seed
            done[key] = result
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)

        n_done = len(done)
        n_correct = sum(value["correct"] for value in done.values())
        accuracy = n_correct / n_done if n_done else 0.0
        print(
            f"[{method:14s}] acc={accuracy:.3f} "
            f"({n_correct}/{n_done}) {time.time() - start:.0f}s"
        )

    print("\n===== SUMMARY =====")
    print(
        f"task={task.name} model={args.model} n={n} "
        f"data_seed={data_seed} run_seed={run_seed} sampling={sampling_protocol}"
    )
    for method in method_list:
        values = results.get(method, {})
        if values:
            accuracy = sum(value["correct"] for value in values.values()) / len(values)
            print(f"  {method:14s} {accuracy:.3f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

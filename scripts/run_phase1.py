"""Phase 1 재현 실행기.

한 번 실행으로 (task, model) 에 대해 여러 method 를 같은 서브셋에 돌려 정확도를 비교한다.
결과는 results/phase1/{task}_{model}_n{n}.json 에 저장되고, 재실행 시 이미 끝난 항목은 건너뛴다(이어하기).

예:
  python scripts/run_phase1.py --task gsm8k                 # 전체 method
  python scripts/run_phase1.py --task mmlu --n 10 --methods vanilla,debate
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import load_model          # noqa: E402
from src import methods                 # noqa: E402
from src.tasks.gsm8k import GSM8K        # noqa: E402
from src.tasks.mmlu import MMLU          # noqa: E402

TASKS = {"gsm8k": GSM8K, "mmlu": MMLU}
METHOD_FNS = {
    "vanilla": methods.run_vanilla,
    "cot": methods.run_cot,
    "majority": methods.run_majority,
    "debate": methods.run_debate,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase1.yaml")
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", default="qwen", choices=["qwen", "exaone"])
    ap.add_argument("--methods", default="vanilla,cot,majority,debate")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default=None)
    # 논문 세팅 override (재현 실험용)
    ap.add_argument("--temperature", type=float, default=None, help="샘플링 온도 override (논문 기본 ~1.0)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="생성 토큰 상한 override")
    ap.add_argument("--n-rounds", type=int, default=None, help="debate 라운드 수 override")
    ap.add_argument("--n-agents", type=int, default=None, help="debate 에이전트 수 override")
    ap.add_argument("--tag", default="", help="결과 파일명 접미사 (변형 실험 구분)")
    return ap.parse_args()


def method_kwargs(method, cfg, args, temperature):
    # 논문 정렬: 모든 method 가 동일 temperature 사용
    kw = {"temperature": temperature}
    if method == "majority":
        kw["k"] = cfg["majority"]["k"]
    if method == "debate":
        kw["n_agents"] = args.n_agents or cfg["debate"]["n_agents"]     # CLI override
        kw["n_rounds"] = args.n_rounds or cfg["debate"]["n_rounds"]     # CLI override
    return kw


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mc = cfg["models"][args.model]
    n = args.n or cfg["n"]
    split = args.split or cfg["split"]
    method_list = args.methods.split(",")
    enable_thinking = False if args.model == "qwen" else None  # Qwen: non-thinking
    sysp = mc.get("system_prompt")
    mnt = args.max_new_tokens or cfg["max_new_tokens"]
    temperature = args.temperature if args.temperature is not None else cfg["sampling"]["temperature"]
    print(f"[cfg] temperature={temperature} max_new_tokens={mnt}")

    task = TASKS[args.task]()
    items = task.load(split=split, n=n, seed=cfg["seed"])
    print(f"[data] {task.name} split={split} n={len(items)}")

    model, tok = load_model(
        mc["id"], load_in_4bit=cfg["load_in_4bit"], gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"], trust_remote_code=mc.get("trust_remote_code", False),
    )

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(f"results/phase1/{task.name}_{args.model}_n{n}{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}

    for method in method_list:
        fn = METHOD_FNS[method]
        kw = method_kwargs(method, cfg, args, temperature)
        done = results.setdefault(method, {})
        t0 = time.time()
        for it in items:
            key = str(it["id"])
            if key in done:                       # 이어하기
                continue
            r = fn(model, tok, task, it, sysp, mnt, enable_thinking=enable_thinking, **kw)
            r["gold"] = it["gold"]
            r["correct"] = bool(task.correct(r["pred"], it))
            done[key] = r
            json.dump(results, open(out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)     # 매 항목 저장(크래시 대비)
        n_done = len(done)
        acc = sum(v["correct"] for v in done.values()) / n_done if n_done else 0.0
        print(f"[{method:8s}] acc={acc:.3f}  ({sum(v['correct'] for v in done.values())}/{n_done})  {time.time()-t0:.0f}s")

    print("\n===== SUMMARY =====")
    print(f"task={task.name}  model={args.model}  n={n}")
    for method in method_list:
        d = results.get(method, {})
        if d:
            acc = sum(v["correct"] for v in d.values()) / len(d)
            print(f"  {method:10s} {acc:.3f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

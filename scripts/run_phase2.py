"""Phase 2 실행기 — K-News-Stance 에 단일 LLM vs Multi-Agent Debate 비교.

Phase 1 과 동일한 method 루프 + 이어하기. 끝에 stance 지표(accuracy/macro-F1/confusion) 출력.

예:
  python scripts/run_phase2.py --n 10 --methods vanilla,debate     # 빠른 확인
  python scripts/run_phase2.py --n 30                              # 전체 method
  python scripts/run_phase2.py --split test --n 200                # 본실험
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import load_model                    # noqa: E402
from src import methods                           # noqa: E402
from src.tasks.stance import Stance               # noqa: E402
from src.metrics import format_report, LABELS_DEFAULT  # noqa: E402

METHOD_FNS = {
    "vanilla": methods.run_vanilla,
    "cot": methods.run_cot,
    "majority": methods.run_majority,
    "debate": methods.run_debate,
}


def method_kwargs(method, cfg):
    # 논문 정렬: 모든 method 가 동일 temperature 사용
    kw = {"temperature": cfg["sampling"]["temperature"]}
    if method == "majority":
        kw["k"] = cfg["majority"]["k"]
    if method == "debate":
        kw["n_agents"] = cfg["debate"]["n_agents"]
        kw["n_rounds"] = cfg["debate"]["n_rounds"]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase2.yaml")
    ap.add_argument("--model", default="qwen", choices=["qwen", "exaone"])
    ap.add_argument("--methods", default="vanilla,cot,majority,debate")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mc = cfg["models"][args.model]
    n = args.n or cfg["n"]
    split = args.split or cfg["split"]
    method_list = args.methods.split(",")
    enable_thinking = False if args.model == "qwen" else None
    sysp = mc.get("system_prompt")
    mnt = cfg["max_new_tokens"]

    task = Stance(cfg["data_path"])
    items = task.load(split=split, n=n, seed=cfg["seed"])
    print(f"[data] stance split={split} n={len(items)}")

    model, tok = load_model(
        mc["id"], load_in_4bit=cfg["load_in_4bit"], gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"], trust_remote_code=mc.get("trust_remote_code", False),
    )

    out_path = Path(f"results/phase2/stance_{args.model}_{split}_n{n}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}

    for method in method_list:
        fn = METHOD_FNS[method]
        kw = method_kwargs(method, cfg)
        done = results.setdefault(method, {})
        t0 = time.time()
        for it in items:
            key = str(it["id"])
            if key in done:
                continue
            r = fn(model, tok, task, it, sysp, mnt, enable_thinking=enable_thinking, **kw)
            r["gold"] = it["gold"]
            r["correct"] = bool(task.correct(r["pred"], it))
            done[key] = r
            json.dump(results, open(out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        print(f"[{method:8s}] done {len(done)} items  {time.time()-t0:.0f}s")

    print("\n===== PHASE 2 REPORT =====")
    print(f"stance  model={args.model}  split={split}  n={n}")
    for method in method_list:
        d = results.get(method, {})
        if not d:
            continue
        preds = [v["pred"] for v in d.values()]
        golds = [v["gold"] for v in d.values()]
        print(f"\n### {method}")
        print(format_report(preds, golds, LABELS_DEFAULT))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

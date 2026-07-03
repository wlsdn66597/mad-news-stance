"""스모크 테스트: 모델 로딩 → 단일 답변 → 1문항 debate trace 저장.

사용 예:
  python scripts/smoke_test.py                    # EXAONE-2.4B (기본)
  python scripts/smoke_test.py --model qwen       # Qwen3-4B, thinking on(기본)
  python scripts/smoke_test.py --model qwen --no-think
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# repo 루트를 import 경로에 추가 (python scripts/smoke_test.py 로 실행 가능하게)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import load_model, build_messages, chat, strip_think  # noqa: E402
from src.debate import run_debate  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/smoke.yaml")
    ap.add_argument("--model", default="exaone", choices=["exaone", "qwen"])
    ap.add_argument("--think", dest="think", action="store_true", help="Qwen thinking 모드 ON")
    ap.add_argument("--no-think", dest="think", action="store_false", help="Qwen thinking 모드 OFF")
    ap.set_defaults(think=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mc = cfg["models"][args.model]

    # Qwen 만 thinking 토글 적용 (미지정 시 ON)
    enable_thinking = None
    if args.model == "qwen":
        enable_thinking = True if args.think is None else args.think

    print(f"[load] {mc['id']}  (4bit={cfg['load_in_4bit']}, mem_fraction={cfg['mem_fraction']})")
    t0 = time.time()
    model, tokenizer = load_model(
        mc["id"],
        load_in_4bit=cfg["load_in_4bit"],
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=mc.get("trust_remote_code", False),
    )
    print(f"[load] done in {time.time() - t0:.1f}s")

    question = cfg["sample_question"]
    system_prompt = mc.get("system_prompt")
    temperature = cfg["debate"]["temperature"]

    # ── Part A: 단일 답변 ───────────────────────────────
    print("\n===== PART A: single answer =====")
    single = chat(
        model, tokenizer, build_messages(question, system_prompt=system_prompt),
        max_new_tokens=cfg["max_new_tokens"], temperature=temperature,
        enable_thinking=enable_thinking,
    )
    print(strip_think(single))

    # ── Part B: 1문항 debate ───────────────────────────
    print("\n===== PART B: debate (n_agents x n_rounds) =====")
    d = cfg["debate"]
    trace = run_debate(
        model, tokenizer, question, system_prompt=system_prompt,
        n_agents=d["n_agents"], n_rounds=d["n_rounds"],
        max_new_tokens=cfg["max_new_tokens"], temperature=d["temperature"],
        enable_thinking=enable_thinking,
    )
    for r, row in enumerate(trace["answers_by_round"]):
        print(f"\n-- round {r} --")
        for a, ans in enumerate(row):
            print(f"  agent{a}: {ans.replace(chr(10), ' ')[:160]}")

    out_dir = Path("results/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "think" if enable_thinking else "nothink"
    out_path = out_dir / f"trace_{args.model}_{tag}.json"
    json.dump(trace, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}")
    print("[ok] 스모크 완료 — 모델 로딩/생성/debate trace 저장이 모두 정상입니다.")


if __name__ == "__main__":
    main()

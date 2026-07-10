#!/usr/bin/env bash
# 논문 충실 재현: Multi-Agent (Majority) = k=3, temp 1.0, MMLU
# 비교 대상 debate(t10)가 n=50이라 여기도 n=50 유지(공정 비교).
# n을 바꾸려면 아래 N만 수정 — 단, debate도 같은 n으로 다시 돌려야 함.
set -e
N=50
cd "$(dirname "$0")/.."          # repo 루트로 이동

python scripts/run_phase1.py --task mmlu --n "$N" --model qwen   --temperature 1.0 --methods majority --tag t10_k3
python scripts/run_phase1.py --task mmlu --n "$N" --model exaone --temperature 1.0 --methods majority --tag t10_k3

echo "== done. report 보기: python scripts/report.py =="

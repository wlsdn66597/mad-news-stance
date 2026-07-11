#!/usr/bin/env bash
# 논문 충실 재현 확정: temp 1.0, majority k=3(기본), n=100, MMLU+GSM8K (전 method)
# 이미 끝난 항목은 스킵(이어하기)되므로 재실행해도 안전.
set -e
N=100
cd "$(dirname "$0")/.."
python scripts/run_phase1.py --task mmlu  --n "$N" --model qwen   --temperature 1.0 --tag faithful_n100
python scripts/run_phase1.py --task mmlu  --n "$N" --model exaone --temperature 1.0 --tag faithful_n100
python scripts/run_phase1.py --task gsm8k --n "$N" --model qwen   --temperature 1.0 --tag faithful_n100
python scripts/run_phase1.py --task gsm8k --n "$N" --model exaone --temperature 1.0 --tag faithful_n100
echo "== done. report: python scripts/report.py =="

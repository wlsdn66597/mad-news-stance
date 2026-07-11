#!/usr/bin/env bash
# 논문 충실 재현 확정: temp 1.0, majority k=3(기본), n=100, MMLU (전 method)
set -e
N=100
cd "$(dirname "$0")/.."
python scripts/run_phase1.py --task mmlu --n "$N" --model qwen   --temperature 1.0 --tag faithful_n100
python scripts/run_phase1.py --task mmlu --n "$N" --model exaone --temperature 1.0 --tag faithful_n100
echo "== done. report: python scripts/report.py =="

#!/usr/bin/env bash
# Phase 1 faithful repeated experiment: one model, n=100, three repeats.
set -euo pipefail

MODEL="${1:-}"
case "$MODEL" in
  qwen|qwen4|qwen8|exaone) ;;
  *)
    echo "usage: bash scripts/run_repeated_n100.sh {exaone|qwen4|qwen8}" >&2
    exit 2
    ;;
esac
shift

cd "$(dirname "$0")/.."
exec python scripts/run_phase1_repeated.py \
  --tasks gsm8k,mmlu \
  --models "$MODEL" \
  --methods paper \
  --n 100 \
  --repeats 3 \
  --data-seed 0 \
  --base-run-seed 1000 \
  --temperature 1.0 \
  --tag-prefix faithful_repeat_trace \
  "$@"

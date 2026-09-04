#!/usr/bin/env bash
# Qwen3-1.7B paper suite, paired with the existing Qwen3-4B/8B conditions.
#
# Runs and resumes, in order:
#   1) same-prompt single / majority / four-round full-exchange debate
#   2) selective judge on non-unanimous same-prompt debate items
#   3) F/S/W majority / four-round full-exchange debate
#   4) selective judge on non-unanimous F/S/W debate items
#
# Usage:
#   nohup bash scripts/run_qwen17_paper_suite.sh \
#     > logs/qwen17_paper_suite_s6000.log 2>&1 &
#
# Optional overrides:
#   RUN_SEED=6001 N=1001 bash scripts/run_qwen17_paper_suite.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2_qwen17_stance_minimal_en.yaml}"
MODEL="qwen"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
DATA_SEED="${DATA_SEED:-0}"
RUN_SEED="${RUN_SEED:-6000}"
ORDER_SEED="${ORDER_SEED:-8001}"
TAG="${TAG:-qwen17b}"
PROFILE="stance_minimal_en"
DATA="data/k-news-stance_nosegment.json"

BASE="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}_a3_r4_think-off_reasoned_exchange_full_${TAG}.json"
FSW="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}_a3_r4_personas_think-off_reasoned_exchange_full_${TAG}.json"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2 \
  "results/judge_qwen17_sameprompt_4r_full_s${RUN_SEED}" \
  "results/judge_qwen17_fsw_4r_full_s${RUN_SEED}"

run_step () {
  local title="$1"
  shift
  local started
  started=$(date +%s)
  echo "===== $title START $(date '+%F %T') ====="
  "$@"
  echo "===== $title COMPLETE $(( $(date +%s) - started )) sec ====="
}

echo "===== QWEN3-1.7B PAPER SUITE START $(date '+%F %T') ====="
echo "config=$CONFIG split=$SPLIT n=$N data_seed=$DATA_SEED run_seed=$RUN_SEED gpu=0"
echo "base_result=$BASE"
echo "fsw_result=$FSW"

run_step "1. SAME-PROMPT: SINGLE / MAJORITY / DEBATE 4R" \
  python scripts/run_phase2.py \
    --config "$CONFIG" --model "$MODEL" \
    --methods single,majority,debate \
    --split "$SPLIT" --n "$N" \
    --data-seed "$DATA_SEED" --run-seed "$RUN_SEED" \
    --n-agents 3 --n-rounds 4 \
    --enable-thinking off \
    --debate-protocol reasoned_exchange_full \
    --tag "$TAG"

run_step "2. SAME-PROMPT DEBATE 4R + SELECTIVE JUDGE" \
  python scripts/run_selective_judge.py \
    --input-result "$BASE" --data-path "$DATA" \
    --judge-model Qwen/Qwen3-1.7B \
    --judge-trigger non_unanimous \
    --judge-input-mode debate_trace --judge-rounds first \
    --judge-order-seed "$ORDER_SEED" --judge-temperature 0 \
    --judge-max-new-tokens 384 --judge-max-retries 1 \
    --output-dir "results/judge_qwen17_sameprompt_4r_full_s${RUN_SEED}"

run_step "3. F/S/W: MAJORITY / DEBATE 4R" \
  python scripts/run_phase2.py \
    --config "$CONFIG" --model "$MODEL" \
    --methods majority,debate --personas \
    --persona-mix mixed \
    --split "$SPLIT" --n "$N" \
    --data-seed "$DATA_SEED" --run-seed "$RUN_SEED" \
    --n-agents 3 --n-rounds 4 \
    --enable-thinking off \
    --debate-protocol reasoned_exchange_full \
    --tag "$TAG"

run_step "4. F/S/W DEBATE 4R + SELECTIVE JUDGE" \
  python scripts/run_selective_judge.py \
    --input-result "$FSW" --data-path "$DATA" \
    --judge-model Qwen/Qwen3-1.7B \
    --judge-trigger non_unanimous \
    --judge-input-mode debate_trace --judge-rounds first \
    --judge-order-seed "$ORDER_SEED" --judge-temperature 0 \
    --judge-max-new-tokens 384 --judge-max-retries 1 \
    --output-dir "results/judge_qwen17_fsw_4r_full_s${RUN_SEED}"

echo "===== QWEN3-1.7B PAPER SUITE COMPLETE $(date '+%F %T') ====="
echo "base_result=$BASE"
echo "fsw_result=$FSW"

#!/usr/bin/env bash
# Run two three-agent role controls once, followed by the paper's selective
# EXAONE judge:
#   1) Generic / Sourcing / Wording
#   2) Grounded Framing / Sourcing / Wording
#
# The script is resumable because both run_phase2.py and
# run_selective_judge.py reuse their saved outputs.
#
# Usage:
#   nohup bash scripts/run_generic_grounded_framing_ablation.sh \
#     > logs/generic_grounded_framing_s6000.log 2>&1 &
#
# Optional overrides:
#   RUN_SEED=6001 N=1001 bash scripts/run_generic_grounded_framing_ablation.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2.yaml}"
MODEL="exaone"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
PROFILE="${PROFILE:-stance_minimal_en}"
DATA_SEED="${DATA_SEED:-0}"
RUN_SEED="${RUN_SEED:-6000}"
ORDER_SEED="${ORDER_SEED:-8001}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
JUDGE_MODEL="${JUDGE_MODEL:-LGAI-EXAONE/EXAONE-4.0-1.2B}"

GENERIC_MIX="generic,sourcing,wording"
GROUNDED_MIX="grounded_framing,sourcing,wording"

BASE="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}_a3_r4_personas"
GENERIC_RESULT="${BASE}_mix-generic-sourcing-wording_reasoned_exchange_full.json"
GROUNDED_RESULT="${BASE}_mix-grounded_framing-sourcing-wording_reasoned_exchange_full.json"

GENERIC_JUDGE_DIR="results/judge_generic_sourcing_wording_4r_full_s${RUN_SEED}"
GROUNDED_JUDGE_DIR="results/judge_grounded_framing_sourcing_wording_4r_full_s${RUN_SEED}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2 "$GENERIC_JUDGE_DIR" "$GROUNDED_JUDGE_DIR"

run_step () {
  local title="$1"
  shift
  local started
  started=$(date +%s)
  echo "===== $title START $(date '+%F %T') ====="
  "$@"
  echo "===== $title COMPLETE $(( $(date +%s) - started )) sec ====="
}

run_debate () {
  local title="$1"
  local mix="$2"
  run_step "$title" \
    python scripts/run_phase2.py \
      --config "$CONFIG" --model "$MODEL" \
      --methods debate --personas --persona-mix "$mix" \
      --split "$SPLIT" --n "$N" \
      --prompt-profile "$PROFILE" \
      --data-seed "$DATA_SEED" --run-seed "$RUN_SEED" \
      --temperature "$TEMPERATURE" \
      --n-agents 3 --n-rounds 4 \
      --debate-protocol reasoned_exchange_full
}

run_judge () {
  local title="$1"
  local input_result="$2"
  local output_dir="$3"
  [ -f "$input_result" ] || {
    echo "missing debate result: $input_result" >&2
    exit 2
  }
  run_step "$title" \
    python scripts/run_selective_judge.py \
      --input-result "$input_result" --data-path "$DATA" \
      --judge-model "$JUDGE_MODEL" \
      --judge-trigger non_unanimous \
      --judge-input-mode debate_trace --judge-rounds first \
      --judge-order-seed "$ORDER_SEED" --judge-temperature 0 \
      --judge-max-new-tokens 384 --judge-max-retries 1 \
      --output-dir "$output_dir"
}

echo "===== GENERIC + GROUNDED FRAMING ABLATION START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED run_seed=$RUN_SEED"
echo "generic_result=$GENERIC_RESULT"
echo "grounded_result=$GROUNDED_RESULT"

run_debate "1. GENERIC / SOURCING / WORDING: DEBATE 4R" "$GENERIC_MIX"
run_judge "2. GENERIC / SOURCING / WORDING: SELECTIVE JUDGE" \
  "$GENERIC_RESULT" "$GENERIC_JUDGE_DIR"

run_debate "3. GROUNDED FRAMING / SOURCING / WORDING: DEBATE 4R" "$GROUNDED_MIX"
run_judge "4. GROUNDED FRAMING / SOURCING / WORDING: SELECTIVE JUDGE" \
  "$GROUNDED_RESULT" "$GROUNDED_JUDGE_DIR"

echo "===== GENERIC + GROUNDED FRAMING ABLATION COMPLETE $(date '+%F %T') ====="
echo "generic_result=$GENERIC_RESULT"
echo "generic_judge_dir=$GENERIC_JUDGE_DIR"
echo "grounded_result=$GROUNDED_RESULT"
echo "grounded_judge_dir=$GROUNDED_JUDGE_DIR"

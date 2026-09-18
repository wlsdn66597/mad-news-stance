#!/usr/bin/env bash
# Run the three new seed-6000 controls required after making the two-agent
# Sourcing/Wording pipeline the proposed NewsMAD model:
#
#   1) two-agent MAD (no persona prompts) + selective Judge
#   2) w/o Sourcing: Wording / Wording + selective Judge
#   3) w/o Wording:  Sourcing / Sourcing + selective Judge
#
# Every condition uses the paper settings: EXAONE-4.0-1.2B, the 1001-item test
# split, stance_minimal_en, temperature 1.0, four rounds of
# reasoned_exchange_full, and the existing non-unanimous Judge configuration.
# run_phase2.py and run_selective_judge.py save per item, so invoking this
# script again resumes an interrupted run instead of regenerating completed
# items.
#
# Usage:
#   nohup bash scripts/run_newsmad_final_controls.sh \
#     > logs/newsmad_final_controls_s6000.log 2>&1 &
#
# Optional smoke run:
#   N=10 bash scripts/run_newsmad_final_controls.sh
# Print every command without running a model:
#   DRY_RUN=1 bash scripts/run_newsmad_final_controls.sh
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
DRY_RUN="${DRY_RUN:-0}"

BASE="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}_a2_r4"
MAD_RESULT="${BASE}_reasoned_exchange_full.json"
NO_SOURCING_RESULT="${BASE}_personas_mix-wording-wording_reasoned_exchange_full.json"
NO_WORDING_RESULT="${BASE}_personas_mix-sourcing-sourcing_reasoned_exchange_full.json"

MAD_JUDGE_DIR="results/judge_mad_a2_4r_full_s${RUN_SEED}"
NO_SOURCING_JUDGE_DIR="results/judge_no_sourcing_ww_a2_4r_full_s${RUN_SEED}"
NO_WORDING_JUDGE_DIR="results/judge_no_wording_ss_a2_4r_full_s${RUN_SEED}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2 \
  "$MAD_JUDGE_DIR" "$NO_SOURCING_JUDGE_DIR" "$NO_WORDING_JUDGE_DIR"

run_step () {
  local title="$1"
  shift
  if [ "$DRY_RUN" = "1" ]; then
    echo "[plan] $title"
    printf '       %q ' "$@"
    echo
    return 0
  fi
  local started
  started=$(date +%s)
  echo "===== $title START $(date '+%F %T') ====="
  "$@"
  echo "===== $title COMPLETE $(( $(date +%s) - started )) sec ====="
}

run_debate () {
  local title="$1"
  local mix="$2"
  local args=(
    python scripts/run_phase2.py
    --config "$CONFIG" --model "$MODEL"
    --methods debate
    --split "$SPLIT" --n "$N"
    --prompt-profile "$PROFILE"
    --data-seed "$DATA_SEED" --run-seed "$RUN_SEED"
    --temperature "$TEMPERATURE"
    --n-agents 2 --n-rounds 4
    --debate-protocol reasoned_exchange_full
  )
  if [ -n "$mix" ]; then
    args+=(--personas --persona-mix "$mix")
  fi
  run_step "$title" "${args[@]}"
}

run_judge () {
  local title="$1"
  local input_result="$2"
  local output_dir="$3"
  [ "$DRY_RUN" = "1" ] || [ -f "$input_result" ] || {
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

print_summary () {
  local label="$1"
  local output_dir="$2"
  local summary
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  summary=$(find "$output_dir" -maxdepth 1 -type f -name '*.summary.json' -print -quit)
  if [ -z "$summary" ]; then
    echo "[summary] $label: no summary file in $output_dir"
    return 0
  fi
  python - "$label" "$summary" <<'PY'
import json
import sys

label, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
before = report["baseline_metrics"]
after = report["judge_metrics"]
trigger = report["trigger_subset"]
print(
    f"[summary] {label}: "
    f"debate ACC={before['accuracy']:.4f} F1={before['macro_f1']:.4f} | "
    f"+Judge ACC={after['accuracy']:.4f} F1={after['macro_f1']:.4f} | "
    f"judge={trigger['triggered_total']}/{report['items']} "
    f"({trigger['trigger_ratio']:.2%})"
)
PY
}

echo "===== NEWSMAD FINAL CONTROLS START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED run_seed=$RUN_SEED"
echo "All conditions: 2 agents, 4 rounds, reasoned_exchange_full, selective Judge"

run_debate "1. MAD (2 AGENTS, NO PERSONAS): DEBATE 4R" ""
run_judge "2. MAD (2 AGENTS): SELECTIVE JUDGE" \
  "$MAD_RESULT" "$MAD_JUDGE_DIR"

run_debate "3. W/O SOURCING (WORDING / WORDING): DEBATE 4R" \
  "wording,wording"
run_judge "4. W/O SOURCING (WORDING / WORDING): SELECTIVE JUDGE" \
  "$NO_SOURCING_RESULT" "$NO_SOURCING_JUDGE_DIR"

run_debate "5. W/O WORDING (SOURCING / SOURCING): DEBATE 4R" \
  "sourcing,sourcing"
run_judge "6. W/O WORDING (SOURCING / SOURCING): SELECTIVE JUDGE" \
  "$NO_WORDING_RESULT" "$NO_WORDING_JUDGE_DIR"

echo "===== FINAL METRICS ====="
print_summary "MAD a2 4R + Judge" "$MAD_JUDGE_DIR"
print_summary "w/o Sourcing (W/W) + Judge" "$NO_SOURCING_JUDGE_DIR"
print_summary "w/o Wording (S/S) + Judge" "$NO_WORDING_JUDGE_DIR"

echo "===== NEWSMAD FINAL CONTROLS COMPLETE $(date '+%F %T') ====="
echo "mad_result=$MAD_RESULT"
echo "mad_judge_dir=$MAD_JUDGE_DIR"
echo "no_sourcing_result=$NO_SOURCING_RESULT"
echo "no_sourcing_judge_dir=$NO_SOURCING_JUDGE_DIR"
echo "no_wording_result=$NO_WORDING_RESULT"
echo "no_wording_judge_dir=$NO_WORDING_JUDGE_DIR"

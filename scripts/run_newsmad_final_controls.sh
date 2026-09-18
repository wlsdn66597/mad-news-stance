#!/usr/bin/env bash
# Run the three new seed-6000 controls required after making the two-agent
# Sourcing/Wording pipeline the proposed NewsMAD model:
#
#   1) two-agent MAD (no persona prompts) + selective Judge
#   2) w/o Sourcing: one Wording agent only
#   3) w/o Wording:  one Sourcing agent only
#
# Every condition uses the paper settings: EXAONE-4.0-1.2B, the 1001-item test
# split, stance_minimal_en, temperature 1.0, and seed 6000.  The matched MAD
# baseline uses four rounds plus the existing non-unanimous Judge.  A one-agent
# role removal has no peer to debate and can never trigger a non-unanimous
# Judge, so it is evaluated once with the same initial CoT task prompt used at
# round 0 of the proposed model.
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

BASE="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}"
MAD_RESULT="${BASE}_a2_r4_reasoned_exchange_full.json"
NO_SOURCING_RESULT="${BASE}_a1_r1_personas_only-wording.json"
NO_WORDING_RESULT="${BASE}_a1_r1_personas_only-sourcing.json"

MAD_JUDGE_DIR="results/judge_mad_a2_4r_full_s${RUN_SEED}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2 "$MAD_JUDGE_DIR"

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

run_single_role () {
  local title="$1"
  local role="$2"
  run_step "$title" \
    python scripts/run_phase2.py \
      --config "$CONFIG" --model "$MODEL" \
      --methods cot --personas --persona-mix "$role" \
      --split "$SPLIT" --n "$N" \
      --prompt-profile "$PROFILE" \
      --data-seed "$DATA_SEED" --run-seed "$RUN_SEED" \
      --temperature "$TEMPERATURE" \
      --n-agents 1 --n-rounds 1
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

print_single_role_summary () {
  local label="$1"
  local result="$2"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  [ -f "$result" ] || {
    echo "[summary] $label: missing result $result"
    return 0
  }
  python - "$label" "$result" <<'PY'
import json
import sys

from src.consensus import classification_metrics

label, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
rows = list(result["cot"].values())
metrics = classification_metrics(
    [row["pred"] for row in rows],
    [row["gold"] for row in rows],
)
print(
    f"[summary] {label}: "
    f"ACC={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f} "
    f"items={len(rows)}"
)
PY
}

echo "===== NEWSMAD FINAL CONTROLS START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED run_seed=$RUN_SEED"
echo "MAD control: 2 agents, 4 rounds, reasoned_exchange_full, selective Judge"
echo "Role removals: one remaining role, one independent response, no Judge"

run_debate "1. MAD (2 AGENTS, NO PERSONAS): DEBATE 4R" ""
run_judge "2. MAD (2 AGENTS): SELECTIVE JUDGE" \
  "$MAD_RESULT" "$MAD_JUDGE_DIR"

run_single_role "3. W/O SOURCING: WORDING AGENT ONLY" "wording"
run_single_role "4. W/O WORDING: SOURCING AGENT ONLY" "sourcing"

echo "===== FINAL METRICS ====="
print_summary "MAD a2 4R + Judge" "$MAD_JUDGE_DIR"
print_single_role_summary "w/o Sourcing (Wording only)" "$NO_SOURCING_RESULT"
print_single_role_summary "w/o Wording (Sourcing only)" "$NO_WORDING_RESULT"

echo "===== NEWSMAD FINAL CONTROLS COMPLETE $(date '+%F %T') ====="
echo "mad_result=$MAD_RESULT"
echo "mad_judge_dir=$MAD_JUDGE_DIR"
echo "no_sourcing_result=$NO_SOURCING_RESULT"
echo "no_wording_result=$NO_WORDING_RESULT"

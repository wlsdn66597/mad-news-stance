#!/usr/bin/env bash
# One-run component ablations for the final two-agent Sourcing/Wording NewsMAD.
# The completed full S/W + Judge condition is never regenerated here.
#
#   1) w/o Sourcing: Wording / Wording, peer debate 4R, selective Judge
#   2) w/o Wording:  Sourcing / Sourcing, peer debate 4R, selective Judge
#   3) w/o Judge:    reuse the saved Sourcing / Wording 4R prediction
#
# Thus the role-removal conditions keep the agent count (2), round count (4),
# debate protocol, Judge policy, prompt profile, data, model, and generation
# settings fixed.  Only the removed role is replaced by a second copy of the
# remaining role.  The default single run uses seed 6000.
#
# Usage:
#   nohup bash scripts/run_newsmad_final_controls.sh \
#     > logs/newsmad_component_ablation_s6000.log 2>&1 &
#
# Optional smoke/inspection runs:
#   N=10 bash scripts/run_newsmad_final_controls.sh
#   DRY_RUN=1 bash scripts/run_newsmad_final_controls.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2.yaml}"
MODEL="${MODEL:-exaone}"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
PROFILE="${PROFILE:-stance_minimal_en}"
DATA_SEED="${DATA_SEED:-0}"
RUN_SEED="${RUN_SEED:-6000}"
ORDER_SEED="${ORDER_SEED:-8001}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
JUDGE_MODEL="${JUDGE_MODEL:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
RUN_NO_SOURCING="${RUN_NO_SOURCING:-1}"
RUN_NO_WORDING="${RUN_NO_WORDING:-1}"
DRY_RUN="${DRY_RUN:-0}"

BASE="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${RUN_SEED}_a2_r4"
FULL_SW_RESULT="${BASE}_personas_mix-sourcing-wording_reasoned_exchange_full.json"
NO_SOURCING_RESULT="${BASE}_personas_mix-wording-wording_reasoned_exchange_full.json"
NO_WORDING_RESULT="${BASE}_personas_mix-sourcing-sourcing_reasoned_exchange_full.json"

NO_SOURCING_JUDGE_DIR="results/judge_no_sourcing_ww_a2_4r_full_s${RUN_SEED}"
NO_WORDING_JUDGE_DIR="results/judge_no_wording_ss_a2_4r_full_s${RUN_SEED}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2 \
  "$NO_SOURCING_JUDGE_DIR" "$NO_WORDING_JUDGE_DIR"

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
  run_step "$title" \
    python scripts/run_phase2.py \
      --config "$CONFIG" --model "$MODEL" \
      --methods debate --personas --persona-mix "$mix" \
      --split "$SPLIT" --n "$N" \
      --prompt-profile "$PROFILE" \
      --data-seed "$DATA_SEED" --run-seed "$RUN_SEED" \
      --temperature "$TEMPERATURE" \
      --n-agents 2 --n-rounds 4 \
      --debate-protocol reasoned_exchange_full
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

print_judge_summary () {
  local label="$1"
  local output_dir="$2"
  local summary
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

print_no_judge_summary () {
  local result="$1"
  if [ ! -f "$result" ]; then
    echo "[summary] w/o Judge: missing existing S/W result $result"
    return 0
  fi
  python - "$result" <<'PY'
import json
import sys

from src.consensus import classification_metrics

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
rows = list(result["debate"].values())
metrics = classification_metrics(
    [row["pred"] for row in rows],
    [row["gold"] for row in rows],
)
print(
    f"[summary] w/o Judge: ACC={metrics['accuracy']:.4f} "
    f"F1={metrics['macro_f1']:.4f} items={len(rows)}"
)
PY
}

echo "===== NEWSMAD COMPONENT ABLATIONS START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED run_seed=$RUN_SEED"
echo "Full S/W + Judge is NOT regenerated."
echo "w/o Sourcing enabled=$RUN_NO_SOURCING: Wording/Wording, 2 agents, peer debate 4R, selective Judge"
echo "w/o Wording enabled=$RUN_NO_WORDING: Sourcing/Sourcing, 2 agents, peer debate 4R, selective Judge"
echo "w/o Judge: metrics read from the existing S/W 4R result"

if [ "$RUN_NO_SOURCING" = "1" ]; then
  run_debate "1. W/O SOURCING (WORDING / WORDING): DEBATE 4R" \
    "wording,wording"
  run_judge "2. W/O SOURCING (WORDING / WORDING): SELECTIVE JUDGE" \
    "$NO_SOURCING_RESULT" "$NO_SOURCING_JUDGE_DIR"
else
  echo "[skip] w/o Sourcing (RUN_NO_SOURCING=$RUN_NO_SOURCING)"
fi

if [ "$RUN_NO_WORDING" = "1" ]; then
  run_debate "3. W/O WORDING (SOURCING / SOURCING): DEBATE 4R" \
    "sourcing,sourcing"
  run_judge "4. W/O WORDING (SOURCING / SOURCING): SELECTIVE JUDGE" \
    "$NO_WORDING_RESULT" "$NO_WORDING_JUDGE_DIR"
else
  echo "[skip] w/o Wording (RUN_NO_WORDING=$RUN_NO_WORDING)"
fi

if [ "$DRY_RUN" != "1" ]; then
  echo "===== FINAL METRICS ====="
  if [ "$RUN_NO_SOURCING" = "1" ]; then
    print_judge_summary "w/o Sourcing (W/W)" "$NO_SOURCING_JUDGE_DIR"
  fi
  if [ "$RUN_NO_WORDING" = "1" ]; then
    print_judge_summary "w/o Wording (S/S)" "$NO_WORDING_JUDGE_DIR"
  fi
  print_no_judge_summary "$FULL_SW_RESULT"
fi

echo "===== NEWSMAD COMPONENT ABLATIONS COMPLETE $(date '+%F %T') ====="
echo "w/o Sourcing result: $NO_SOURCING_RESULT"
echo "w/o Sourcing Judge: $NO_SOURCING_JUDGE_DIR"
echo "w/o Wording result: $NO_WORDING_RESULT"
echo "w/o Wording Judge: $NO_WORDING_JUDGE_DIR"
echo "w/o Judge source (existing): $FULL_SW_RESULT"

#!/usr/bin/env bash
# Run the controls needed after defining the two-agent Sourcing/Wording pipeline
# as the final NewsMAD model.
#
#   1) matched generic MAD: two agents, four rounds, selective Judge (seed 6000)
#   2) w/o Sourcing: one Wording agent, four self-refinement rounds, no Judge
#   3) w/o Wording: one Sourcing agent, four self-refinement rounds, no Judge
#
# Role-removal ablations retain the paper's four-round setting.  Since only one
# role remains, peer exchange is replaced with the framework's full self-refine
# prompt: the agent receives its previous response and produces a complete new
# rationale and stance each round.  All other data/model/generation settings are
# unchanged.  Re-running this script resumes completed per-item outputs.
#
# Defaults:
#   - matched MAD is run once with seed 6000 for the one-run main table
#   - role ablations use seeds 6000, 6001, and 6002 for mean/std reporting
#
# Usage:
#   nohup bash scripts/run_newsmad_final_controls.sh \
#     > logs/newsmad_final_controls.log 2>&1 &
#
# Run only the role-removal ablations (e.g. MAD is already complete):
#   nohup env RUN_MAD=0 bash scripts/run_newsmad_final_controls.sh \
#     > logs/newsmad_role_ablation_selfrefine.log 2>&1 &
#
# Optional smoke/inspection runs:
#   N=10 ABLATION_SEEDS="6000" bash scripts/run_newsmad_final_controls.sh
#   DRY_RUN=1 bash scripts/run_newsmad_final_controls.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2.yaml}"
MODEL="${MODEL:-exaone}"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
PROFILE="${PROFILE:-stance_minimal_en}"
DATA_SEED="${DATA_SEED:-0}"
MAD_SEED="${MAD_SEED:-6000}"
ABLATION_SEEDS="${ABLATION_SEEDS:-6000 6001 6002}"
ORDER_SEED="${ORDER_SEED:-8001}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
JUDGE_MODEL="${JUDGE_MODEL:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
RUN_MAD="${RUN_MAD:-1}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing dataset: $DATA" >&2; exit 2; }
mkdir -p logs results/phase2

base_path () {
  local seed="$1"
  printf 'results/phase2/stance_%s_%s_n%s_%s_d%s_s%s' \
    "$MODEL" "$SPLIT" "$N" "$PROFILE" "$DATA_SEED" "$seed"
}

mad_result () {
  local seed="$1"
  printf '%s_a2_r4_reasoned_exchange_full.json' "$(base_path "$seed")"
}

role_result () {
  local seed="$1"
  local role="$2"
  printf '%s_a1_r4_personas_selfrefine-full_only-%s.json' \
    "$(base_path "$seed")" "$role"
}

mad_judge_dir () {
  local seed="$1"
  printf 'results/judge_mad_a2_4r_full_s%s' "$seed"
}

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

run_mad_debate () {
  local seed="$1"
  run_step "MAD S${seed}: DEBATE 4R" \
    python scripts/run_phase2.py \
      --config "$CONFIG" --model "$MODEL" \
      --methods debate \
      --split "$SPLIT" --n "$N" \
      --prompt-profile "$PROFILE" \
      --data-seed "$DATA_SEED" --run-seed "$seed" \
      --temperature "$TEMPERATURE" \
      --n-agents 2 --n-rounds 4 \
      --debate-protocol reasoned_exchange_full
}

run_role_self_refine () {
  local seed="$1"
  local role="$2"
  local label="$3"
  run_step "${label} S${seed}: ${role^^} SELF-REFINE 4R" \
    python scripts/run_phase2.py \
      --config "$CONFIG" --model "$MODEL" \
      --methods debate --personas --persona-mix "$role" \
      --peer-mode self --self-refine-format full \
      --split "$SPLIT" --n "$N" \
      --prompt-profile "$PROFILE" \
      --data-seed "$DATA_SEED" --run-seed "$seed" \
      --temperature "$TEMPERATURE" \
      --n-agents 1 --n-rounds 4
}

run_judge () {
  local seed="$1"
  local input_result
  local output_dir
  input_result=$(mad_result "$seed")
  output_dir=$(mad_judge_dir "$seed")
  mkdir -p "$output_dir"
  [ "$DRY_RUN" = "1" ] || [ -f "$input_result" ] || {
    echo "missing MAD result: $input_result" >&2
    exit 2
  }
  run_step "MAD S${seed}: SELECTIVE JUDGE" \
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
  local seed="$1"
  local output_dir
  local summary
  [ "$DRY_RUN" = "1" ] && return 0
  output_dir=$(mad_judge_dir "$seed")
  summary=$(find "$output_dir" -maxdepth 1 -type f -name '*.summary.json' -print -quit)
  if [ -z "$summary" ]; then
    echo "[summary] MAD S${seed}: no summary file in $output_dir"
    return 0
  fi
  python - "$seed" "$summary" <<'PY'
import json
import sys

seed, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
before = report["baseline_metrics"]
after = report["judge_metrics"]
trigger = report["trigger_subset"]
print(
    f"[summary] MAD S{seed}: "
    f"debate ACC={before['accuracy']:.4f} F1={before['macro_f1']:.4f} | "
    f"+Judge ACC={after['accuracy']:.4f} F1={after['macro_f1']:.4f} | "
    f"judge={trigger['triggered_total']}/{report['items']} "
    f"({trigger['trigger_ratio']:.2%})"
)
PY
}

print_role_summary () {
  local seed="$1"
  local role="$2"
  local label="$3"
  local result
  [ "$DRY_RUN" = "1" ] && return 0
  result=$(role_result "$seed" "$role")
  if [ ! -f "$result" ]; then
    echo "[summary] ${label} S${seed}: missing result $result"
    return 0
  fi
  python - "$label" "$seed" "$result" <<'PY'
import json
import sys

from src.consensus import classification_metrics, parse_stance

label, seed, path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
rows = list(result["debate"].values())

gold = [row["gold"] for row in rows]
round1 = []
for row in rows:
    first_response = row["debate_trace"]["answers_by_round"][0][0]
    round1.append(parse_stance(first_response) or "none")
round4 = [row["pred"] for row in rows]

r1 = classification_metrics(round1, gold)
r4 = classification_metrics(round4, gold)
print(
    f"[summary] {label} S{seed}: "
    f"R1 ACC={r1['accuracy']:.4f} F1={r1['macro_f1']:.4f} | "
    f"R4 ACC={r4['accuracy']:.4f} F1={r4['macro_f1']:.4f} | "
    f"items={len(rows)}"
)
PY
}

echo "===== NEWSMAD FINAL CONTROLS START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED temperature=$TEMPERATURE"
echo "MAD: seed=$MAD_SEED, two generic agents, peer debate 4R, selective Judge"
echo "Role removal seeds: $ABLATION_SEEDS"
echo "w/o Sourcing: one Wording agent, self-refine 4R, no Judge"
echo "w/o Wording: one Sourcing agent, self-refine 4R, no Judge"

if [ "$RUN_MAD" = "1" ]; then
  run_mad_debate "$MAD_SEED"
  run_judge "$MAD_SEED"
else
  echo "[skip] matched MAD control (RUN_MAD=$RUN_MAD)"
fi

for seed in $ABLATION_SEEDS; do
  run_role_self_refine "$seed" wording "W/O SOURCING"
  run_role_self_refine "$seed" sourcing "W/O WORDING"
done

echo "===== FINAL METRICS ====="
if [ "$RUN_MAD" = "1" ]; then
  print_judge_summary "$MAD_SEED"
fi
for seed in $ABLATION_SEEDS; do
  print_role_summary "$seed" wording "w/o Sourcing"
  print_role_summary "$seed" sourcing "w/o Wording"
done

echo "===== NEWSMAD FINAL CONTROLS COMPLETE $(date '+%F %T') ====="
echo "MAD result: $(mad_result "$MAD_SEED")"
echo "MAD Judge directory: $(mad_judge_dir "$MAD_SEED")"
for seed in $ABLATION_SEEDS; do
  echo "w/o Sourcing S${seed}: $(role_result "$seed" wording)"
  echo "w/o Wording S${seed}: $(role_result "$seed" sourcing)"
done

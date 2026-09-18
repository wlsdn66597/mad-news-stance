#!/usr/bin/env bash
# Evaluate the three ablations below an already-completed Sourcing/Wording
# NewsMAD run.  This script never regenerates the full S/W model.
#
#   1) w/o Sourcing: one Wording agent, four self-refinement rounds
#   2) w/o Wording: one Sourcing agent, four self-refinement rounds
#   3) w/o Judge: reuse the saved S/W four-round prediction before Judge
#
# The role-removal runs retain the paper's four-round setting.  With one role
# remaining there is no peer, so rounds 2--4 use the framework's full
# self-refine prompt: the agent receives its own preceding response and writes
# a complete revised rationale and stance.  The selective Judge has no
# disagreement to resolve in a one-agent condition and therefore makes no
# additional call.  All model/data/generation settings otherwise stay fixed.
#
# Defaults: seeds 6000, 6001, and 6002 for mean/std reporting.
# Re-running resumes per-item output rather than regenerating completed items.
#
# Usage:
#   nohup bash scripts/run_newsmad_final_controls.sh \
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
ABLATION_SEEDS="${ABLATION_SEEDS:-6000 6001 6002}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
mkdir -p logs results/phase2

base_path () {
  local seed="$1"
  printf 'results/phase2/stance_%s_%s_n%s_%s_d%s_s%s' \
    "$MODEL" "$SPLIT" "$N" "$PROFILE" "$DATA_SEED" "$seed"
}

full_sw_result () {
  local seed="$1"
  printf '%s_a2_r4_personas_mix-sourcing-wording_reasoned_exchange_full.json' \
    "$(base_path "$seed")"
}

role_result () {
  local seed="$1"
  local role="$2"
  printf '%s_a1_r4_personas_selfrefine-full_only-%s.json' \
    "$(base_path "$seed")" "$role"
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

print_metrics () {
  local label="$1"
  local seed="$2"
  local result="$3"
  local include_round1="$4"
  if [ ! -f "$result" ]; then
    echo "[summary] ${label} S${seed}: missing result $result"
    return 0
  fi
  python - "$label" "$seed" "$result" "$include_round1" <<'PY'
import json
import sys

from src.consensus import classification_metrics, parse_stance

label, seed, path, include_round1 = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
rows = list(result["debate"].values())
gold = [row["gold"] for row in rows]
final = classification_metrics([row["pred"] for row in rows], gold)

prefix = ""
if include_round1 == "1":
    round1 = []
    for row in rows:
        first_response = row["debate_trace"]["answers_by_round"][0][0]
        round1.append(parse_stance(first_response) or "none")
    first = classification_metrics(round1, gold)
    prefix = f"R1 ACC={first['accuracy']:.4f} F1={first['macro_f1']:.4f} | "

print(
    f"[summary] {label} S{seed}: {prefix}"
    f"R4 ACC={final['accuracy']:.4f} F1={final['macro_f1']:.4f} | "
    f"items={len(rows)}"
)
PY
}

echo "===== NEWSMAD ROLE ABLATIONS START $(date '+%F %T') ====="
echo "split=$SPLIT n=$N data_seed=$DATA_SEED temperature=$TEMPERATURE"
echo "seeds=$ABLATION_SEEDS"
echo "Full S/W + Judge is NOT regenerated."
echo "w/o Sourcing: one Wording agent, self-refine 4R"
echo "w/o Wording: one Sourcing agent, self-refine 4R"
echo "w/o Judge: metrics read from the existing S/W 4R result"

for seed in $ABLATION_SEEDS; do
  run_role_self_refine "$seed" wording "W/O SOURCING"
  run_role_self_refine "$seed" sourcing "W/O WORDING"
done

if [ "$DRY_RUN" != "1" ]; then
  echo "===== FINAL METRICS ====="
  for seed in $ABLATION_SEEDS; do
    print_metrics "w/o Sourcing" "$seed" "$(role_result "$seed" wording)" 1
    print_metrics "w/o Wording" "$seed" "$(role_result "$seed" sourcing)" 1
    print_metrics "w/o Judge" "$seed" "$(full_sw_result "$seed")" 0
  done
fi

echo "===== NEWSMAD ROLE ABLATIONS COMPLETE $(date '+%F %T') ====="
for seed in $ABLATION_SEEDS; do
  echo "w/o Sourcing S${seed}: $(role_result "$seed" wording)"
  echo "w/o Wording S${seed}: $(role_result "$seed" sourcing)"
  echo "w/o Judge S${seed} (existing): $(full_sw_result "$seed")"
done

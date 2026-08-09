#!/usr/bin/env bash
# Do distinct agent roles widen the candidate pool? Validation split, round 0.
#
#   nohup bash scripts/run_persona_pool_check.sh > logs/persona_pool.log 2>&1 &
#
# The three agents currently score 0.4735, 0.4725 and 0.4745, agree pairwise
# 86.5% of the time and are unanimous on 800 of 1001 items, so `single` and
# `majority k=3` are the same number and the candidate pool -- how often any
# agent names the gold label -- is stuck at 0.5465. That pool is the hard
# ceiling on every aggregation rule downstream, so the only question worth 50
# minutes is whether personas move it.
#
# 199 items x 3 agents x 2 profiles. No debate: round 0 is where the pool is set.
set -uo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2_exaone_stance_minimal_en.yaml}"
MODEL="${MODEL:-exaone}"
DATA_SEED="${DATA_SEED:-0}"
SEED="${SEED:-6000}"
VAL_N="${VAL_N:-199}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

for personas in "" "--personas"; do
  echo "===== VALIDATION majority ${personas:-shared prompt} START $(date) ====="
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_phase2.py \
    --config "$CONFIG" --model "$MODEL" --methods majority \
    --split validation --n "$VAL_N" --data-seed "$DATA_SEED" --run-seed "$SEED" \
    ${personas}
  echo "===== exit $? at $(date) ====="
done

echo "===== ALL COMPLETE $(date) ====="
base="results/phase2/stance_${MODEL}_validation_n${VAL_N}_stance_minimal_en_d${DATA_SEED}_s${SEED}_a3_r2"
echo "compare the pool with:"
echo "  python scripts/analyze_agent_diversity.py --result ${base}.json"
echo "  python scripts/analyze_agent_diversity.py --result ${base}_personas.json"
echo "and the vote with:"
echo "  python scripts/compare_methods.py ${base}.json:majority=shared ${base}_personas.json:majority=personas"

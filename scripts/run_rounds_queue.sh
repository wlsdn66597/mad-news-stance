#!/usr/bin/env bash
# Round extension + compute-matched majority baseline for one model.
#
#   bash scripts/run_rounds_queue.sh <config> <model-key> <finished r2 result> [run_seed]
#
# 1) seeds the round-extension run with the finished run's shared Round 0, so the
#    two runs are strictly paired and Round 0 is not re-sampled
# 2) runs debate with ROUNDS rounds   (n_agents * ROUNDS generations per item)
# 3) runs majority with k=K samples   (K generations per item, the compute match)
#
# Env overrides: ROUNDS=4 K=12 N=1001 SPLIT=test DATA_SEED=0
set -euo pipefail

CONFIG="${1:?usage: bash scripts/run_rounds_queue.sh <config> <model-key> <r2 result> [run_seed]}"
MODEL="${2:?usage: bash scripts/run_rounds_queue.sh <config> <model-key> <r2 result> [run_seed]}"
SOURCE="${3:?usage: bash scripts/run_rounds_queue.sh <config> <model-key> <r2 result> [run_seed]}"
SEED="${4:-6000}"

cd "$(dirname "$0")/.."

ROUNDS="${ROUNDS:-4}"
K="${K:-12}"
N="${N:-1001}"
SPLIT="${SPLIT:-test}"
DATA_SEED="${DATA_SEED:-0}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$SOURCE" ] || { echo "missing source result: $SOURCE" >&2; exit 2; }

read -r PROFILE AGENTS BASE_ROUNDS <<EOF
$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["prompt_profile"], cfg["debate"]["n_agents"], cfg["debate"]["n_rounds"])
PY
)
EOF

STEM="stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${SEED}"
ROUNDS_OUT="results/phase2/${STEM}_a${AGENTS}_r${ROUNDS}.json"
MATCHED_OUT="results/phase2/${STEM}_a${K}_r${BASE_ROUNDS}.json"

echo "config=$CONFIG model=$MODEL profile=$PROFILE agents=$AGENTS seed=$SEED"
echo "  round extension -> $ROUNDS_OUT   ($((AGENTS * ROUNDS)) generations/item, Round 0 reused)"
echo "  compute match   -> $MATCHED_OUT  ($K generations/item)"

# --- 1) pair Round 0 with the finished run --------------------------------
if [ -f "$ROUNDS_OUT" ]; then
  echo "===== SEED ROUND0 SKIPPED (output exists, resuming) ====="
else
  echo "===== SEED ROUND0 $(date) ====="
  python scripts/seed_shared_round0.py --from "$SOURCE" --to "$ROUNDS_OUT"
fi

# --- 2) debate with more rounds -------------------------------------------
echo "===== DEBATE r${ROUNDS} START $(date) ====="
START=$(date +%s)
python scripts/run_phase2.py \
  --config "$CONFIG" --model "$MODEL" --methods debate \
  --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$SEED" \
  --n-rounds "$ROUNDS"
echo "===== DEBATE r${ROUNDS} COMPLETE: $(( $(date +%s) - START )) sec ====="

# --- 3) compute-matched majority ------------------------------------------
# Deliberately NOT seeded from the 3-sample Round 0: majority needs exactly K
# answers per item, and a 3-answer Round 0 would abort the run.
echo "===== MAJORITY k=${K} START $(date) ====="
START=$(date +%s)
python scripts/run_phase2.py \
  --config "$CONFIG" --model "$MODEL" --methods majority \
  --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$SEED" \
  --n-agents "$K" --k "$K"
echo "===== MAJORITY k=${K} COMPLETE: $(( $(date +%s) - START )) sec ====="

echo "===== ALL COMPLETE $(date) ====="
echo "compare with:"
echo "  python scripts/compare_methods.py \\"
echo "    ${SOURCE}:majority=majority_k${AGENTS} \\"
echo "    ${MATCHED_OUT}:majority=majority_k${K} \\"
echo "    ${SOURCE}:debate=debate_r${BASE_ROUNDS} \\"
echo "    ${ROUNDS_OUT}:debate=debate_r${ROUNDS}"

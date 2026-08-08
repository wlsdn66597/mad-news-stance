#!/usr/bin/env bash
# Two jobs back to back on one GPU: the cheap prompt question first, the long
# replication second.
#
#   nohup bash scripts/run_neutral_prompt_and_repeat.sh > logs/neutral_and_repeat.log 2>&1 &
#
# 1) Neutral as a gate rather than a third label, one agent, VALIDATION split.
#    199 items x 1 call x 3 profiles, roughly 50 minutes. Validation, not test:
#    this step chooses a prompt, and choosing on the split you report would be
#    fitting to it. Whatever wins gets confirmed on test afterwards, once.
#
# 2) EXAONE four-round debate at run seed 6001, TEST split. The +17 over
#    majority (p = 0.027) is the only positive result in the project, it is one
#    seed, and its p-value is not corrected for the configurations looked at
#    before it. Everything conditional now rests on it. 12 generations per item
#    over 1001 items -- check the finished s6000 run's `_meta` for the wall
#    clock to expect; budget overnight.
#
# The order is deliberate: step 1 answers something by the time you wake up,
# step 2 runs regardless of what step 1 says.
set -uo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2_exaone_stance_minimal_en.yaml}"
MODEL="${MODEL:-exaone}"
DATA_SEED="${DATA_SEED:-0}"
SEED="${SEED:-6000}"
REPEAT_SEED="${REPEAT_SEED:-6001}"
VAL_N="${VAL_N:-199}"
TEST_N="${TEST_N:-1001}"
PROFILES="${PROFILES:-stance_minimal_en stance_twostep_en stance_gate_en}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }

# ---- 1) does framing neutral as a gate help the base agent? ----------------
for profile in $PROFILES; do
  echo "===== VALIDATION single, $profile START $(date) ====="
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_phase2.py \
    --config "$CONFIG" --model "$MODEL" --methods single \
    --prompt-profile "$profile" \
    --split validation --n "$VAL_N" --data-seed "$DATA_SEED" --run-seed "$SEED"
  echo "===== exit $? at $(date) ====="
done

echo "===== VALIDATION COMPLETE $(date) ====="
echo "compare the profiles with:"
for profile in $PROFILES; do
  echo "  results/phase2/stance_${MODEL}_validation_n${VAL_N}_${profile}_d${DATA_SEED}_s${SEED}_a3_r2.json:single=${profile}"
done
echo "(feed those to scripts/compare_methods.py; per-class recall is the number"
echo " that matters, not accuracy alone -- a gate can trade supportive for neutral)"

# ---- 2) the replication the headline depends on ----------------------------
echo "===== EXAONE r4 seed ${REPEAT_SEED} START $(date) ====="
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_phase2.py \
  --config "$CONFIG" --model "$MODEL" --methods majority,debate \
  --split test --n "$TEST_N" --data-seed "$DATA_SEED" --run-seed "$REPEAT_SEED" \
  --n-rounds 4
echo "===== exit $? at $(date) ====="

echo "===== ALL COMPLETE $(date) ====="
echo "then:"
echo "  python scripts/compare_methods.py \\"
echo "    results/phase2/stance_${MODEL}_test_n${TEST_N}_stance_minimal_en_d${DATA_SEED}_s${REPEAT_SEED}_a3_r4.json:majority=majority_s${REPEAT_SEED} \\"
echo "    results/phase2/stance_${MODEL}_test_n${TEST_N}_stance_minimal_en_d${DATA_SEED}_s${REPEAT_SEED}_a3_r4.json:debate=debate_r4_s${REPEAT_SEED}"

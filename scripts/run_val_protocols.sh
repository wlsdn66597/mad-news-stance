#!/usr/bin/env bash
# Choose a debate protocol on the validation split, before it costs a test run.
#
#   nohup bash scripts/run_val_protocols.sh > logs/val_protocols.log 2>&1 &
#   DRY_RUN=1 bash scripts/run_val_protocols.sh                      # plan only
#   PROTOCOLS="evidence_gated_v2 round_specific_v2" bash scripts/run_val_protocols.sh
#   SPLIT=test N=1001 PROTOCOLS=round_specific_v2 bash scripts/run_val_protocols.sh
#
# The handoff's rule: 199 items choose the prompt, 1001 report it. Every
# protocol here is scored against one baseline that shares its Round 0, so the
# only thing that differs between the runs is the user prompt of rounds 1..3.
#
# The baseline generates Round 0 once; every protocol reuses it through
# seed_shared_round0.py, which cuts each protocol run from 12 to 9 generations
# per item. Everything resumes per item, so re-running finishes the leftovers
# and costs nothing for what is already on disk.
#
# CONFIG matters. config/phase2.yaml is NOT what the report was run with: it
# sets temperature 0.7 against the 1.0 in the file below, and a protocol run at
# the wrong temperature cannot be compared with anything already measured.
set -uo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/phase2_exaone_stance_minimal_en.yaml}"
MODEL="${MODEL:-exaone}"
PROFILE="${PROFILE:-stance_minimal_en}"
SPLIT="${SPLIT:-validation}"
N="${N:-199}"
DATA_SEED="${DATA_SEED:-0}"
SEED="${SEED:-6000}"
ROUNDS="${ROUNDS:-4}"
PROTOCOLS="${PROTOCOLS:-evidence_gated evidence_gated_v2 round_specific round_specific_v2}"
DRY_RUN="${DRY_RUN:-0}"

stem () {  # stem [suffix]
  printf 'results/phase2/stance_%s_%s_n%s_%s_d%s_s%s_a3_r%s_personas%s.json' \
    "$MODEL" "$SPLIT" "$N" "$PROFILE" "$DATA_SEED" "$SEED" "$ROUNDS" "${1:-}"
}

step () {
  local label="$1"; shift
  echo
  echo "===== $label   $(date '+%Y-%m-%d %H:%M:%S')"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [plan] $*"
    return 0
  fi
  "$@" || { echo "[failed] $label" >&2; return 1; }
}

phase2 () {  # phase2 <label> [extra args...]
  local label="$1"; shift
  step "$label" python scripts/run_phase2.py \
    --config "$CONFIG" --model "$MODEL" --methods majority,debate \
    --prompt-profile "$PROFILE" --personas \
    --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$SEED" \
    --n-rounds "$ROUNDS" "$@"
}

base=$(stem)

echo "config: $CONFIG"
echo "split: $SPLIT   n: $N   seed: $SEED   rounds: $ROUNDS"
echo "protocols: $PROTOCOLS"
echo "baseline: $base"
echo "per item: baseline 3*$ROUNDS generations, each protocol 3*$((ROUNDS - 1)) (Round 0 reused)"

phase2 "baseline (no protocol)"
if [ "$DRY_RUN" != 1 ] && [ ! -f "$base" ]; then
  echo "[stop] $base was not produced; nothing to pair the protocols against" >&2
  exit 1
fi

for protocol in $PROTOCOLS; do
  target=$(stem "_$protocol")
  # seed_shared_round0.py refuses an existing target, which is what makes a
  # re-run safe: the second pass skips the copy and resumes the generation
  if [ -f "$target" ]; then
    echo "[skip] Round 0 already seeded into $(basename "$target")"
  else
    step "carry Round 0 into $(basename "$target")" \
      python scripts/seed_shared_round0.py --from "$base" --to "$target"
  fi
  phase2 "protocol $protocol" --debate-protocol "$protocol"
done

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "===== plan complete; nothing was loaded"
  exit 0
fi

echo
echo "===== reading the runs   $(date '+%Y-%m-%d %H:%M:%S')"
python scripts/collect_report_table.py --seeds "$SEED" --split "$SPLIT" --n "$N" \
  --model "$MODEL" --profile "$PROFILE" --data-seed "$DATA_SEED"

for protocol in $PROTOCOLS; do
  echo
  echo "===== $protocol vs baseline"
  python scripts/analyze_rounds.py "$(stem "_$protocol")" --compare "$base"
done

cat <<'NOTE'

===== what to read first

199 items means one item is 0.5 points, so read the mechanism before the score.

  1. evidence_gated_v2: does anything move at all? The v1 gate flipped 0 of
     3003 agent labels on test. If "agents flipped" is still 0, the prompt is
     not the lever.
  2. round_specific_v2: does round 2 still collapse to neutral? On test v1 it
     went to neutral recall 0.8061 and accuracy 0.3976. Under 0.4 and no
     accuracy crater means it is fixed.
  3. Only for the protocols that pass both, compare the macro-F1 and neutral
     recall trajectory across rounds. Baseline debate on test rose 0.3273 ->
     0.4061 on neutral recall; that rise is where its whole macro-F1 advantage
     came from.
  4. Accuracy decides nothing here. Take the winner to the test split and
     report it there.
NOTE

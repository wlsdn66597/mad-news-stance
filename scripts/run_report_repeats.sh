#!/usr/bin/env bash
# The four pipelines the report needs, repeated over seeds.
#
#   nohup bash scripts/run_report_repeats.sh > logs/report_repeats.log 2>&1 &
#   SEEDS="6002" STAGES="4 5" bash scripts/run_report_repeats.sh     # a slice
#   DRY_RUN=1 bash scripts/run_report_repeats.sh                     # plan only
#
# Stages, numbered as in the report outline:
#
#   1  MAD as published:  single / majority k=3 / debate 2R / debate 4R
#   3  stance advocacy:   one agent per label + judge          (diagnostic)
#   4  persona debate:    framing / sourcing / wording, 2R + 4R
#   5  selective judge:   persona 2R and 4R + judge on non-unanimous items
#
# Stage 2 is offline (analyze_agent_diversity.py) and stage 6 is a choice
# between two stage-5 outputs, so neither generates anything.
#
# Every model in every stage is EXAONE-4.0-1.2B, judges included, so the report
# can state one backbone without qualification. Set JUDGE_MODEL to a stronger
# model if a "MAD + strong judge" row is wanted later; that has to be labelled
# as such, because it is no longer a single-model result.
#
# Everything resumes per item, so re-running a finished seed costs nothing and
# an interrupted one picks up where it stopped.
set -uo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-6000 6001 6002}"
STAGES="${STAGES:-1 3 4 5}"
CONFIG="${CONFIG:-config/phase2_exaone_stance_minimal_en.yaml}"
MODEL="${MODEL:-exaone}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
PROFILE="${PROFILE:-stance_minimal_en}"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
DATA_SEED="${DATA_SEED:-0}"
# empty means the judge is whatever model produced the run: EXAONE
JUDGE_MODEL="${JUDGE_MODEL:-}"
ORDER_SEED="${ORDER_SEED:-8001}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$DATA" ]   || { echo "missing dataset: $DATA" >&2; exit 2; }

stem () {  # stem <seed> <rounds> [suffix]
  echo "results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s$1_a3_r$2${3:-}.json"
}

step () {
  local name="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    echo "[plan] $name"
    printf '        %q ' "$@"; echo
    return 0
  fi
  echo "===== $name START $(date '+%F %T') ====="
  "$@"
  echo "===== $name exit $? at $(date '+%F %T') ====="
}

# Round 0 is generated from (run_seed, item) alone, so a follow-up run would
# resample it identically -- but copying it is free, provably paired, and saves
# three generations per item.
carry_round0 () {  # carry_round0 <from> <to>
  [ -f "$1" ] || return 0
  [ -f "$2" ] && return 0
  step "carry round0 into $(basename "$2")" \
    python scripts/seed_shared_round0.py --from "$1" --to "$2"
}

stage_1 () {  # published MAD, shared prompt
  local seed="$1" r2 r4
  r2=$(stem "$seed" 2); r4=$(stem "$seed" 4)
  step "s${seed} stage1 r2 (single, majority, debate)" \
    python scripts/run_phase2.py --config "$CONFIG" --model "$MODEL" \
      --methods single,majority,debate --prompt-profile "$PROFILE" \
      --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$seed" \
      --n-rounds 2
  carry_round0 "$r2" "$r4"
  step "s${seed} stage1 r4 (majority, debate)" \
    python scripts/run_phase2.py --config "$CONFIG" --model "$MODEL" \
      --methods majority,debate --prompt-profile "$PROFILE" \
      --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$seed" \
      --n-rounds 4
}

stage_3 () {  # one agent per label, judge on the same backbone
  local seed="$1"
  step "s${seed} stage3 advocacy + EXAONE judge" \
    python scripts/run_advocacy_judge.py --config "$CONFIG" --model "$MODEL" \
      --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$seed" \
      --rounds 2 --prompt-style toc --order-seed "$ORDER_SEED" \
      --baseline-result "$(stem "$seed" 2)" --baseline-method majority \
      --output-dir results/advocacy
}

stage_4 () {  # persona debate
  local seed="$1" r2 r4
  r2=$(stem "$seed" 2 _personas); r4=$(stem "$seed" 4 _personas)
  step "s${seed} stage4 personas r2" \
    python scripts/run_phase2.py --config "$CONFIG" --model "$MODEL" \
      --methods single,majority,debate --prompt-profile "$PROFILE" --personas \
      --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$seed" \
      --n-rounds 2
  carry_round0 "$r2" "$r4"
  step "s${seed} stage4 personas r4" \
    python scripts/run_phase2.py --config "$CONFIG" --model "$MODEL" \
      --methods majority,debate --prompt-profile "$PROFILE" --personas \
      --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$seed" \
      --n-rounds 4
}

stage_5 () {  # judge only where the personas did not agree
  local seed="$1" rounds source args
  for rounds in 2 4; do
    source=$(stem "$seed" "$rounds" _personas)
    if [ ! -f "$source" ]; then
      echo "[skip] s${seed} stage5 r${rounds}: $source not built yet" >&2
      continue
    fi
    args=(--input-result "$source" --data-path "$DATA"
          --judge-trigger non_unanimous --judge-input-mode article_only
          --judge-order-seed "$ORDER_SEED" --judge-temperature 0
          --output-dir results/selective_judge_report)
    [ -n "$JUDGE_MODEL" ] && args+=(--judge-model "$JUDGE_MODEL")
    step "s${seed} stage5 personas r${rounds} + judge${JUDGE_MODEL:+ ($JUDGE_MODEL)}" \
      python scripts/run_selective_judge.py "${args[@]}"
  done
}

echo "seeds: $SEEDS   stages: $STAGES   judge: ${JUDGE_MODEL:-$MODEL (same backbone)}"
echo "per seed, generations per item: stage1 15, stage3 7, stage4 15,"
echo "          stage5 ~0.56 judge calls (non-unanimous items only)"

for seed in $SEEDS; do
  for s in $STAGES; do
    case "$s" in
      1) stage_1 "$seed" ;;
      3) stage_3 "$seed" ;;
      4) stage_4 "$seed" ;;
      5) stage_5 "$seed" ;;
      *) echo "unknown stage: $s" >&2 ;;
    esac
  done
done

echo "===== ALL COMPLETE $(date '+%F %T') ====="
echo "collect the report tables with:"
echo "  python scripts/collect_report_table.py --seeds $SEEDS"

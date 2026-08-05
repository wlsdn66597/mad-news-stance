#!/usr/bin/env bash
# Selective consensus for every debate repeat, three conditions each.
#
#   bash scripts/run_selective_consensus_queue.sh            # all repeats
#   bash scripts/run_selective_consensus_queue.sh 6001 6002  # only these
#
# Per repeat: direct_article_judge (cache only, no GPU), extra_round_then_judge,
# reconsideration_then_judge. The consensus seed defaults to that repeat's own
# debate run seed, so the three repeats also vary the consensus sampling.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${CONSENSUS_MODEL:-Qwen/Qwen3-8B}"
OUT="${OUTPUT_DIR:-results/selective_consensus}"
JUDGE_SUFFIX_TEMPLATE="_judge-Qwen_Qwen3-8B_trigger-instability_input-article_only_seed-%s_temp-0_tok-384_retry-1_q4-0_agg-current_final_majority_a3_r2.items.json"

# run seed | phase2 stem | judge order seed of the cached article-only run
REPEATS=(
  "6000|stance_qwen_test_n1001_stance_minimal_en_d0_s6000_a3_r2_qwen8_test_full_n1001_s6000|7002"
  "6001|stance_qwen_test_n1001_stance_minimal_en_d0_s6001_a3_r2_qwen8_debate_repeat_n1001_s6001|7001"
  "6002|stance_qwen_test_n1001_stance_minimal_en_d0_s6002_a3_r2_qwen8_debate_repeat_n1001_s6002|7001"
)

WANTED=("$@")
selected() {
  [ ${#WANTED[@]} -eq 0 ] && return 0
  for want in "${WANTED[@]}"; do [ "$want" = "$1" ] && return 0; done
  return 1
}

# Fail before any GPU time if an input is missing.
PLANNED=()
for entry in "${REPEATS[@]}"; do
  IFS='|' read -r SEED STEM JSEED <<< "$entry"
  selected "$SEED" || continue
  RESULT="results/phase2/${STEM}.json"
  # shellcheck disable=SC2059
  JUDGE="results/selective_judge/${STEM}$(printf "$JUDGE_SUFFIX_TEMPLATE" "$JSEED")"
  [ -f "$RESULT" ] || { echo "missing debate result: $RESULT" >&2; exit 2; }
  [ -f "$JUDGE" ] || { echo "missing judge cache: $JUDGE" >&2; exit 2; }
  PLANNED+=("$SEED|$RESULT|$JUDGE|$JSEED")
done
[ ${#PLANNED[@]} -gt 0 ] || { echo "no repeats selected" >&2; exit 2; }

echo "planned repeats: ${#PLANNED[@]}"
for plan in "${PLANNED[@]}"; do
  IFS='|' read -r SEED RESULT JUDGE JSEED <<< "$plan"
  for MODE in none plain_extra_round independent_reconsideration; do
    echo "===== SEED ${SEED} ${MODE} START $(date) ====="
    START=$(date +%s)
    python scripts/run_selective_consensus.py \
      --input-result "$RESULT" \
      --consensus-prompt-mode "$MODE" \
      --consensus-model "$MODEL" \
      --consensus-seed "$SEED" \
      --judge-model "$MODEL" \
      --judge-result "$JUDGE" \
      --judge-order-seed "$JSEED" \
      --output-dir "$OUT"
    echo "===== SEED ${SEED} ${MODE} COMPLETE: $(( $(date +%s) - START )) sec ====="
  done
done

echo "===== ALL SELECTIVE CONSENSUS RUNS COMPLETE $(date) ====="

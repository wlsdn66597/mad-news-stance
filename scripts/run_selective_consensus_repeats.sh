#!/usr/bin/env bash
# Selective consensus: the three conditions on one saved MAD result.
#
#   usage: bash scripts/run_selective_consensus_repeats.sh \
#            <mad_result.json> <article_only_judge.items.json> [consensus seeds...]
#
# Condition 1 (direct_article_judge) reuses the cached article-only judge
# predictions and makes no model call at all, so it reproduces the existing
# selective-judge numbers as a sanity check before any GPU time is spent.
set -euo pipefail

RESULT="${1:?usage: bash scripts/run_selective_consensus_repeats.sh <mad_result.json> <article_only_judge.items.json> [seeds...]}"
JUDGE="${2:?usage: bash scripts/run_selective_consensus_repeats.sh <mad_result.json> <article_only_judge.items.json> [seeds...]}"
shift 2

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(6000)
fi

DATA="${DATA_PATH:-data/k-news-stance_nosegment.json}"
OUT="${OUTPUT_DIR:-results/selective_consensus}"
MODEL="${CONSENSUS_MODEL:-Qwen/Qwen3-8B}"
# Must match the --judge-order-seed of the cached article-only judge run:
# it also seeds the deterministic fallback used when the judge returns nothing.
JUDGE_ORDER_SEED="${JUDGE_ORDER_SEED:-7001}"

cd "$(dirname "$0")/.."

echo "===== DIRECT ARTICLE JUDGE (cache only) START ====="
python scripts/run_selective_consensus.py \
  --input-result "$RESULT" \
  --data-path "$DATA" \
  --consensus-prompt-mode none \
  --judge-model "$MODEL" \
  --judge-result "$JUDGE" \
  --judge-order-seed "$JUDGE_ORDER_SEED" \
  --output-dir "$OUT"
echo "===== DIRECT ARTICLE JUDGE COMPLETE ====="

for SEED in "${SEEDS[@]}"; do
  for MODE in plain_extra_round independent_reconsideration; do
    echo "===== ${MODE} CONSENSUS-SEED ${SEED} START ====="
    python scripts/run_selective_consensus.py \
      --input-result "$RESULT" \
      --data-path "$DATA" \
      --consensus-prompt-mode "$MODE" \
      --consensus-model "$MODEL" \
      --consensus-seed "$SEED" \
      --judge-model "$MODEL" \
      --judge-result "$JUDGE" \
      --judge-order-seed "$JUDGE_ORDER_SEED" \
      --output-dir "$OUT"
    echo "===== ${MODE} CONSENSUS-SEED ${SEED} COMPLETE ====="
  done
done

echo "===== ALL CONDITIONS COMPLETE ====="

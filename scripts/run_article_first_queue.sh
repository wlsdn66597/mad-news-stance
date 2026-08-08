#!/usr/bin/env bash
# The article-first condition and the two ablations that attribute its effect.
#
#   nohup bash scripts/run_article_first_queue.sh > logs/article_first_queue.log 2>&1 &
#
# Sequential, because there is one GPU. Roughly 2h05 in total:
#
#  1) article_first, quotes tagged   -- 168 advocate + 174 judge calls  (~55 min)
#  2) the same, quotes untagged      -- 174 judge calls, cases reused   (~35 min)
#  3) the same, different candidate order -- 174 judge calls, reused    (~35 min)
#
# 1 asks whether the analyses are worth anything once the judge has already
# formed its own view of the article. 2 subtracts the quote tagging, 3 subtracts
# candidate order, so any gain in 1 can be attributed rather than just observed.
# Every step keeps article_only (0.5355) as its floor by replaying that run's
# verdict as the judge's first turn.
set -uo pipefail

cd "$(dirname "$0")/.."

INPUT="${INPUT:-results/phase2/stance_exaone_test_n1001_stance_minimal_en_d0_s6000_a3_r2.json}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
CONFIG="${CONFIG:-config/phase2_qwen8_stance_minimal_en.yaml}"
MODEL="${MODEL:-qwen}"
ADVOCATE="${ADVOCATE:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
SEED="${SEED:-6000}"
ORDER_SEED="${ORDER_SEED:-8001}"
ALT_ORDER_SEED="${ALT_ORDER_SEED:-9002}"
OUT="${OUT:-results/selective_advocacy}"

STEM="$OUT/seladv_$(basename "$INPUT" .json)_unstable_or_split-last"
SUFFIX="toc_ord${ORDER_SEED}_s${SEED}"
STAGE1="${STAGE1:-${STEM}_abl-article_only_${SUFFIX}_adv-Qwen3-8B_judge-Qwen3-8B_jt0.items.json}"
TAGGED="${STEM}_abl-article_first_${SUFFIX}_adv-EXAONE-4_0-1_2B_judge-Qwen3-8B_jt0.items.json"

[ -f "$INPUT" ]  || { echo "missing input result: $INPUT" >&2; exit 2; }
[ -f "$STAGE1" ] || { echo "missing article_only run to replay: $STAGE1" >&2; exit 2; }

common=(
  --input-result "$INPUT" --data-path "$DATA"
  --config "$CONFIG" --model "$MODEL"
  --ablation article_first --stage1-from "$STAGE1"
  --run-seed "$SEED" --output-dir "$OUT"
)

run () {
  echo "===== $1 START $(date) ====="
  shift
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_selective_advocacy.py "$@"
  echo "===== exit $? at $(date) ====="
}

# 1) the condition itself. This is the only step that generates advocate cases.
run "article_first, quotes tagged" \
  "${common[@]}" --order-seed "$ORDER_SEED" \
  --advocate-model "$ADVOCATE" --verify-quotes

if [ ! -f "$TAGGED" ]; then
  echo "step 1 produced no $TAGGED; stopping before the ablations" >&2
  exit 1
fi

# 2) same cases, same order, no quote tags: is the tagging doing the work?
run "article_first, quotes untagged" \
  "${common[@]}" --order-seed "$ORDER_SEED" \
  --reuse-commissioned "$TAGGED" --no-verify-quotes

# 3) same cases, tagged, different candidate order: how much is position noise?
#    The Qwen judge's winning position is non-uniform at p = 0.0014.
run "article_first, alternate candidate order" \
  "${common[@]}" --order-seed "$ALT_ORDER_SEED" \
  --reuse-commissioned "$TAGGED" --verify-quotes

echo "===== ALL COMPLETE $(date) ====="
echo "compare with:"
echo "  python scripts/analyze_selective_runs.py ${STEM}*.items.json"

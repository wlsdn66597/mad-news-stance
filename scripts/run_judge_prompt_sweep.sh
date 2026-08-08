#!/usr/bin/env bash
# Does a shorter judge prompt help the local 8B on the routed items?
#
#   nohup bash scripts/run_judge_prompt_sweep.sh > logs/judge_prompt_sweep.log 2>&1 &
#
# The article-only judge is the only condition in this line that beats majority,
# and the published prompts it was built from are short and were written for API
# models. Every measured condition here has the shorter prompt winning -- 64
# words 0.5355, 108 words 0.5215, 145 words 0.5165, 176 words 0.5175 -- but
# prompt length was confounded with payload content. These runs hold the payload
# at the article and vary only the instruction and the output format:
#
#   adjudicator_json  64w, JSON out   (already run: 0.5355)
#   brief_json        34w, JSON out   -- length alone, format held
#   minimal_line      19w, prose out  -- shortest
#   stance_profile    32w, prose out  -- the debaters' own prompt, so this equals
#                                        one sample of the strong model, and the
#                                        saved `single` method is its free check
#
# 174 judge calls each, no advocate model, about 35 minutes per condition.
set -uo pipefail

cd "$(dirname "$0")/.."

INPUT="${INPUT:-results/phase2/stance_exaone_test_n1001_stance_minimal_en_d0_s6000_a3_r2.json}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
CONFIG="${CONFIG:-config/phase2_qwen8_stance_minimal_en.yaml}"
MODEL="${MODEL:-qwen}"
SEED="${SEED:-6000}"
ORDER_SEED="${ORDER_SEED:-8001}"
OUT="${OUT:-results/selective_advocacy}"
PROMPTS="${PROMPTS:-brief_json minimal_line stance_profile}"

[ -f "$INPUT" ] || { echo "missing input result: $INPUT" >&2; exit 2; }

for prompt in $PROMPTS; do
  echo "===== judge prompt $prompt START $(date) ====="
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_selective_advocacy.py \
    --input-result "$INPUT" --data-path "$DATA" \
    --config "$CONFIG" --model "$MODEL" \
    --ablation article_only --judge-prompt "$prompt" \
    --run-seed "$SEED" --order-seed "$ORDER_SEED" --output-dir "$OUT"
  echo "===== exit $? at $(date) ====="
done

echo "===== ALL COMPLETE $(date) ====="
echo "compare with:"
echo "  python scripts/analyze_selective_runs.py $OUT/seladv_$(basename "$INPUT" .json)_unstable_or_split-last*.items.json"

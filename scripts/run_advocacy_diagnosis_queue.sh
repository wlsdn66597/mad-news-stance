#!/usr/bin/env bash
# The three diagnostic runs that reuse the saved advocate cases, in order.
#
#   bash scripts/run_advocacy_diagnosis_queue.sh <config> <model-key> <cases .items.json> [baseline phase2 .json]
#
# Nothing here regenerates advocacy: full regeneration is 7 calls/item (~10.7h),
# these are 1 judge call/item (~1.5h with an 8B judge, minutes with a 1.2B one).
#
#  0) offline ceiling on the saved cases      -- no GPU at all
#  1) judge round 0 vs the round it read      -- is the rebuttal contaminating?
#  2) same judge, different candidate order   -- is the position bias real?
#
# Env overrides:
#   JUDGE_MODEL=Qwen/Qwen3-8B  STYLE=structured  N=1001  SPLIT=test
#   DATA_SEED=0  RUN_SEED=6000  ROUNDS=2  ORDER_SEED=8001  ALT_ORDER_SEED=9002
#   JUDGE_TOKENS=384  OUT=results/advocacy
set -euo pipefail

CONFIG="${1:?usage: bash scripts/run_advocacy_diagnosis_queue.sh <config> <model-key> <cases .items.json> [baseline]}"
MODEL="${2:?usage: bash scripts/run_advocacy_diagnosis_queue.sh <config> <model-key> <cases .items.json> [baseline]}"
CASES="${3:?usage: bash scripts/run_advocacy_diagnosis_queue.sh <config> <model-key> <cases .items.json> [baseline]}"
BASELINE="${4:-}"

cd "$(dirname "$0")/.."

STYLE="${STYLE:-structured}"
N="${N:-1001}"
SPLIT="${SPLIT:-test}"
DATA_SEED="${DATA_SEED:-0}"
RUN_SEED="${RUN_SEED:-6000}"
ROUNDS="${ROUNDS:-2}"
ORDER_SEED="${ORDER_SEED:-8001}"
ALT_ORDER_SEED="${ALT_ORDER_SEED:-9002}"
JUDGE_TOKENS="${JUDGE_TOKENS:-384}"
OUT="${OUT:-results/advocacy}"

baseline_args=()
if [[ -n "$BASELINE" ]]; then
  baseline_args=(--baseline-result "$BASELINE" --baseline-method "${BASELINE_METHOD:-majority}")
fi
judge_args=()
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  judge_args=(--judge-model "$JUDGE_MODEL")
fi

common=(
  --config "$CONFIG" --model "$MODEL"
  --split "$SPLIT" --n "$N" --data-seed "$DATA_SEED" --run-seed "$RUN_SEED"
  --rounds "$ROUNDS" --order-seed "$ORDER_SEED"
  --prompt-style "$STYLE" --judge-max-new-tokens "$JUDGE_TOKENS"
  --reuse-advocacy "$CASES" --output-dir "$OUT"
  "${judge_args[@]}" "${baseline_args[@]}"
)

echo "=== 0/3  offline ceiling on the saved cases (no GPU) ==="
python scripts/advocacy_oracle.py "$CASES" --round 0 --json "${CASES%.items.json}.oracle_r0.json"
python scripts/advocacy_oracle.py "$CASES" --round saved --json "${CASES%.items.json}.oracle_saved.json"

# run one judge pass and echo the .items.json it wrote, so the comparison below
# never has to guess a filename
run_pass() {
  local log
  log="$(mktemp)"
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} python scripts/run_advocacy_judge.py "$@" 2>&1 | tee "$log" >&2
  sed -n 's/^\[saved\] \(.*\)\.\*.*$/\1.items.json/p' "$log" | tail -1
}

echo "=== 1/3  judge reads round 0 (independent cases, no rebuttal) ==="
ROUND0="$(run_pass "${common[@]}" --judge-round 0)"

echo "=== 2/3  judge reads the last round, same settings (the paired control) ==="
LAST="$(run_pass "${common[@]}" --judge-round last)"

# The position-bias pass costs as much as the round comparison and only confirms
# a number the offline oracle already reports from saved runs. Skip it unless the
# judge's winning position was actually non-uniform.
alt_spec=()
if [[ "${SKIP_ALT_ORDER:-0}" == "1" ]]; then
  echo "=== 3/3  skipped (SKIP_ALT_ORDER=1) ==="
else
  echo "=== 3/3  last round again, different candidate order (position bias) ==="
  ALT="$(run_pass "${common[@]}" --judge-round last --judge-order-seed "$ALT_ORDER_SEED")"
  alt_spec=("${ALT}:pred=judge_last_altorder")
fi

echo "=== paired comparison across the judge conditions ==="
python scripts/compare_methods.py \
  "${ROUND0}:pred=judge_round0" \
  "${LAST}:pred=judge_last" \
  "${alt_spec[@]}" \
  ${BASELINE:+"$BASELINE:${BASELINE_METHOD:-majority}=majority"}

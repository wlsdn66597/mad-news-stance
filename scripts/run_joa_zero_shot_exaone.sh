#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config/phase2_exaone_stance_minimal_en.yaml}"
SOURCE="${SOURCE:-data/k-news-stance-for-prediction-test1001-joa-icl.json}"
SEGMENTS="${SEGMENTS:-data/k-news-stance_test1001_joa_segments_unlabeled.json}"
N="${N:-1001}"
DATA_SEED="${DATA_SEED:-0}"
RUN_SEED="${RUN_SEED:-6000}"

if [[ ! -f "$SEGMENTS" ]]; then
  if [[ ! -f "$SOURCE" ]]; then
    echo "Missing $SEGMENTS and source segmentation file $SOURCE" >&2
    echo "Place the supplied JoA-ICL JSON at $SOURCE, then rerun." >&2
    exit 1
  fi
  python scripts/prepare_joa_zero_shot.py \
    --input "$SOURCE" \
    --output "$SEGMENTS"
fi

python scripts/run_joa_zero_shot.py \
  --config "$CONFIG" \
  --model exaone \
  --segments "$SEGMENTS" \
  --split test \
  --n "$N" \
  --data-seed "$DATA_SEED" \
  --run-seed "$RUN_SEED" \
  "$@"

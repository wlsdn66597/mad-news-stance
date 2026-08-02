# Paper-faithful Phase 1 sampling

The repeated runner defaults to `--sampling-protocol paper` and keeps `--data-seed 0` fixed across runs. Only the generation seed changes between repetitions.

## Sampling algorithms

- GSM8K: load the test split, seed Python's random generator with 0, shuffle all rows, and evaluate the first 100. This follows the authors' `gsm/gen_gsm.py` implementation.
- MMLU: choose one subject uniformly, then choose one row uniformly from that subject, repeating this 100 times with replacement. Subjects are sorted before sampling so the result is stable across filesystems. This follows the authors' `mmlu/gen_mmlu.py` algorithm.

The original MMLU code used unsorted filesystem `glob` order and did not publish sampled row IDs. Therefore the repository reproduces the published algorithm deterministically, but cannot guarantee the exact same 100 MMLU rows used by the authors.

Each result stores `draw_index`, `source_index`, and `subject` when available. A unique draw index prevents repeated MMLU samples from overwriting one another.

## Repeated experiment

```bash
python scripts/run_phase1_repeated.py \
  --config config/phase1.yaml \
  --tasks gsm8k,mmlu \
  --models exaone,qwen4,qwen8 \
  --methods vanilla,cot,majority,debate \
  --n 100 \
  --repeats 3 \
  --data-seed 0 \
  --base-run-seed 1000 \
  --sampling-protocol paper \
  --temperature 1.0 \
  --n-agents 3 \
  --n-rounds 2 \
  --tag-prefix paper100
```

Sources:

- Paper: https://arxiv.org/pdf/2305.14325
- GSM8K sampling: https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/gen_gsm.py
- MMLU sampling: https://github.com/composable-models/llm_multiagent_debate/blob/main/mmlu/gen_mmlu.py

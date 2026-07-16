# K-News-Stance v2 prompt and memory experiments

## Design

This application track is separate from the frozen GSM8K/MMLU reproduction
prompts. It uses Qwen3-14B with bitsandbytes NF4 4-bit weights.

Qwen3 supports Korean and more than 100 languages, so English is not assumed to
be inherently better. The default uses English control instructions with the
original Korean article. `stance_v2_ko` mirrors the same protocol in Korean;
validation decides which language to keep.

Profiles:

- `legacy_ko`: original prompt.
- `stance_v2_en_generic_memory`: improved English question/debate prompts with
  the original generic memory.
- `stance_v2_en`: English evidence-audit prompts and structured memory.
- `stance_v2_ko`: Korean counterpart of the structured English profile.

The v2 prompts separate authorial framing from quoted speakers, require concrete
article evidence before a label change, reject vote counts as evidence, preserve
grounded minority evidence, and flag unsupported or misattributed claims.

With `share_round0: true`, majority, direct debate, and memory debate reuse the
exact same independent initial answers. Results store these under
`_shared_round0`, so communication methods can be compared pairwise.

## Smoke test

```bash
python scripts/run_phase2.py \
  --config config/phase2_qwen14b4_stance_v2.yaml \
  --methods majority,debate,debate_memory \
  --n 3 --run-seed 3000 --tag smoke
```

Startup must show `profile=stance_v2_en` and `load_in_4bit=True`.

## Experiment order

### 1. English versus Korean, validation n=60, three runs

```bash
for PROFILE in stance_v2_en stance_v2_ko; do
  for SEED in 3000 3001 3002; do
    python scripts/run_phase2.py \
      --config config/phase2_qwen14b4_stance_v2.yaml \
      --prompt-profile "$PROFILE" \
      --methods majority,debate,debate_memory \
      --n 60 --data-seed 0 --run-seed "$SEED" \
      --tag "lang_${PROFILE}_s${SEED}"
  done
done
```

Use macro-F1 first, then accuracy, per-label F1, parse failures, and
correction/degradation transitions.

### 2. Structured-memory ablation

Keep the English question and debate prompts fixed. Compare
`stance_v2_en_generic_memory` with `stance_v2_en` using the same items, seeds,
and shared Round 0. Report paired transitions, peak context, executed calls,
runtime, and unsupported/misattributed evidence in traces.

### 3. Agent scaling

Compare 3 and 5 agents on the same 60 validation items. Smoke-test 7 agents on
30 items before expanding.

### 4. Full validation, then frozen test

Run the best two configurations on all 199 validation articles. Freeze prompt,
decoding, agent count, and tie handling before the 1,001-item test. Since the
splits are issue-disjoint, test tuning would invalidate cross-issue evaluation.
Report accuracy, macro/per-label F1, issue-level averages, genre slices,
confusion matrices, context compression, calls, and runtime.

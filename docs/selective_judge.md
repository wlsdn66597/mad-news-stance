# Offline consensus and selective judge

This workflow separates the effect of the saved debate trajectory from the effect of the final decision rule. It never changes the original Phase 2 result JSON.

## Inputs

- Full Phase 2 result JSON (`--input-result`)
- Optional 104-item transition export (`--transition-file`)
- Original K-News-Stance JSON (`--data-path`) when article fields are absent from the saved trace

## 1. Offline aggregation (no model calls)

```bash
python scripts/evaluate_consensus.py \
  --input-result results/phase2/stance_qwen_test_n1001_stance_minimal_en_d0_s6000_a3_r2_qwen8_test_full_n1001_s6000.json \
  --transition-file stance_consensus_transition_cases.json \
  --aggregation-method round0_fallback \
  --output-dir results/consensus
```

Run again with `cross_round_pooled_vote`, `free_mad_exact`, or `current_final_majority`. `exploratory_neutral_boundary_fallback` is post-hoc exploratory analysis and must not be reported as a preregistered main result.

`free_mad_exact` implements Algorithm 1's trajectory score from *Free-MAD: Consensus-Free Multi-Agent Debate* (Findings of ACL 2026), with weights `(20,25,30,20)` and round correction `1/(k+1)`. The paper's random tie is made reproducible with a per-item seed. This reproduces the scoring decision mechanism offline; it is not a rerun of the complete Free-MAD debate protocol.

## 2. Selective judge on the fixed 104-item instability subset

Judge-only (article only):

```bash
python scripts/run_selective_judge.py \
  --input-result results/phase2/stance_qwen_test_n1001_stance_minimal_en_d0_s6000_a3_r2_qwen8_test_full_n1001_s6000.json \
  --transition-file stance_consensus_transition_cases.json \
  --data-path data/k-news-stance_nosegment.json \
  --aggregation-method current_final_majority \
  --judge-enabled \
  --judge-model Qwen/Qwen3-8B \
  --judge-trigger instability \
  --judge-input-mode article_only \
  --judge-temperature 0 \
  --judge-order-seed 7001 \
  --judge-max-new-tokens 384 \
  --judge-max-retries 1 \
  --output-dir results/selective_judge
```

Judge+debate uses the same command with `--judge-input-mode debate_trace`. Candidate order is anonymized and shuffled deterministically. Vote counts, current predictions, confidence, and gold labels are not included in the judge input. Gold is joined only after inference for evaluation. A malformed response is repaired once; a second failure falls back to the deterministic Round 0 rule.

For a strict paired comparison, use the same model, trigger, and order seed, then run:

```bash
python scripts/compare_judge_modes.py \
  --article-only results/selective_judge/ARTICLE_ONLY.items.json \
  --debate-trace results/selective_judge/DEBATE_TRACE.items.json \
  --output results/selective_judge/judge_mode_comparison.json
```

On the 104-item subset, Round 0 fallback corrects 47 and degrades 37 (net +10). Therefore a selective judge must correctly resolve at least 48 of 104 items to exceed that fallback's number of corrected items. Report overall and triggered-subset accuracy/macro-F1, per-class scores, confusion matrices, wrong-to-correct/correct-to-wrong counts, neutral/polar boundary changes, exact McNemar, bootstrap CI, judge calls, tokens, latency, and parse failures.

## Verification

```bash
python -m unittest tests.test_consensus tests.test_selective_judge -v
STANCE_TRANSITION_FILE=stance_consensus_transition_cases.json \
  python -m unittest tests.test_consensus.TransitionExportIntegrationTest -v
```

## Consensus round in front of this judge

To let the original agents reconsider once before the judge is called, see
[selective_consensus.md](selective_consensus.md). It reuses this judge, prompt,
parser, retry and fallback unchanged, and can reuse a run's `.items.json` as a
prediction cache.

Reference: Selective Judge, ACL 2025: https://aclanthology.org/2025.acl-long.1210/
Reference: Free-MAD, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1600/

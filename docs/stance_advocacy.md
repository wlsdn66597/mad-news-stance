# Stance-advocacy debate with a contrastive judge

One agent is assigned each label in advance and builds the strongest
article-grounded case for it. A judge then reads the article plus the three
anonymized, shuffled cases and decides.

## Why this shape

Free-form sampling makes the three agents share one bias, so they rarely put the
gold label on the table at all. Measured on the saved EXAONE run: of 497 items
still wrong after four rounds, **441 (88%) never had the gold label proposed by
any agent in any round**. No aggregation rule can repair those. Assigning one
agent per label puts the gold label among the candidates for every item.

It is also cheaper than debate: `n_labels + 1 = 4` generations per item
(`+ n_labels` with `--rebuttal`), against 6 for a two-round debate and 12 for
four rounds.

## Sources

- **Tree-of-Counterfactual prompting** (Weinzierl & Harabagiu, ACL 2024) is the
  backbone: generate one rationale per stance value, then pick a winner with
  contrastive verification. On SemEval-2016 Task 6A its macro-F1 goes 67.9
  (direct) → 70.5 (CoT) → 77.1 (ToC) with GPT-4.
- **PREDICT** (Park et al., EMNLP 2024) supplies the rebuttal round, where each
  side may refute or concede after seeing the opposing case, and the JSON judge
  verdict.
- **MAD** (Liang et al., EMNLP 2024) supplies the debater/judge separation.
- **M-MAD** (Feng et al., ACL 2025) motivates decoupling the decision into
  separate agents before synthesis; its ablation attributes most of the gain to
  that decoupling rather than to the debate itself.

## Prompt style is an ablation, not a choice

The published prompts are short: ToC's Chain-of-Explanation system prompt is 34
words and its contrastive-verification judge 82; PREDICT's debater is 32 and MAD's
16. This repository's own stance profile is 32 words, and every measured run uses
it. A longer, more structured advocate prompt would therefore confound "assigning
stances helps" with "a longer instruction helps", so both are available and the
short one is the default.

`--prompt-style toc` (default) keeps the published lengths: 33-word advocate,
68-word judge, no extra output fields.

`--prompt-style structured` is the variant with two deliberate departures from ToC:

1. Each advocate must also report the strongest counter-evidence and how well
   the article supports its assigned stance (`weak|moderate|strong`). Every
   advocate is fluent by construction, so the judge needs a signal other than
   persuasiveness, and the declared support doubles as a routing signal.
2. The judge is told explicitly that the analyses are assigned advocacy rather
   than independent opinions, and candidate order is shuffled per item with a
   stable seed. ToC does neither.

## Running

```bash
python scripts/run_advocacy_judge.py \
  --config config/phase2_exaone_stance_minimal_en.yaml --model exaone \
  --split test --n 1001 --data-seed 0 --run-seed 6000 \
  --baseline-result results/phase2/stance_exaone_test_n1001_stance_minimal_en_d0_s6000_a3_r2.json \
  --baseline-method majority \
  --output-dir results/advocacy
```

`--prompt-style structured` runs the variant; the two differ only in prompt
wording, so the pair isolates the instruction from the architecture.
`--rebuttal` adds the PREDICT-style second round. `--limit N` runs the first N
items. Runs are item-resumable: re-running the same command reuses
`<prefix>.items.json` and only processes what is missing.

The item order comes from `Stance.load(split, n, seed=data_seed)`, so with the
same `--split/--n/--data-seed` the run is paired item-for-item with the phase2
results and `--baseline-result` reports exact McNemar against them.

## Outputs

`<output-dir>/<prefix>.items.json`, `.summary.json`, `.config.json`, `.csv`.

Per item: `assigned_stances`, every advocate case with its `analysis`,
declared `support` and `stated_label`, `peer_orders`, `candidate_order`, the
judge's raw and parsed output, `pred`, `pred_source` (`judge` / `fallback`) and
`correct`.

The summary carries accuracy, macro-F1, per-class P/R/F1, confusion matrix,
prediction-source counts, per-stage call/token/latency cost, the paired
comparison against `--baseline-method`, and a compliance block.

## What to check first

- **Prompt style.** With `--prompt-style toc` the advocates are not asked for a
  support level, so `declared_support` is empty and a judge failure falls back
  deterministically instead of by declared support.
- **Compliance.** `compliance.stated_label_mismatch` counts advocate answers
  whose own stated label contradicts the stance they were told to argue. If a
  small model refuses the assigned side often, the design's premise is broken.
- **Neutral advocacy.** Arguing for the absence of a stance is structurally
  harder than arguing for a polarity; watch the per-class neutral scores and the
  `declared_support` distribution for `neutral`.
- **Position bias.** Re-run with a different `--order-seed`; the judge should
  not track candidate position.
- **Fallbacks.** When the judge returns nothing there is no majority to fall
  back on, so the label whose advocate declared the strongest support wins
  (`fallback_reason`). A high fallback count invalidates the comparison.

## Baselines to report alongside

The comparison that matters is against `majority`, not against debate: it is
the strongest baseline measured in this repository. Match the generation budget
too — advocacy uses 4 calls per item, so `majority --n-agents 4 --k 4` is the
compute-matched control.

The other informative control already exists: a judge fed *free-form* agent
analyses (`run_selective_judge.py --judge-input-mode debate_trace`) scored below
the article-only judge. This experiment asks whether *assigning* the stances
makes those analyses useful where free-form ones were not.

# Stance-advocacy debate with a contrastive judge

One agent is assigned each label in advance and builds the strongest
article-grounded case for it. The agents exchange cases for one or more rebuttal
rounds, and a judge then reads the article plus the three anonymized, shuffled
final cases and decides.

The base configuration is **two rounds** — independent cases, then one rebuttal
round — matching PREDICT's fixed two-round debate. `--rounds` raises the limit.

## Why this shape

Free-form sampling makes the three agents share one bias, so they rarely put the
gold label on the table at all. Measured on the saved EXAONE run: of 497 items
still wrong after four rounds, **441 (88%) never had the gold label proposed by
any agent in any round**. No aggregation rule can repair those. Assigning one
agent per label puts the gold label among the candidates for every item.

Cost is `n_labels * rounds + 1` generations per item: 4 at one round, **7 at the
two-round base**, 13 at four rounds. A two-round debate costs 6 and a four-round
debate 12.

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

## No self-reported confidence

An earlier draft asked each advocate to rate how strongly the article supported
its assigned stance (`weak|moderate|strong`) and used that to break ties when
the judge failed. It was dropped. None of the source papers ask for it, verbal
confidence from a 1.2B model is not a trustworthy signal, and it accounted for
much of the prompt-length inflation. The judge always decides, so no tie-break
was needed in the first place.

## Consensus is never checked

The three agents disagree by construction, so there is no "consensus failed"
state and no consensus rule. ToC, PREDICT and MAD all work the same way: the
judge always produces the answer. Only M-MAD has a non-consensus rule, and that
is to end its two-agent per-dimension debates early.

The fallback in this pipeline is therefore not a consensus rule. It covers one
thing: a judge whose output carried no stance label in any attempt. That is a
broken generation, so the label is chosen deterministically from the item id and
the count is reported. A run with a non-trivial fallback count is invalid.

## Prompt style is an ablation, not a choice

The published prompts are short: ToC's Chain-of-Explanation system prompt is 34
words and its contrastive-verification judge 82; PREDICT's debater is 32 and MAD's
16. This repository's own stance profile is 32 words, and every measured run uses
it. A longer, more structured advocate prompt would therefore confound "assigning
stances helps" with "a longer instruction helps", so both are available and the
short one is the default.

`--prompt-style toc` (default) keeps the published lengths and the published
output contract: a 33-word advocate, a 76-word judge that writes prose and names
the label in its final line, and no extra output fields. ToC never asks for JSON,
and requiring it broke every judge call on the first EXAONE-1.2B run: the model
followed "discuss your reasoning step-by-step" and never emitted an object, so
all 20 items fell back. The final line is read with the same parser the debate
methods use.

`--prompt-style structured` is the variant with two deliberate departures from ToC:

1. Each advocate is asked for named evidence fields, including the strongest
   counter-evidence against its own assigned stance.
2. The judge is told explicitly that the analyses are assigned advocacy rather
   than independent opinions, returns the strict JSON schema this repository's
   selective judge already uses, and candidate order is shuffled per item with a
   stable seed. ToC does none of these.

A JSON answer that fails validation is searched for a stance label before the
item is handed to the fallback, since a truncated object usually still carries
one (`label_recovered_from_text`).

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
wording, so the pair isolates the instruction from the architecture. `--rounds N`
sets the debate length (2 is the base). `--limit N` runs the first N items. Runs are item-resumable: re-running the same command reuses
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

- **Fallback count.** `pred_source` should be `judge` for essentially every
  item. A high `fallback_used` means the judge is not producing labels, not that
  the debate was inconclusive.
- **Compliance.** `compliance.stated_label_mismatch` counts advocate answers
  whose own stated label contradicts the stance they were told to argue. If a
  small model refuses the assigned side often, the design's premise is broken.
- **Neutral advocacy.** Arguing for the absence of a stance is structurally
  harder than arguing for a polarity; watch the per-class neutral scores and the
  `declared_support` distribution for `neutral`.
- **Position bias.** Re-run with a different `--order-seed`; the judge should
  not track candidate position.
- **Drift across rounds.** `label_trajectory` records, per round, whether each
  agent's text still reads as its assigned label. It never affects the decision,
  but it shows whether advocates concede as the round limit rises.

## Baselines to report alongside

The comparison that matters is against `majority`, not against debate: it is
the strongest baseline measured in this repository. Match the generation budget
too — advocacy uses 4 calls per item, so `majority --n-agents 4 --k 4` is the
compute-matched control at one round; at the two-round base the match is
`majority --n-agents 7 --k 7`.

The other informative control already exists: a judge fed *free-form* agent
analyses (`run_selective_judge.py --judge-input-mode debate_trace`) scored below
the article-only judge. This experiment asks whether *assigning* the stances
makes those analyses useful where free-form ones were not.

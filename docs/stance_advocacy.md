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

## The judge must be allowed to reason

The judge closing instruction was first written as *"Your final line should be
exactly: Final stance: <label>"*. A 1.2B judge read that as "output only this
line": on the full 1001-item run, **841 of 1001 judge answers were under 30
characters** (median 24, e.g. `Final stance: supportive`) and the whole judge
stage took 2.4 minutes. The contrastive verification that the method depends on
never ran, and the result was 4.8 points below majority voting.

The wording is now ToC's own — *"Your final sentence should include only one
possible stance value"* — which names the value after the discussion instead of
replacing it. `judge_diagnostics.judge_output_chars` reports the median length
and how many answers were bare verdicts, so this failure is visible in the
summary rather than only in a post-hoc script.

Because the verdict now sits in prose, it is read from the closing sentences
(`parse_final_verdict`) rather than from the last label mentioned anywhere: a
label named while rejecting a rationale must not win.

## Re-running only the judge

A judge change does not need the advocates regenerated. `--reuse-advocacy
<previous .items.json>` takes the saved cases and runs the judge alone, which
turned a 10.7-hour run into a judge-only pass. Output files gain a `_rejudge`
marker so the two do not collide.

A judge-only pass **does not load the advocate model**: it has nothing to run,
and loading it both wastes VRAM and forces the judge to inherit the advocate
config's quantization. Pass `--judge-load-in-4bit/--no-judge-load-in-4bit` to
set the judge's quantization independently of the yaml config.

Reuse is only valid if the cases came from the run you think they did, so the
sibling `.config.json` is checked against this run's arguments. Enforced:
`--split`, `--n`, `--data-seed`, `--run-seed`, `--order-seed`, `--rounds`,
`--limit` — the split, the item sampling, the label assignment and the round
count. Reported but never enforced: `--config`, `--model`, `--prompt-style` and
the advocate decoding settings, because in a judge-only pass no advocate runs
and those describe *this run's judge*. Judging EXAONE cases under a Qwen config
is the intended use, not a mismatch; the summary records who actually wrote the
cases in `config.advocate_model_id`. Settings resolved from the yaml rather than
the command line are reported as unchecked, and `--allow-reuse-mismatch`
downgrades the enforced check to a warning. Per-round advocacy costs in a rejudge summary are inherited from the
source run (`cost.advocacy.inherited_from_reuse`), not produced by that pass.

`--judge-round last` writes no filename tag, so a re-run lands on the same
prefix as an older run of the same configuration and `--resume` would adopt its
rows. That happened once and silently compared a fresh round-0 pass against a
`last` pass restored from a run made before the strict-JSON fix. Resume now
validates the sibling `.config.json` against the judge-relevant arguments and
refuses rather than mixing two experiments in one file; use `--no-resume` or a
different `--output-dir`. When two runs really were made by different judge
code, `scripts/compare_methods.py --exclude-parse-errors` drops the items where
either run hit a parse error, which are the only ones the change could have
moved, leaving the rest a clean paired comparison.

Two knobs exist for judge-only ablations:

- `--judge-round 0|last` chooses which round the judge reads. 84.7% of round-1
  outputs mention "Analyst" and round 1 is shorter than round 0 (1187 vs 1489
  chars): the agents argue about each other rather than about the article, and
  the judge only ever saw the last round. `--judge-round 0` gives it the
  independent cases instead, at judge-only cost.
- `--judge-order-seed` reshuffles the candidates without touching the label
  assignment the cases were written under, so position bias can be re-tested on
  the same cases. The summary reports the winning-position counts with a
  chi-square test (`winning_candidate_position_test`).

```bash
python scripts/run_advocacy_judge.py   --config config/phase2_exaone_stance_minimal_en.yaml --model exaone   --split test --n 1001 --data-seed 0 --run-seed 6000   --reuse-advocacy results/advocacy/<previous>.items.json   --baseline-result results/phase2/<majority run>.json --baseline-method majority   --output-dir results/advocacy
```

## Prompt style is an ablation, not a choice

The published prompts are short: ToC's Chain-of-Explanation system prompt is 34
words and its contrastive-verification judge 82; PREDICT's debater is 32 and MAD's
16. This repository's own stance profile is 32 words, and every measured run uses
it. A longer, more structured advocate prompt would therefore confound "assigning
stances helps" with "a longer instruction helps", so both are available and the
short one is the default.

`--prompt-style toc` (default) keeps the published lengths, input format and
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

A JSON answer that fails validation is **retried first**; only on the final
attempt is the text searched for a stance label, since a truncated object
usually still carries one (`label_recovered_from_text`). Salvaging on the first
failure, as an earlier version did, accepts malformed output silently and
reports the run as clean: `retries=0, fallback=0` in a saved summary says
nothing unless `judge_diagnostics.label_recovered_from_text` and
`judge_parse_errors` are also zero. Both are now counted from the attempt log,
so a prose (`toc`) verdict is no longer miscounted as a recovery.

The ToC repair turn resends the article and the three rationales along with the
invalid output. Asking a judge to "answer again" with only its own broken output
in context leaves it nothing to judge.

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

`<output-dir>/<prefix>.items.json`, `.summary.json`, `.config.json`, `.csv`,
`.qualitative.md`.

`.qualitative.md` groups items by (gold → predicted) and, for a few per cell,
prints the headline, which stance each agent was told to argue, the judge's own
reasoning, and the opening of the case that argued the gold label. That is
enough to tell apart the two failure modes: the judge ignored a good argument,
or no advocate ever made one. `--qualitative-per-cell` sets the sample size.

Per item: `assigned_stances`, every advocate case with its `analysis`,
declared `support` and `stated_label`, `peer_orders`, `candidate_order`, the
judge's raw and parsed output, `pred`, `pred_source` (`judge` / `fallback`) and
`correct`.

The summary carries accuracy, macro-F1, per-class P/R/F1 and the confusion
matrix; a `runtime` block with UTC start/end, wall clock, and how many items were
generated rather than resumed; per-stage and **per-round** call/latency/token
cost; `judge_diagnostics` (which candidate position won, retry count); the paired
comparison against `--baseline-method` with exact McNemar, a bootstrap CI and
neutral/polar boundary switches; and a compliance block.

Progress lines carry a running seconds-per-item and an ETA, so a long run can be
watched with `tail -f`.

## What to check first

- **Fallback count.** `pred_source` should be `judge` for essentially every
  item. A high `fallback_used` means the judge is not producing labels, not that
  the debate was inconclusive.
- **Compliance.** `compliance.declared_defection` counts advocates whose own
  explicit "Final stance:" line contradicts the stance they were told to argue.
  Only an explicit declaration counts. A label that merely appears last in the
  prose is reported separately as `last_mention_differs`: an advocate that
  argues its side and then names counter-evidence ends on another label word all
  the time, and counting those as defections put the first smoke run at 26/60
  when the real rate was far lower.
- **Neutral advocacy.** Arguing for the absence of a stance is structurally
  harder than arguing for a polarity; watch the per-class neutral scores and the
  `declared_support` distribution for `neutral`.
- **Position bias.** Re-judge the same cases with a different
  `--judge-order-seed`; the judge should not track candidate position. Measured
  on the Qwen3-8B structured judge: A/B/C = 383/328/290, chi-square 13.10, df 2,
  p = 0.0014, with the order already shuffled per item. That is the judge
  preferring the first slot.
- **Advocacy input tokens.** `cost.advocacy.input_tokens` counts the rendered
  chat template, so it includes the system prompt, the article and the agent's
  own earlier turns. Any summary written before that fix counted only the newest
  user turn and understates round 1 by most of the prompt; do not compare the
  two.
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

## The offline ceiling: are the cases separable at all?

`scripts/advocacy_oracle.py` needs no GPU. Because one agent argues each label,
the gold label is on the ballot for every item and a literally perfect judge
scores 1.0 — a vacuous ceiling. The script measures the question that is not
vacuous: **is the case arguing gold distinguishable from the two that are not?**

It ranks the three saved cases by surface features — length, quoted passages,
hedging cues, mentions of other analysts, and whether the advocate explicitly
conceded its assigned side — and reports how often the gold case ranks first,
with an exact two-sided binomial p-value against 1/3. It also reports how often
each feature's pick agrees with the pick the judge actually made, which shows
what the judge was tracking, and the winning-position chi-square.

```bash
python scripts/advocacy_oracle.py results/advocacy/<run>.items.json --round saved
python scripts/advocacy_oracle.py results/advocacy/<run>.items.json --round 0
```

If no feature beats chance and the judge agrees with none of them, the cases
carry no recoverable signal and no judge change will help.

## Selective advocacy: keep the vote, commission only what is missing

`src/selective_advocacy.py` + `scripts/run_selective_advocacy.py`.

Assigning one agent per label makes every item 1:1:1 by construction, which
discards the only aggregate signal a small model reliably produces: how many
independent agents converged on a label. The measured consequence is a judge
that does not behave like a repair step at all — against majority's 487 correct
and 514 wrong items, the EXAONE structured judge kept 40.0% of the correct
answers and fixed 38.3% of the errors (independent of majority), and the
Qwen3-8B judge kept 29.4% and fixed 53.1% (anti-correlated with it).

Selective advocacy keeps the existing free-form debate and its vote, and
commissions a counterfactual case only for labels no agent proposed, only on
items where the vote is unstable or a label is missing. The judge is given, per
label, either the rationale of the agents that chose it *with their count* or a
rationale marked as commissioned precisely because nobody chose it.

**The trigger is the vote, not the missing label.** "Commission a case for a
label nobody proposed" sounds selective and is not. With three agents and three
labels a missing label is the normal case: on the EXAONE test run, 830 of 1001
items are unanimous (two labels missing), 162 are 2:1 splits and only 9 are
1:1:1. Triggering on a missing label fires on 99% of items and overrides the
vote precisely where the vote is strongest. Unanimity is not accuracy — majority
scores 0.4865 on those same items — but a commissioned counterfactual is not
evidence against a 3:0 vote either.

- `--trigger instability` — tied final round, or round 0 overturned (the
  existing selective-judge trigger). 42/1001 on the EXAONE run.
- `--trigger non_unanimous` — a 2:1 split or a tie. 171/1001.
- `--trigger unstable_or_split` (default) — the union. 174/1001, about
  0.35 generations/item.
- `--trigger missing_label` — kept as the ablation that shows why it is the
  wrong trigger: 992/1001, 2.82 generations/item.
- `--trigger all` — judge everything.
- `--trigger-scope last|any_round` — whether a label proposed only in an earlier
  round counts as proposed.
- `--dry-run` prints the trigger counts and the exact call budget without
  loading a model, and warns when the trigger fires on more than half the items.
  Use it before committing GPU time.

Cost is zero extra calls on stable items and at most (missing labels + 1) on the
rest, against 7 per item for full two-round advocacy. Report it against
`majority` at a matched budget, as always.

```bash
python scripts/run_selective_advocacy.py \
  --input-result results/phase2/<debate run>.json \
  --data-path data/k-news-stance_nosegment.json \
  --config config/phase2_qwen8_stance_minimal_en.yaml --model qwen \
  --advocate-model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --trigger unstable_or_split --dry-run
```

`judge_diagnostics.picked_commissioned` counts how often the judge went with a
counterfactual nobody had proposed, and how many of those were right. If the
judge almost never picks one, the commissioned cases are not the bottleneck; if
it picks them often and is wrong, they are persuasive noise.

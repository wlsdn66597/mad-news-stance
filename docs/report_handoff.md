# Report handoff: MAD on K-News-Stance

Everything needed to continue from a clone. Repo `wlsdn66597/mad-news-stance`,
branch **`codex/selective-judge-consensus`**.

**Code is in git; results and data are not.** They live on the GPU server at
`~/mad/mad-news-stance` (RTX 4090). `.gitignore` excludes `results/*` and
`data/*`, so a fresh clone has every script and no numbers. The measurement log
with every figure quoted below is in [`HANDOFF.md`](../HANDOFF.md) section 0; the
design rationale for the advocacy line is in
[`docs/stance_advocacy.md`](stance_advocacy.md).

---

## 1. What the report argues, and which run fills each section

The through-line is one question answered six times: *published MAD gives little
on this task — why, and what does help?* Each section exists because it forced
the next one.

| § | Section | Run that fills it | Status |
|---|---|---|---|
| 1 | MAD applied as published | `single / majority k=3 / debate 2R / debate 4R` | done, 2 seeds |
| 2 | Why it gives little | `analyze_agent_diversity.py` on §1 | offline, done |
| 3 | Force one stance per agent | advocacy + judge, both EXAONE | done, s6000 |
| 4 | Differentiate roles instead | persona debate 2R + 4R | 4R done, 2R pending |
| 5 | Judge only where they disagree | persona 2R/4R + judge on non-unanimous | pending on one backbone |
| 6 | Choose the final setting | 2R+judge vs 4R+judge from §5 | pending |

**Section 2 is the hinge.** It is not a result table, it is the diagnosis that
makes sections 3 and 4 make sense: `single` and `majority k=3` score
*identically* (0.4865 both) because three samples of one prompt are near-copies.
Sections 5-2 through 5-5 of the older notes (selective advocacy, judge prompt
sweep, routing simulation, neutral prompting) are **not** separate sections;
cite them as supporting diagnostics where they explain a failure.

### The one-line story

> Published MAD helps a little → the agents are near-copies and the gold label
> is often absent from every candidate → forcing one stance per agent widens the
> candidates but destroys the vote and the grounding → differentiating the
> *reading role* instead widens them without either loss → the disagreement that
> survives marks the hard items → judging only those gives the best result.

---

## 2. Everything runs from one script

```bash
SEEDS=6000 JUDGE_ROUNDS=last bash scripts/run_report_repeats.sh
```

Stages 1, 3, 4, 5 for one seed. Stage 2 and 6 generate nothing. Every stage
resumes per item, so re-running a finished seed costs nothing.

| env | default | meaning |
|---|---|---|
| `SEEDS` | `6000 6001 6002` | run seeds, space separated |
| `STAGES` | `1 3 4 5` | subset to run |
| `JUDGE_MODEL` | empty | empty keeps the judge on EXAONE; set it only for a labelled "MAD + strong judge" row |
| `JUDGE_INPUT` | `debate_trace` | `article_only` is the control |
| `JUDGE_ROUNDS` | `all` | `last` or `first` show the judge one round instead of all |
| `DRY_RUN` | `0` | print the plan, load nothing |

**Every model is EXAONE-4.0-1.2B, judges included.** Anything else has to be
labelled in the caption, because it stops being a single-model result.

Cost per seed: stage 1 = 15 generations/item, stage 3 = 7, stage 4 = 15,
stage 5 ≈ 0.56 judge calls/item. About 37k generations for a full fresh seed,
roughly a day; a seed with most stages already on disk is a few hours.

---

## 3. Metrics every table must carry

**Accuracy alone is not enough.** The classes are near balanced (test: 348
oppositional, 330 neutral, 323 supportive) but the models are not — EXAONE
recovers ~90% of supportive and ~20% of neutral — so a method can move accuracy
by trading one class for another.

| metric | why | where |
|---|---|---|
| Accuracy | headline | `collect_report_table.py` |
| **Macro-F1** | catches class trading | same |
| **Neutral recall** | the class everything fails on | same |
| **Paired exact McNemar** | every comparison | `compare_methods.py` |
| Cost, generations/item | the honest axis for any judge row | recorded per stage |

Diversity tables (§2, §4) additionally need, per round: pairwise agreement,
unanimity count, label spread, candidate pool, majority accuracy, neutral
recall, agent label change rate — all from `analyze_agent_diversity.py`.

Judge tables (§5) need: trigger ratio, subset accuracy before and after,
W→C, C→W, net — from `collect_report_table.py` or the run's own `.md`.

### Two rules that are easy to get wrong

1. **Never compare absolute accuracy across seeds.** Seed variance is 11-16
   items in 1001 (majority 487→471, debate 504→493 between 6000 and 6001) while
   the *paired within-seed* delta held at +17 and +22. Report the paired delta.
2. **`compare_methods.py` prints `A -> B` net as the change going to B.** A
   negative net means B is worse. This has been misread more than once.

---

## 4. Seeds: the current mess and the fix

Numbers so far come from different seeds, which is the single biggest
methodological weakness of the current draft.

| run | 6000 | 6001 | 6002 |
|---|---|---|---|
| phase2 r2 (single/majority/debate) | yes | no | no |
| phase2 r4 (debate) | yes | yes (+majority) | no |
| persona r2 | **no** | yes | no |
| persona r4 | yes | yes | no |
| advocacy + EXAONE judge | yes | no | no |
| selective judge (Qwen, personas) | no | yes | no |
| majority k=12 | yes | no | no |

**advocacy is s6000 and the persona/judge results are s6001.** Putting them in
one table invites the objection that the difference is seed noise.

**Fix, in order:**

1. Complete **s6000** for every stage — that is what
   `SEEDS=6000 bash scripts/run_report_repeats.sh` does, and most of it resumes.
   Report the whole pipeline at one seed.
2. Only then repeat. Three seeds is ~111k generations, three to four days.
   Repeat the rows whose differences sit near the ±16 noise floor; several
   results (advocacy at −38 to −95) are far outside it and do not need it.
3. Validation split (n=199) is for **choosing** prompts, test (n=1001) for
   reporting. The two-step neutral profiles were selected that way; keep it.

---

## 5. Results already measured

Full log in `HANDOFF.md` section 0. The load-bearing ones:

- **Debate beats majority only for the weak model, only at 4 rounds.** EXAONE
  +17 (p=0.027, s6000) and +22 (p=0.003, s6001); two rounds +2 (ns). Qwen3-8B
  loses 10 at both 2 and 4 rounds, and rounds 3-4 change 90 predictions with net
  exactly 0.
- **The gain sits where the agents disagreed at round 0** — +15 (p=0.036) and
  +20 (p=0.003) on the 2:1 items, +3 and +3 on the unanimous ones. Found on
  6000, confirmed on 6001 without refitting.
- **The ensemble is three copies.** Round 0: per-agent 0.4735/0.4725/0.4745,
  pairwise agreement 0.8651, unanimous on 800/1001, all three wrong on 45.4%
  where independence would give 14.6%. Candidate pool 0.5465 caps every
  aggregation rule.
- **Forcing one stance per agent loses everywhere**, −38 to −95 against
  majority, and a judge that reasons more does worse. 81% of the advocates'
  Korean quotes are not in the article.
- **Personas widen the pool without touching per-agent accuracy.** Validation:
  109 → 135 of 199, gained 26, lost 0, p ≈ 3e-8. Test s6001: 0.5465 → 0.6214.
  Debate on top: 0.4925 → 0.5045; at s6000, 0.5035 → 0.5265.
- **Persona majority costs three generations per item** (share_round0) and at
  s6000 scored 515 against 504 for four-round shared debate at twelve.

---

## 6. Analyses still owed, in priority order

1. **2R non-unanimous → 4R unanimous.** The items the extra rounds talk into
   agreement stop being triggered, so the judge never sees them. This is the
   explanation for 2R+judge beating 4R+judge and it is the highest-value
   remaining analysis. `analyze_consensus_shift.py`, needs §4 and §5 at one seed.
2. **Round-by-round tables** for shared and persona debate.
   `analyze_agent_diversity.py` prints agreement / unanimity / pool / majority
   accuracy / neutral recall / change rate per round.
3. **Judge input ablation**: `JUDGE_INPUT` × `JUDGE_ROUNDS`. `article_only` won
   every earlier comparison, but those traces were near-copies; persona traces
   are three different readings, and round 0 beat the last round by +19
   (p=0.016) in the advocacy diagnostic. Three conditions, ~30 min each.
4. **Macro-F1 for every condition** — many current numbers are accuracy only.
   `collect_report_table.py` fixes this in one pass.
5. **p-value for majority-shared → majority-personas at s6000** (+28,
   uncomputed). It is the evidence for the cost claim in §4 of the report.
6. Qwen `majority k=12`, for table symmetry only. Cannot change any conclusion.

### Item and type breakdowns worth reporting

`analyze_when_debate_helps.py` splits any paired comparison four ways. The first
two carry the argument:

- **Round-0 agreement** (unanimous / 2:1 / 1:1:1) — where the gain lives.
- **Candidate availability** — whether gold was proposed at round 0 at all.
  Majority is 0.0000 by construction where it was not, so this separates "debate
  found the answer" from "debate could not have". Debate introduces a label
  nobody proposed on 2.0% of such items for EXAONE and 6.25% for Qwen.
- **Gold label** — debate de-biases: oppositional +14, neutral +10, supportive
  −2 under the shared prompt; under personas the gain moves to neutral (+26,
  p<0.001).
- **Article length quartile** — flat everywhere, so long context is not the
  bottleneck. Worth one sentence to close it off.

Buckets are exploratory and not multiplicity corrected; the overall line is the
only confirmatory test. Say so.

---

## 7. Traps already hit

- The **advocacy output filename contains `stance_minimal_en`, but the advocates
  do not use that prompt.** They use the ToC prompts in `src/prompts/advocacy.py`.
- `--judge-round last` writes no filename tag, so a re-run lands on an older
  file's prefix and `--resume` adopts its rows. That contaminated one comparison
  before a guard was added.
- A judge asked for strict JSON used to salvage a label from malformed output on
  the first failure, reporting `retries=0, fallback=0` on a run whose labels had
  never parsed. Check `judge_diagnostics.label_recovered_from_text` and
  `judge_parse_errors` on any older summary.
- `git clean -fdx` on the server deletes `results/`. Never use `-x`.
- With personas, a `single` row would be agent 0 alone (the framing reader), not
  a single-agent baseline. Stage 4 no longer produces one.

---

## 8. Script map

| script | GPU | what |
|---|---|---|
| `run_report_repeats.sh` | yes | the four generating stages, over seeds |
| `run_phase2.py` | yes | single / majority / debate, `--personas`, `--n-rounds` |
| `run_advocacy_judge.py` | yes | one agent per label + judge, `--reuse-advocacy` |
| `run_selective_judge.py` | yes | judge on triggered items, `--judge-input-mode`, `--judge-rounds` |
| `run_selective_advocacy.py` | yes | commission only missing labels, `--ablation` |
| `collect_report_table.py` | no | the report tables, mean ± sd over seeds |
| `compare_methods.py` | no | paired exact McNemar across any saved runs |
| `analyze_agent_diversity.py` | no | per-round agreement, pool, per-agent accuracy |
| `analyze_when_debate_helps.py` | no | paired comparison split by four conditions |
| `analyze_consensus_shift.py` | no | what the extra rounds talk into agreement |
| `advocacy_oracle.py` | no | case separability, quote grounding audit |
| `simulate_routing.py` | no | is the trigger a good router, vs random and oracle |
| `analyze_selective_runs.py` | no | ablation comparison, gates, per-item ceiling |
| `analyze_neutral_abstention.py` | no | neutral as abstention, two-stage decomposition |

Tests: `python -m unittest tests.test_advocacy tests.test_selective_advocacy
tests.test_selective_judge tests.test_consensus tests.test_selective_consensus
tests.test_stance_prompt_profiles`

# Handoff: stance advocacy + judge on K-News-Stance

Everything needed to pick this up cold. Repo `wlsdn66597/mad-news-stance`,
branch **`codex/selective-judge-consensus`**. Results and data live on the GPU
server (`~/mad/mad-news-stance`, RTX 4090), not in git.

Shared setup for every number below: K-News-Stance `test` split, n=1001,
`--data-seed 0 --run-seed 6000`, prompt profile `stance_minimal_en`,
temperature 1.0, `max_new_tokens` 1024, judge temperature 0. Runs are paired
item-for-item, so all comparisons are exact McNemar on the same items.

---

## 0. Session of 2026-08-08: the advocacy line is closed

Everything in sections 1–4 below was written before this session and is kept as
the record of how the question was reached. The experiments it proposed have now
been run. Read this section first; it overrides several of its conclusions.

### The three findings

**1. The rebuttal round contaminates the cases.** Giving the judge round 0
instead of round 1 is worth +19 items (0.4251 → 0.4044 on the 915 items where
neither run hit a parse error, p = 0.016; +21, p = 0.015 over all 1001). The
offline oracle says the same independently: ranking the three cases by hedging
picks the gold case 42.4% of the time at round 0 (p = 0.0025) and 32.8% at
round 1 (p = 0.87); quoted-passage density goes from p = 0.0053 to p = 0.84.
Mentions of "Analyst" decide 12 items at round 0 and 815 at round 1 — the agents
stop arguing about the article and start arguing about each other.

**2. Advocacy loses to majority in every configuration.** Best case is the
round-0 judge at 0.4366, which is still −50 against majority's 0.4865
(p = 0.048). Judge capacity is not the bottleneck: Qwen3-8B, which demonstrably
reasoned (0 bare verdicts, median 483 chars), scores 0.4156, below the
EXAONE-1.2B judge that emitted a bare label 895 times out of 1001 (0.4486).
Every judge is anti-correlated with majority rather than repairing it — the
round-0 judge keeps 31.6% of majority's correct answers and fixes 55.1% of its
errors.

**3. The analyses have no evidential value; their whole measured effect is
anchoring.** Selective advocacy (commission a case only where the agents split,
174/1001 items, 0.34 generations/item) scores 0.5175 against majority's 0.4865.
But the ablations attribute all of it to routing, not to the machinery:

| condition | judge sees | acc |
|---|---|---:|
| `article_only` | the article | **0.5355** |
| `article_first` | its own verdict first, then the analyses | 0.5345 |
| `article_first --no-verify-quotes` | same, quotes untagged | 0.5345 |
| `no_votes` | analyses, no vote count | 0.5215 |
| `full` | analyses + origin + vote count | 0.5175 |
| `no_commissioned` | only the analyses agents wrote | 0.5165 |
| debate / majority | | 0.4885 / 0.4865 |

Analyses shown *before* the judge forms a view cost 18 items against showing
none (`article_only → full`, p = 0.006). Shown *after* it has already answered
from the article, they cost nothing and change nothing: the judge revised its
own answer on 4 of 174 items, 2 right and 2 wrong, and the tagged and untagged
runs produced **identical predictions on all 1001 items**. Commissioning
contributes nothing (`full → no_commissioned`, +1, p = 1.000) and neither does
the vote count (`full → no_votes`, −4, p = 0.627).

Caveat to carry: the second-turn instruction was deliberately conservative
("change your answer only if the article itself shows you were wrong"), so some
of that inertia is by construction. A neutral instruction would move the judge
more — in the direction that was measured to cost 18 items.

### Why, mechanically

The advocates fabricate their evidence. Of 9,596 Korean-script quoted passages
across the round-0 cases, 16.8% appear in the article exactly, 18.3% ignoring
spacing and punctuation, and **19.3% share even their first twelve characters**
with it. (The 2,033 Latin-script quotes verify at 0.1% and are translations, not
inventions.) The judge has no way to tell — and when it is told, via Khan et
al.'s verified/unverified tagging, it does not use the information.

Khan et al. (ICML 2024) explain the inversion. Debate helps their judge because
the judge **cannot see the passage** and verified quotes are its only grounding.
Our judge reads the full article, so the analyses add no grounding and only a
false one.

### What is real: agent disagreement as a router

Non-unanimity among the weak agents predicts weak-model failure. On the EXAONE
run, agents are unanimous on 830/1001 items, split 2:1 on 162, tied on 9. On the
174 that are not unanimous or not stable, the vote scores 0.4023 against 0.5067
elsewhere, and handing just those to Qwen3-8B with the article alone gives
0.5355 overall (+47 over majority, w2c 62 / c2w 15) at 0.174 extra calls per
item. Offline routing simulation confirms the criterion rather than the budget:
random routing of the same 174 items matched or beat it in 0.2% of 5,000 draws.
The revision trajectory adds nothing on top of the final vote (`instability`
alone is 42 items, random beats it 31.3% of the time; the union is worse than
`non_unanimous` alone).

### Unrelated thread, now answered: Qwen debate at 4 rounds

Qwen does **not** flip. Rounds 2 and 4 both score 0.5894 against majority k=3's
0.5994 (−10, p = 0.387 and p = 0.440). Rounds 3–4 change 90 of 1001 predictions
with net exactly 0 — 35 each way. So round-conditionality is not a general
property: debate beats majority only for EXAONE-1.2B and only at 4 rounds
(+17, p = 0.027). The claim to carry is that debate's value **shrinks as the
model improves**, not that enough rounds make it work.

`majority k=12` for Qwen is still unrun (its file has only 758 shared round-0
answers and no results). It can no longer change anything: debate r4 already
loses to the cheaper k=3.

### Replicated at a second seed (2026-08-09)

The headline holds and the mechanism holds with it.

| seed | majority | debate r4 | net | p |
|---|---:|---:|---:|---:|
| 6000 | 0.4865 | 0.5035 | +17 | 0.027 |
| 6001 | 0.4705 | 0.4925 | +22 | 0.003 |

The concentration in split votes replicates too: on round-0 2:1 items the net
is +15 (p = 0.036, 189 items) at seed 6000 and +20 (p = 0.003, 198 items) at
seed 6001, against +3 and +3 on the unanimous ones. That bucket was found by
looking at seed 6000 and confirmed on seed 6001 without refitting, and the
second p-value survives Bonferroni over the twelve buckets. The label pattern
repeats as well -- oppositional +14, neutral +10, supportive −2.

Seed variance is now measured: absolute accuracy moves 11 to 16 items between
seeds (majority 487 → 471, debate 504 → 493) while the paired within-seed delta
stays at +17 and +22. Report the paired delta, never the absolute level.

**Defensible claim.** Four-round debate beats majority for EXAONE-4.0-1.2B,
mean +19.5 items over two seeds, and the gain sits in the 20% of items where
the agents disagreed at round 0. Two rounds is not enough (+2, ns) and the
effect is absent for Qwen3-8B at either round count (−10).

### Neutral cannot be reached by prompting (2026-08-09)

Perfect neutral detection is worth +24.9 points, more than everything else in
this project combined, so the three-label prompt was reframed as two decisions
-- does the article take a side, and only then which one -- and compared on the
**validation** split, one agent, one call.

| profile | acc | supportive recall | neutral recall | times it said neutral |
|---|---:|---:|---:|---:|
| `stance_minimal_en` (3-way) | 0.4874 | 0.906 | 0.101 | 16 / 69 |
| `stance_twostep_en` | 0.3618 | 0.031 | 0.855 | 151 / 69 |
| `stance_gate_en` | 0.3518 | 0.000 | 0.971 | 193 / 69 |

Both new framings lose (−25, p = 0.041 and −27, p = 0.032) and they lose the
same way: the model does not start detecting neutral, it starts defaulting to
it. The gate variant calls 193 of 199 items neutral and its supportive recall
is exactly zero. A framing change that flips the prior wholesale is evidence
that the discrimination is absent rather than weak, so this headroom is not
reachable by prompting this model. The profiles stay in the registry as the
record of that.

### Agent roles open the candidate pool (2026-08-09)

The three agents were never an ensemble. At round 0 they scored 0.4735,
0.4725 and 0.4745, agreed pairwise 86.5% of the time and were unanimous on 800
of 1001 items; all three missed on 45.4% where independent failures at those
accuracies would miss on 14.6%. That is why `single` and `majority k=3` are the
same number, and it caps the candidate pool -- how often any agent names the
gold label -- at 0.5465.

`--personas` gives each agent a distinct reading role (narrative framing,
sourcing, wording). Same three-way question, same vote, same call count; only
what each agent attends to differs, and no persona names a side.

On validation the pool went 109 → 135 of 199, **gained 26 and lost 0**, exact
McNemar p ≈ 3e-8, with per-agent accuracy unchanged (mean 0.4891 → 0.4874). The
agents did not get better, they started failing differently. It is monotone
because personas *add* labels rather than replace them: the shared ensemble puts
one label on the table, personas put about 1.5.

On test at seed 6001 it replicates, smaller: pool 0.5465 → 0.6214 (+75 items),
agreement 0.8651 → 0.7396, unanimous 800 → 624.

| seed 6001 | accuracy | neutral recall |
|---|---:|---:|
| majority, shared prompt | 0.4705 | 0.197 |
| debate r4, shared prompt | 0.4925 | 0.227 |
| majority, personas | 0.4875 | 0.285 |
| **debate r4, personas** | **0.5045** | **0.364** |

`majority shared → debate personas` is +34, p = 0.011, the best result in the
project. **Neutral recall nearly doubles**, which the two-step prompts could not
do: they only flipped which class the model defaulted to, while personas improved it
without touching the label framing at all. Under personas the debate gain moves
to neutral (+26, p < 0.001) from oppositional (+14) under the shared prompt, and
`gold never proposed at round 0` falls from 454 to 379 items.

**Second seed (6000).** Personas win again, and the cost picture is the more
interesting half.

| method | calls/item | s6000 | s6001 | mean |
|---|---:|---:|---:|---:|
| majority k=3, shared | 3 | 487 | 471 | 479 |
| majority k=12, shared | 12 | 481 | — | — |
| **majority k=3, personas** | **3** | **515** | 488 | **501.5** |
| debate r4, shared | 12 | 504 | 493 | 498.5 |
| **debate r4, personas** | **12** | **527** | 505 | **516** |

Because `share_round0` is on, `majority` uses only the round-0 answers, so the
persona majority costs **three generations per item**. At seed 6000 it scores
515 against 504 for four-round debate on the shared prompt at twelve, and on
average the two are level. Sampling the same prompt four times more (k=12)
loses six items; splitting what three agents read gains twenty-eight. The
ensemble literature's basic claim, confirmed the hard way: what an ensemble is
worth comes from independence, not from member count, and this one was three
copies.

Personas on top of debate: +23 (p = 0.096) at seed 6000 and +12 (p = 0.404) at
seed 6001. Same direction twice, mean +17.5, Stouffer combination around
p = 0.08 -- suggestive, not established.

**What is not established.** The marginal contributions are not individually
significant at either seed on their own, and the pooled estimate sits near
p = 0.08. Only the combination against the plain baseline clears significance
at a single seed. The p-value for majority-shared → majority-personas at seed
6000 (+28) has not been computed yet and is the load-bearing number for the
cost claim.
Debate still shrinks the pool it is given (622 → 587), and 117 items remain
where some agent holds gold and the vote does not take it -- more unharvested
pool than before, not less.

Next: personas at seed 6000, which pairs with the existing shared run there and
gives the second seed.

### What is left

1. **EXAONE 4-round seed 6002.** 6001 is done and replicated; a third seed
   would let the claim be reported as a mean over three rather than two.
   `python scripts/run_phase2.py --config config/phase2_exaone_stance_minimal_en.yaml
   --model exaone --methods majority,debate --split test --n 1001 --data-seed 0
   --run-seed 6001 --n-rounds 4` (share_round0 gives both methods for 12
   generations/item).
2. Qwen `majority k=12`, for table symmetry only.
3. The advocacy line itself: closed.

### Tooling added this session

| script | what, and whether it needs a GPU |
|---|---|
| `scripts/advocacy_oracle.py` | offline. Ranks the saved cases by surface features against gold, audits quoted passages against the article by script and by match strictness |
| `scripts/analyze_selective_runs.py` | offline. Accuracy by trigger reason, `evidence_sufficient` gate, agreement gate, per-item ceiling over saved runs |
| `scripts/simulate_routing.py` | offline. Substitutes a strong run's predictions on triggered items; compares against random and oracle routing at the same coverage |
| `scripts/run_selective_advocacy.py` | GPU. `--trigger`, `--ablation`, `--stage1-from`, `--reuse-commissioned`, `--dry-run` |
| `scripts/run_advocacy_diagnosis_queue.sh` | GPU. Round-0 vs round-1 judge passes plus the paired comparison |
| `scripts/run_article_first_queue.sh` | GPU. The three article-first conditions, sequential |

Six code defects were fixed on the way; section 3 below is the list, and all of
them are done. Two more were found during the session: `--reuse-advocacy`
wrongly enforced the judge's `--config`/`--model` against the advocates', and
`--resume` silently adopted rows produced by a different judge, which
contaminated one comparison before it was caught.

---

## 1. Where the numbers stand

### Baselines (EXAONE-4.0-1.2B)

| method | calls/item | accuracy | note |
|---|---:|---:|---|
| single | 1 | 0.4865 | |
| **majority k=3** | 3 | **0.4865** | strongest baseline |
| majority k=12 | 12 | 0.4815 | compute-matched; −5 vs k=3, p=0.522 |
| debate 2 rounds | 6 | 0.4885 | vs majority +2, p=0.845 |
| debate 4 rounds | 12 | 0.5035 | vs majority **+17, p=0.027**; vs majority k=12 **+22, p=0.014** |
| debate 4R + selective judge | 13 | 0.5085 | vs debate 4R +5, p=0.383 |

### Baselines (Qwen3-8B)

| method | accuracy | note |
|---|---:|---|
| single | 0.5904 | |
| **majority k=3** | **0.5994** | |
| debate 2 rounds | 0.5894 | vs majority **−10**, p=0.387 |
| debate 2R + selective judge | 0.6034 | vs majority **+4, p=0.683** |

Read that last row carefully: the selective judge's headline "+14 over debate"
disappears against the right baseline. Debate loses 10 to majority and the judge
gives them back.

### Stance advocacy + judge (EXAONE advocates, 2 rounds)

One agent per label, `--prompt-style toc`, `--rounds 2`. Advocate cases are
identical across all four rows — only the judge changed, via `--reuse-advocacy`.

| judge | judge prompt | bare verdicts (<60 chars) | accuracy | vs majority | p |
|---|---|---:|---:|---:|---:|
| EXAONE-1.2B | toc, `should be exactly` | 841/1001 | 0.4386 | −48 | 0.023 |
| EXAONE-1.2B | toc, ToC's own wording | 895/1001 | 0.4486 | −38 | 0.047 |
| EXAONE-1.2B | structured JSON | 91/1001 | 0.3916 | −95 | 0.000 |
| **Qwen3-8B** | structured JSON | **0/1001** (median 483 chars) | 0.4156 | **−71** | 0.005 |

Every configuration is significantly **worse** than plain majority voting.

---

## 2. The diagnosis this points to

**A judge that actually reasons does worse.** The Qwen3-8B judge demonstrably
performed the contrastive verification — 0 bare verdicts, median 483 characters,
194 minutes for 1001 calls — and landed at 0.4156, below the EXAONE judge that
emitted a bare label 895 times out of 1001 (0.4486). Judge capacity was not the
bottleneck. Reading the three advocacy cases carefully makes the decision worse,
which means the cases are persuasive noise rather than evidence.

**The system is not adjudicating majority, it is replacing it.** Against
majority's 487 correct and 514 wrong items:

| judge | majority's correct answers kept | majority's errors fixed |
|---|---:|---:|
| EXAONE structured | 195/487 = 40.0% | 197/514 = 38.3% |
| Qwen3-8B structured | 143/487 = 29.4% | 273/514 = 53.1% |

The EXAONE judge is roughly *independent* of majority (40% vs 38%). The Qwen
judge is *anti-correlated* (29% vs 53%) — it does better on the items majority
got wrong than on the ones it got right. Neither behaves like a repair step.

**Forcing 1:1:1 throws away the vote signal.** Free-form majority carries
"two of three independently chose supportive", which for a small model is real
information. Assigning one agent per label makes every item 1:1:1 by
construction, so the judge sees the same prior on every item. The earlier
finding that "88% of residual errors never had gold proposed" was read as a
candidate-generation bottleneck; it may instead be a straightforward statement
that the 1.2B model cannot identify those golds at all. Forcing it to argue for
gold produces a plausible counterfactual, not evidence.

**The Qwen judge has a significant first-position bias.** Winning candidate
position A/B/C = 383/328/290, χ²=13.10, df=2, **p=0.0014**. Candidate order is
already shuffled per item with a stable seed, so this is the judge preferring
whatever is presented first, not an artifact of a fixed layout.

**Rebuttal rounds may be contaminating the cases.** 84.7% of round-1 outputs
mention "Analyst", and round 1 is shorter than round 0 (1187 vs 1489 chars): the
agents argue about each other rather than about the article, and the judge only
sees the last round. An agent can also concede in round 1 while its case is
still labelled with its originally assigned stance.

---

## 3. Code issues to fix (all verified in the branch) — ALL FIXED, see section 0

1. **Structured JSON is not strict.** `run_advocacy_judge` catches a
   `parse_judge_json` failure and immediately salvages a label with
   `parse_stance(raw)`, marking success without a retry. Malformed output is
   accepted silently. Retry first, salvage only on the final attempt. Check
   `judge_diagnostics.label_recovered_from_text` on past runs before trusting
   "retries=0, fallback=0". *(src/advocacy.py, `run_advocacy_judge`)*
2. **Advocacy input-token counts are wrong.** Only
   `contexts[agent_index][-1]["content"]` is counted, so round-1 numbers omit the
   article and the agent's own prior turns. Count the rendered chat template
   instead. Do not trust `cost.advocacy.input_tokens` in any saved summary.
   *(src/advocacy.py, `run_advocacy`)*
3. **`--reuse-advocacy` still loads the advocate model.** `load_model` runs
   before `reused_advocacy` is populated, so a judge-only pass puts an unused
   model on the GPU. It also forces the judge to inherit the advocate config's
   quantization. *(scripts/run_advocacy_judge.py, `main`)*
4. **`--reuse-advocacy` validation is too weak.** Only the round count and the
   presence of cases are checked; prompt style, model, order seed and run seed
   can all differ silently. Validate against the sibling `.config.json`.
5. **The ToC retry prompt drops the evidence.** `TOC_JUDGE_REPAIR_TEMPLATE`
   resends only the invalid output, not the article and the three rationales.
   *(src/prompts/advocacy.py)*

---

## 4. Experiments to run, in order — ALL RUN, see section 0 for the results

All of 1–3 reuse the saved `.items.json`, so they need **no new advocate
generation**. Full advocacy regeneration is 7 calls/item ≈ 10.7 hours.

**1. Round 0 cases vs round 1 cases, same judge** *(needs a small code change:
`--judge-round 0|last`)*
Build the judge input from `analyses_by_round[0]` instead of the last round. If
round 0 beats round 1, the PREDICT-style rebuttal is contaminating the
candidates and should be dropped for this task. Judge-only cost.

**2. Position-bias confirmation**
Re-judge with a different `--order-seed`. If A still wins ~38% of the time, the
Qwen judge's first-position preference is real and must be reported.

**3. Oracle ceiling on the existing cases** *(offline, no GPU)*
For each item, check whether *any* of the three saved cases argues the gold
label convincingly — i.e. what a perfect judge could score on this candidate
set. If the ceiling is near 1.0, the judge is the whole problem; if it is near
0.45, the cases themselves carry no signal and the design is done.

**4. Selective advocacy — the design the evidence actually supports**
Keep the existing 3-agent free-form prediction and its vote signal. Generate
missing-label counterfactuals **only** where majority is unstable or a label
never appeared, then call a strong judge on just those items. This targets the
"gold candidate missing" problem without discarding the ensemble signal that
majority provides, and it is what the 40%/38% and 29%/53% tables argue for.

**5. Compute-matched baseline for whatever survives**
Advocacy at 2 rounds is 7 calls/item, so the honest control is
`majority --n-agents 7 --k 7`. Not yet run.

**6. Qwen debate 4 rounds** *(unrelated thread, still open)*
EXAONE showed debate beating majority only at 4 rounds. Qwen at 2 rounds loses
to majority. `scripts/run_rounds_queue.sh` runs the 4-round extension plus the
compute-matched `majority k=12`. Roughly 18 GPU-hours; status unknown.

---

## 5. File map

| path | what |
|---|---|
| `src/prompts/advocacy.py` | both prompt styles, `PROMPT_STYLES` registry |
| `src/advocacy.py` | label assignment, round loop, judge, verdict parsing, fallback |
| `scripts/run_advocacy_judge.py` | CLI, resume, `--reuse-advocacy`, metrics, qualitative report |
| `src/prompts/stance.py` | `stance_minimal_en`, the profile every baseline uses |
| `src/debate.py`, `src/methods.py` | Du et al. debate and the single/majority/debate methods |
| `src/selective_judge.py`, `src/consensus.py` | article-only judge, instability trigger, metrics |
| `scripts/analyze_rounds.py` | per-round transitions, agreement × accuracy, judge trigger counts |
| `scripts/analyze_advocacy_rounds.py` | offline. What the rebuttal round does to the cases: the ballot, homogenisation, grounding, truncation |
| `scripts/compare_methods.py` | paired McNemar across files, reads phase2 and judge items |
| `docs/stance_advocacy.md` | design rationale and the failures already diagnosed |
| `docs/selective_judge.md` | the earlier selective-judge pipeline |

Tests: `python -m unittest tests.test_advocacy tests.test_advocacy_rounds tests.test_selective_judge tests.test_consensus tests.test_selective_consensus`

---

## 6. Commands

Judge-only re-run on saved cases:

```bash
HF_HUB_OFFLINE=1 python scripts/run_advocacy_judge.py \
  --config config/phase2_qwen8_stance_minimal_en.yaml --model qwen \
  --split test --n 1001 --data-seed 0 --run-seed 6000 \
  --prompt-style structured \
  --reuse-advocacy results/advocacy/advocacy_exaone_test_n1001_stance_minimal_en_d0_s6000_toc_r2_ord8001_judge-EXAONE-4_0-1_2B_jt0.items.json \
  --baseline-result results/phase2/stance_exaone_test_n1001_stance_minimal_en_d0_s6000_a3_r2.json \
  --baseline-method majority --output-dir results/advocacy
```

Paired comparison across any saved methods:

```bash
python scripts/compare_methods.py \
  results/phase2/<run>.json:majority \
  results/phase2/<run>.json:debate \
  results/advocacy/<run>.items.json:pred=advocacy
```

Round dynamics of a debate run:

```bash
python scripts/analyze_rounds.py results/phase2/<r4>.json --compare results/phase2/<r2>.json
```

---

## 7. Papers this is built on

| paper | first author | venue | citations (OpenAlex, 2026-08) | used for |
|---|---|---|---|---|
| Encouraging Divergent Thinking in LLMs through Multi-Agent Debate (MAD) | Tian Liang | EMNLP 2024 | 231 (1,291 on Semantic Scholar, preprint merged) | debater/judge separation; judge always decides, no consensus rule |
| Tree-of-Counterfactual Prompting for Zero-Shot Stance Detection (ToC) | Maxwell A. Weinzierl | ACL 2024 | 7 | the backbone: one rationale per stance value, contrastive verification; round-0 and judge prompts are near-verbatim |
| PREDICT: Multi-Agent-based Debate Simulation for Generalized Hate Speech Detection | Someen Park | EMNLP 2024 | 7 | the fixed two-round structure and the rebuttal prompt |
| M-MAD: Multidimensional Multi-Agent Debate for MT Evaluation | Zhaopeng Feng | ACL 2025 | 5 | decompose-then-synthesize; its ablation (decoupling ≫ debate) is why rounds are a parameter |

Base reproduction target: Du et al., *Improving Factuality and Reasoning in
Language Models through Multiagent Debate* (ICML 2024).

Prompt provenance in detail is in `docs/stance_advocacy.md`. Departures from ToC
that live in code rather than prompts: per-item shuffled label assignment, and
per-item shuffled + anonymized candidate order for the judge (ToC does neither).

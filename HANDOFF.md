# Handoff: stance advocacy + judge on K-News-Stance

Everything needed to pick this up cold. Repo `wlsdn66597/mad-news-stance`,
branch **`codex/selective-judge-consensus`**. Results and data live on the GPU
server (`~/mad/mad-news-stance`, RTX 4090), not in git.

Shared setup for every number below: K-News-Stance `test` split, n=1001,
`--data-seed 0 --run-seed 6000`, prompt profile `stance_minimal_en`,
temperature 1.0, `max_new_tokens` 1024, judge temperature 0. Runs are paired
item-for-item, so all comparisons are exact McNemar on the same items.

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

## 3. Code issues to fix (all verified in the branch)

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

## 4. Experiments to run, in order

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
| `scripts/compare_methods.py` | paired McNemar across files, reads phase2 and judge items |
| `docs/stance_advocacy.md` | design rationale and the failures already diagnosed |
| `docs/selective_judge.md` | the earlier selective-judge pipeline |

Tests: `python -m unittest tests.test_advocacy tests.test_selective_judge tests.test_consensus tests.test_selective_consensus`

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

# Selective consensus round before the article-only judge

This adds one optional step in front of the existing selective judge: the
unstable items are re-examined once by the same three debate agents, and a
**unanimous** revised label is accepted without calling the judge.

> This is **not** a reproduction of a published consensus protocol. It is
> `selective independent reconsideration with unanimity-based acceptance`
> applied to this repository's MAD pipeline. The final adjudicator is the
> unchanged article-only judge documented in [selective_judge.md](selective_judge.md).

## Pipeline

```
existing instability trigger (src/consensus.judge_trigger)
  -> consensus round: each original agent answers once more in its own saved context
  -> three identical revised labels  -> accept that label, skip the judge
  -> 2-1 / 1-1-1 / parse failure     -> existing article-only judge
  -> judge returns nothing           -> existing round0_fallback
not triggered                        -> existing final majority, unchanged
```

Gold labels never reach the trigger, the consensus prompt or the judge; they are
joined only afterwards for evaluation. The number of triggered items is computed
from the input result, never hardcoded.

## Conditions

| Condition | `--consensus-prompt-mode` | Consensus | On failure |
|---|---|---|---|
| `direct_article_judge` | `none` | none | every triggered item goes to the article-only judge |
| `extra_round_then_judge` | `plain_extra_round` | saved debate prompt, unchanged | only non-unanimous items go to the judge |
| `reconsideration_then_judge` | `independent_reconsideration` | same prompt + independence instruction | only non-unanimous items go to the judge |

Everything else is held equal across conditions: input result, trigger,
aggregation, model, agent count, generation config, peer-order seed, label
parser and judge settings.

## The two consensus prompts

Both modes reuse `debate_trace.debate_template` from the saved run verbatim, with
`{others}` filled by the anonymized peer responses in the repository's existing
format. With `stance_minimal_en` the delivered prompt is:

```text
Use the other agents' responses as additional information and reconsider your previous judgment.
Briefly explain your reasoning, and answer on the final line in exactly this format:

Final stance: <supportive|oppositional|neutral>

The other agents' judgments are as follows:

Agent 1:
<peer response>

Agent 2:
<peer response>
```

`independent_reconsideration` appends exactly one paragraph and changes nothing
else:

```text
Re-evaluate the stance independently against the original article,
without following the majority automatically. Keep or revise your
answer based on the article evidence.
```

No agent is told to agree, to persuade anyone, or which label is currently in
the majority; no vote counts, no confidence field, no JSON schema, no
`[AGREE]/[DISAGREE]` marker. Unanimity is decided in code
(`src.consensus_round.consensus_decision`) after the calls.

Each agent receives its own saved `agent_contexts` with the consensus request
appended as one more user turn, so the article and its own previous answers stay
in place. If a saved result has no `agent_contexts`, the context is rebuilt from
the saved question plus that agent's own final answer, and the per-item field
`consensus_agent_context_source` records which path was used. Peer responses are
shuffled per item and per agent with `stable_seed(order_seed, item_id, ...)`;
the realized order is stored in `peer_orders`.

## Running

```bash
python scripts/run_selective_consensus.py \
  --input-result results/phase2/<mad_result>.json \
  --data-path data/k-news-stance_nosegment.json \
  --consensus-prompt-mode none \
  --judge-model Qwen/Qwen3-8B \
  --output-dir results/selective_consensus
```

```bash
python scripts/run_selective_consensus.py \
  --input-result results/phase2/<mad_result>.json \
  --data-path data/k-news-stance_nosegment.json \
  --consensus-prompt-mode plain_extra_round \
  --consensus-model Qwen/Qwen3-8B \
  --judge-model Qwen/Qwen3-8B \
  --judge-result results/selective_judge/<article_only_run>.items.json \
  --output-dir results/selective_consensus
```

```bash
python scripts/run_selective_consensus.py \
  --input-result results/phase2/<mad_result>.json \
  --data-path data/k-news-stance_nosegment.json \
  --consensus-prompt-mode independent_reconsideration \
  --consensus-model Qwen/Qwen3-8B \
  --judge-model Qwen/Qwen3-8B \
  --judge-result results/selective_judge/<article_only_run>.items.json \
  --output-dir results/selective_consensus
```

Consensus generation defaults to the debater settings of the source run
(`_meta.model_id`, `_meta.temperature`, `_meta.max_new_tokens`, `_meta.run_seed`);
`--consensus-model`, `--consensus-temperature`, `--consensus-max-new-tokens`,
`--consensus-seed`, `--consensus-order-seed` override them and every resolved
value is written to the config and summary files. The run is item-resumable:
re-running the same command reuses `<prefix>.items.json` and only processes
missing items (`--no-resume` to force a fresh pass). `--limit N` runs the first
N items for a smoke check.

## Repeats

The article-only judge is deterministic: `temperature=0` and, with
`input_mode=article_only`, `--judge-order-seed` only shuffles candidate analyses
that are never sent. Repeating it with a different judge seed reproduces the
same predictions exactly, so repeats belong to the consensus round instead,
which inherits the debater `temperature=1.0`:

```bash
bash scripts/run_selective_consensus_repeats.sh <mad_result.json> <article_only_judge.items.json> 6000 6001 6002
```

Each debate rerun is its own instability subset, so pair every
`results/phase2/<debate run>.json` with the judge `.items.json` produced from
that same run, and run the script once per pair. `JUDGE_ORDER_SEED=7002 bash …`
overrides the judge seed when the cached run used a different one.

`scripts/run_selective_consensus_queue.sh` holds that pairing for the existing
`s6000 / s6001 / s6002` debate repeats and runs all three conditions for each,
with the consensus seed set to that repeat's own debate run seed:

```bash
bash scripts/run_selective_consensus_queue.sh            # all repeats
bash scripts/run_selective_consensus_queue.sh 6001 6002  # only these
```

It verifies every input file exists before starting, so a wrong path fails
immediately instead of after hours of generation.

## Reusing an existing article-only judge run

`--judge-result <run>.items.json` reuses judge predictions by item id instead of
calling the model again. A cached row is used only when the judge model, input
mode, temperature, max new tokens and retry count all match this run, and when
the companion `<run>.config.json` agrees on those settings and on the source
`--input-result` file. New runs additionally store `judge_payload_sha256`, the
hash of the exact article-only judge input; when a cached row carries that
field, it must match or the item falls back to a real model call
(`judge_cache_rejected: "payload_mismatch"`). A cache without a companion config
is rejected unless `--no-judge-cache-require-config` is passed. Rows whose
config disagrees are dropped and counted in `judge_cache.mismatched_rows`;
a disagreeing run-level config raises instead of silently mixing conditions.

The same file is also used as the `direct_article_judge` reference: the summary
reports `direct_article_judge_comparison` with accuracy/macro-F1 deltas and the
transition counts against that run.

## Outputs

`<output-dir>/<prefix>.items.json`, `.summary.json`, `.config.json`, `.csv`, `.md`.
Existing result files are never modified.

Per item: `triggered`, `trigger_reason`, `consensus_prompt_mode`,
`previous_agent_labels`, `revised_agent_labels`, `revised_agent_outputs`,
`peer_orders`, `consensus_reached`, `consensus_label`, `consensus_reason`
(`unanimous` / `split_2_1` / `split_1_1_1` / `parse_failure`), `judge_required`,
`judge_source` (`cache` / `model_call` / `none`), `judge_prediction`,
`final_prediction`, `final_source` (`baseline` / `consensus` / `judge` /
`fallback`), plus per-stage latency and token usage.

Summary: overall and triggered-subset accuracy, macro-F1, per-class P/R/F1,
confusion matrices, wrong_to_correct / correct_to_wrong / net improvement,
exact McNemar, bootstrap CI, boundary switches, and a consensus-stage block with
trigger count, consensus reached count and ratio, consensus accuracy, wrong
unanimous consensus count, preserved/fixed/broken counts, parse failures, items
sent to the judge, and consensus vs judge call counts, tokens and latency.

## Verification

```bash
python -m unittest tests.test_selective_consensus tests.test_selective_judge tests.test_consensus -v
```

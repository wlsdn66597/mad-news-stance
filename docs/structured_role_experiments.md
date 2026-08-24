# Four-round structured-role experiments

Both scripts use the existing K-News-Stance loader, label parser, model loader,
sampling temperature, and deterministic per-item seeding. A round count of four
means Round 0 plus three response-exchange rounds, matching the reported MoRE
debate condition.

## Segment agents

`scripts/run_segment_debate.py` assigns one agent to each of Headline, Lead,
Conclusion, and Quotations. Each agent sees only its assigned raw segment in
Round 0. In later rounds it retains the same role and original segment while
receiving the other agents' analyses. An always-called article-level aggregator
then verifies their final analyses against the full article and returns the
label. It does not count agent labels.

By default the aggregator uses the same sampling temperature as the debate
agents, so the final stance-generating call does not introduce a decoding
change. It can be overridden explicitly with `--aggregator-temperature`.

With four rounds, the method uses 17 generations per item: 16 segment-agent
generations and one aggregation.

```bash
python scripts/run_segment_debate.py \
  --config config/phase2_exaone_stance_minimal_en.yaml \
  --model exaone --split test --n 1001 \
  --data-seed 0 --run-seed 6000 --n-rounds 4
```

## Planner-selected roles

`scripts/run_planner_debate.py` first asks a deterministic planner to choose
exactly three distinct roles from a closed journalism-based library. Only the
three role IDs determine the agent prompts; the planner's observed features are
saved for analysis but are not handed to the debate agents. The three agents
then run the existing `reasoned_exchange_full` debate and use its existing
majority vote. Invalid planner output falls back to Foregrounding, Sourcing,
and Wording and records the parsing error.

With four rounds, the method uses 13 generations per item: one planning call
and 12 debate-agent generations.

```bash
python scripts/run_planner_debate.py \
  --config config/phase2_exaone_stance_minimal_en.yaml \
  --model exaone --split test --n 1001 \
  --data-seed 0 --run-seed 6000 --n-rounds 4 \
  --role-pool journalism --debate-protocol reasoned_exchange_full
```

For an ablation using only the roles already present in the repository, replace
`--role-pool journalism` with `--role-pool legacy`.

To preserve the two evidence channels that the free planner rarely selected,
fix Sourcing and Wording and let the planner choose only the third role:

```bash
python scripts/run_planner_debate.py \
  --config config/phase2_exaone_stance_minimal_en.yaml \
  --model exaone --split test --n 1001 \
  --data-seed 0 --run-seed 6000 --n-rounds 4 \
  --role-pool journalism --planner-mode sw_plus_one \
  --debate-protocol reasoned_exchange_full
```

## Comparison fields

Every output contains the full configuration in `_meta`, per-item gold and
prediction fields, the complete debate trace, and generation counts. Planner
outputs additionally store selected roles, observed features, fallback status,
and stance-label leakage. Segment outputs store extracted segments, each
agent's final label, and the aggregator prompt and response.

"""Project the context budget of a debate run before spending GPU time.

`src.llm.validate_context_budget` refuses to generate when
``input_tokens + max_new_tokens`` exceeds the context window: inputs are never
truncated, so a single long article stops the run. These helpers estimate the
peak requirement per item from token counts alone, with no model weights.

Debate round r>0 sends, in one context: the original question, this agent's own
previous answer, and the debate template filled with the other agents' previous
answers. Every previous answer is bounded by ``max_new_tokens``, so the peak is
an upper bound, not a measurement.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def round_requirements(
    question_tokens: int,
    max_new_tokens: int,
    n_agents: int = 3,
    template_tokens: int = 0,
    round_index: int = 1,
) -> dict[str, int]:
    """Tokens needed by one agent's context at ``round_index`` (0-based).

    The context accumulates: at round r it holds the question, this agent's r
    previous answers, and r debate prompts, each carrying the template and the
    other agents' answers from that round. Every answer is bounded by
    ``max_new_tokens``, so this is an upper bound and it grows linearly with the
    round index.
    """
    if n_agents < 1:
        raise ValueError("n_agents must be at least 1")
    if round_index < 0:
        raise ValueError("round_index must be at least 0")
    own_answers = round_index * max_new_tokens
    debate_prompts = round_index * (template_tokens + (n_agents - 1) * max_new_tokens)
    input_tokens = question_tokens + own_answers + debate_prompts
    return {
        "round_index": round_index,
        "input": input_tokens,
        "required": input_tokens + max_new_tokens,
    }


def peak_requirement(
    question_tokens: int,
    max_new_tokens: int,
    n_agents: int = 3,
    template_tokens: int = 0,
    n_rounds: int = 2,
) -> int:
    """The last round is the widest, so it decides whether the run fits."""
    return round_requirements(
        question_tokens, max_new_tokens, n_agents, template_tokens, max(0, n_rounds - 1)
    )["required"]


def percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def project_budgets(
    question_tokens_by_id: Mapping[Any, int],
    max_new_tokens: int,
    context_window: int | None,
    n_agents: int = 3,
    template_tokens: int = 0,
    n_rounds: int = 2,
) -> dict[str, Any]:
    """Per-item projection plus the items that would stop the run."""
    rows = []
    for item_id, tokens in question_tokens_by_id.items():
        rows.append(
            {
                "item_id": item_id,
                "question_tokens": tokens,
                "peak_required": peak_requirement(
                    tokens, max_new_tokens, n_agents, template_tokens, n_rounds
                ),
            }
        )
    peaks = [row["peak_required"] for row in rows]
    questions = [row["question_tokens"] for row in rows]
    over = (
        [row for row in rows if row["peak_required"] > context_window]
        if context_window
        else []
    )
    headroom = (context_window - max(peaks)) if (context_window and peaks) else None
    return {
        "items": len(rows),
        "context_window": context_window,
        "max_new_tokens": max_new_tokens,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "template_tokens": template_tokens,
        "question_tokens": {
            "min": min(questions) if questions else 0,
            "p50": percentile(questions, 0.50),
            "p95": percentile(questions, 0.95),
            "max": max(questions) if questions else 0,
        },
        "peak_required": {
            "p50": percentile(peaks, 0.50),
            "p95": percentile(peaks, 0.95),
            "max": max(peaks) if peaks else 0,
        },
        "headroom_at_worst_item": headroom,
        "fits": bool(context_window) and not over,
        "over_budget_items": sorted(
            (
                {"item_id": row["item_id"], "question_tokens": row["question_tokens"],
                 "peak_required": row["peak_required"]}
                for row in over
            ),
            key=lambda row: -row["peak_required"],
        ),
        "max_new_tokens_that_would_fit": (
            max_new_tokens_that_fits(
                max(questions), context_window, n_agents, template_tokens, n_rounds
            )
            if context_window and questions
            else None
        ),
    }


def max_new_tokens_that_fits(
    question_tokens: int,
    context_window: int,
    n_agents: int = 3,
    template_tokens: int = 0,
    n_rounds: int = 2,
) -> int:
    """Largest output budget whose last-round peak still fits, 0 when hopeless."""
    last = max(0, n_rounds - 1)
    # required = question + last*template + max_new_tokens * (last*n_agents + 1)
    room = context_window - question_tokens - last * template_tokens
    return max(0, room // (last * n_agents + 1))

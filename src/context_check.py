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
) -> dict[str, int]:
    """Tokens needed by the widest context of each debate round."""
    if n_agents < 1:
        raise ValueError("n_agents must be at least 1")
    round0_input = question_tokens
    # question + own previous answer + template + (n_agents - 1) peer answers
    round1_input = (
        question_tokens + max_new_tokens + template_tokens + (n_agents - 1) * max_new_tokens
    )
    return {
        "round0_input": round0_input,
        "round0_required": round0_input + max_new_tokens,
        "round1_input": round1_input,
        "round1_required": round1_input + max_new_tokens,
    }


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
        requirements = round_requirements(tokens, max_new_tokens, n_agents, template_tokens)
        peak = (
            requirements["round1_required"] if n_rounds > 1 else requirements["round0_required"]
        )
        rows.append({"item_id": item_id, "question_tokens": tokens, "peak_required": peak,
                     **requirements})
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
            max_new_tokens_that_fits(max(questions), context_window, n_agents, template_tokens)
            if context_window and questions
            else None
        ),
    }


def max_new_tokens_that_fits(
    question_tokens: int, context_window: int, n_agents: int = 3, template_tokens: int = 0
) -> int:
    """Largest output budget whose round-1 peak still fits, 0 when hopeless."""
    # round1_required = question + template + (n_agents + 1) * max_new_tokens
    room = context_window - question_tokens - template_tokens
    return max(0, room // (n_agents + 1))

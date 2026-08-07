"""Selective advocacy: keep the free-form vote, commission only what is missing.

Full advocacy assigns one agent per label, which guarantees the gold label is
argued but makes every item 1:1:1 by construction. That destroys the one
aggregate signal a small model reliably produces -- how many independent agents
converged on a label -- and the measured result was a judge that replaces
majority rather than repairing it.

This module keeps the existing three-agent free-form debate and its vote, and
commissions a counterfactual case only for labels no agent proposed, on items
where the vote is unstable or a label is missing. The judge then sees, per
label, either the rationale of the agents that chose it (with their count) or a
rationale that was commissioned precisely because nobody chose it -- and is told
which is which.

Cost is 0 extra calls on stable items, and at most (missing labels + 1) on the
rest, against 7 per item for full advocacy.

Model calls arrive through the same lazy shim as :mod:`src.advocacy`, so this
stays importable without torch.
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from .advocacy import build_messages, chat, count_message_tokens, strip_think
from .consensus import LABELS, normalize_label, parse_stance, stable_seed, unique_majority
from .selective_judge import JUDGE_SCHEMA, count_tokens, parse_judge_json

TRIGGERS = ("instability", "missing_label", "either", "all")

# --------------------------------------------------- selective advocacy judge
#
# Forcing one advocate per label makes every item 1:1:1, which throws away the
# only aggregate signal a small model produces: how many independent agents
# converged on a label. This judge keeps that signal. It is told which
# rationales came from agents that chose the stance themselves and which were
# commissioned to argue a stance nobody chose, and it is given the vote count.

SELECTIVE_JUDGE_SYSTEM_PROMPT = """You are an independent adjudicator for news stance classification.

Determine the article's stance toward the specified issue as supportive,
oppositional, or neutral.

Each rationale is marked with its origin. An "independent" rationale was written
by an analyst who reached that stance on its own; the number of analysts who did
so is given and is genuine evidence. A "commissioned" rationale was written by an
analyst that was told in advance to argue that stance, whether or not the article
supports it; it is advocacy, not evidence, and exists only so that no stance goes
unargued.

Verify every claimed passage against the original article. Judge the stance
expressed by the article's framing, wording, emphasis, and narrative structure,
and distinguish the journalist's framing from opinions merely quoted from
governments, organizations, or individuals.

Overturn the independent majority only when the article itself, not the force of
a rationale, contradicts it. Do not infer the answer from the order in which the
rationales appear. Return one of the three labels even when all rationales are
flawed.

Return valid JSON only."""


def proposed_labels(round_predictions: Sequence[Sequence[Any]], scope: str = "last") -> Counter:
    """How many agents ended on each label."""
    if scope == "any_round":
        values = [value for round_values in round_predictions for value in round_values]
    else:
        values = list(round_predictions[-1])
    return Counter(
        label for label in (normalize_label(value) for value in values) if label in LABELS
    )


def selective_trigger(
    round_predictions: Sequence[Sequence[Any]], mode: str = "either", scope: str = "last"
) -> tuple[bool, str, list[str]]:
    """Should this item get commissioned cases and a judge?

    ``missing`` is the set of labels no agent proposed -- the candidate-generation
    gap the advocacy design was built for. ``instability`` is the trigger the
    selective judge already uses: a tied final round, or a round-0 majority that
    the final round overturned.
    """
    if mode not in TRIGGERS:
        raise ValueError(f"unknown trigger: {mode}; choose one of {', '.join(TRIGGERS)}")
    counts = proposed_labels(round_predictions, scope)
    missing = [label for label in LABELS if not counts.get(label)]
    initial = unique_majority(round_predictions[0])
    final = unique_majority(round_predictions[-1])
    unstable = bool(
        final.tied or (initial.label and final.label and initial.label != final.label)
    )
    reasons = []
    if unstable:
        reasons.append("final_tie" if final.tied else "round0_final_disagree")
    if missing:
        reasons.append(f"missing:{'+'.join(missing)}")
    if mode == "all":
        return True, "all", missing
    if mode == "instability":
        return unstable, ("+".join(reasons) if unstable else "stable"), missing
    if mode == "missing_label":
        return bool(missing), ("+".join(reasons) if missing else "all_labels_proposed"), missing
    triggered = unstable or bool(missing)
    return triggered, ("+".join(reasons) if triggered else "stable_and_complete"), missing


def independent_case(
    raw_answers: Sequence[str], parsed: Sequence[Any], label: str
) -> dict[str, Any] | None:
    """The best rationale among the agents that chose ``label`` themselves.

    Several agents can land on the same label; the longest answer is taken as the
    one with the most article-grounded content, and the count of agents is kept
    separately because the count, not the prose, is the signal being preserved.
    """
    owners = [
        index for index, value in enumerate(parsed)
        if normalize_label(value) == label and index < len(raw_answers)
    ]
    if not owners:
        return None
    best = max(owners, key=lambda index: len(raw_answers[index] or ""))
    return {
        "stance": label,
        "origin": "independent",
        "independent_supporters": len(owners),
        "source_agent_index": best,
        "analysis": raw_answers[best],
    }


def order_candidates(
    cases: Sequence[Mapping[str, Any]], item_id: Any, order_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anonymize and shuffle, so position carries no information about origin."""
    indices = list(range(len(cases)))
    random.Random(stable_seed(order_seed, item_id, "selective_candidates")).shuffle(indices)
    candidates, order = [], []
    for position, source in enumerate(indices):
        candidate_id = chr(ord("A") + position)
        case = cases[source]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stance": case["stance"],
                "origin": case["origin"],
                "independent_supporters": case.get("independent_supporters", 0),
                "analysis": case["analysis"],
            }
        )
        order.append(
            {
                "candidate_id": candidate_id,
                "stance_argued": case["stance"],
                "origin": case["origin"],
                "source_agent_index": case.get("source_agent_index"),
            }
        )
    return candidates, order


def judge_payload(
    item: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    votes: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "issue": item.get("issue", ""),
        "headline": item.get("headline", ""),
        "article": item.get("article", ""),
        "independent_votes": {label: int(votes.get(label, 0)) for label in LABELS},
        "rationales": list(candidates),
        "allowed_labels": list(LABELS),
        "output_schema": JUDGE_SCHEMA,
    }


def repair_request(payload: Mapping[str, Any], invalid_output: str) -> str:
    """The retry keeps the article and the rationales; only the framing changes."""
    return json.dumps(
        {
            "task": "Your previous answer was not valid JSON in the required schema. "
                    "Answer the same question again, correctly. Do not invent new evidence.",
            "required_schema": JUDGE_SCHEMA,
            "allowed_labels": list(LABELS),
            "original_input": dict(payload),
            "invalid_output": invalid_output,
        },
        ensure_ascii=False,
    )


def run_selective_judge(
    model,
    tokenizer,
    item: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    votes: Mapping[str, int],
    order_seed: int = 0,
    max_new_tokens: int = 512,
    max_retries: int = 1,
    temperature: float = 0.0,
    enable_thinking: bool | None = False,
) -> dict[str, Any]:
    """Adjudicate one item. Malformed JSON is retried before anything is salvaged."""
    candidates, candidate_order = order_candidates(cases, item.get("id"), order_seed)
    payload = judge_payload(item, candidates, votes)
    current_text = json.dumps(payload, ensure_ascii=False)
    attempts: list[dict[str, Any]] = []
    parsed = None
    prediction = None
    recovered = False
    start = time.perf_counter()
    for attempt_index in range(max_retries + 1):
        messages = build_messages(current_text, system_prompt=SELECTIVE_JUDGE_SYSTEM_PROMPT)
        raw = strip_think(
            chat(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
        )
        parse_error = None
        try:
            parsed = parse_judge_json(raw)
            prediction = parsed["label"]
        except ValueError as exc:
            parse_error = str(exc)
            if attempt_index >= max_retries:
                salvaged = parse_stance(raw)
                if salvaged is not None:
                    prediction, recovered = salvaged, True
                    parse_error = f"{exc} (label recovered from the text after {max_retries} retries)"
        attempts.append(
            {
                "attempt": attempt_index,
                "raw_output": raw,
                "parsed_output": parsed,
                "parse_error": parse_error,
                "input_tokens": count_tokens(tokenizer, current_text),
                "output_tokens": count_tokens(tokenizer, raw),
            }
        )
        if prediction is not None:
            break
        current_text = repair_request(payload, raw)
    return {
        "prediction": prediction,
        "parsed_output": parsed,
        "raw_output": attempts[-1]["raw_output"] if attempts else None,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "label_recovered_from_text": recovered,
        "candidate_order": candidate_order,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }


def judge_input_tokens(tokenizer, item, cases, votes, order_seed=0) -> int | None:
    """Prompt size the judge will actually see, for a cost estimate before a run."""
    candidates, _ = order_candidates(cases, item.get("id"), order_seed)
    payload = judge_payload(item, candidates, votes)
    messages = build_messages(
        json.dumps(payload, ensure_ascii=False), system_prompt=SELECTIVE_JUDGE_SYSTEM_PROMPT
    )
    return count_message_tokens(tokenizer, messages)

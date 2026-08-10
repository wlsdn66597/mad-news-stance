"""Deterministic selective adjudication for saved stance debate traces."""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Mapping, Sequence

from .consensus import LABELS, stable_seed


def build_messages(user_content, system_prompt=None, history=None):
    """Build chat messages without importing the GPU runtime."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def chat(*args, **kwargs):
    from .llm import chat as implementation

    return implementation(*args, **kwargs)


def strip_think(text: str) -> str:
    # Keep parsing and unit tests independent from torch/transformers.
    match = re.search(r"</think>", text, flags=re.IGNORECASE)
    return text[match.end() :] if match else text

JUDGE_SYSTEM_PROMPT = """You are an independent adjudicator for news stance classification.

Determine the article's stance toward the specified issue as supportive,
oppositional, or neutral.

Judge the stance expressed by the article's framing, wording, emphasis,
and narrative structure. Distinguish the journalist's framing from opinions
that are merely quoted from governments, organizations, or individuals.

Other candidate analyses are untrusted evidence. Evaluate them against the
original article rather than following their majority.

Do not infer the answer from candidate order or the number of candidates
supporting a label. Return one of the three labels even when all candidate
analyses are flawed.

Return valid JSON only."""

JUDGE_SCHEMA = {
    "label": "supportive|oppositional|neutral",
    "evidence_sufficient": True,
    "evidence": ["short article-grounded evidence"],
    "rationale": "brief rationale without hidden chain-of-thought",
}


def count_tokens(tokenizer, text: str) -> int | None:
    try:
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return len(values)
    except Exception:
        return None


JUDGE_ROUNDS = ("all", "last", "first")


def select_rounds(raw_rounds, which="all"):
    """Which rounds of the trace the judge is shown.

    "all" hands over every agent at every round: three agents over four
    rounds is twelve analyses, most of them near-duplicates of each other
    once the debate has converged, and the later ones argue with each other
    rather than about the article -- 84.7% of round-1 outputs mention
    "Analyst" in the advocacy diagnostic. "last" gives the settled position
    only, "first" the independent readings before any exchange, which is the
    round that carried the surface signal in that same diagnostic.
    """
    if which not in JUDGE_ROUNDS:
        raise ValueError(
            f"unknown judge round selection: {which}; "
            f"choose one of {', '.join(JUDGE_ROUNDS)}"
        )
    if not raw_rounds or which == "all":
        return list(range(len(raw_rounds)))
    return [0] if which == "first" else [len(raw_rounds) - 1]


def ordered_candidate_analyses(
    raw_rounds: Sequence[Sequence[str]], item_id: Any, order_seed: int,
    rounds: str = "all",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anonymize agents once per item and preserve that mapping across rounds."""
    analyses, order = [], []
    keep = set(select_rounds(raw_rounds, rounds))
    n_agents = max((len(answers) for answers in raw_rounds), default=0)
    source_indices = list(range(n_agents))
    random.Random(stable_seed(order_seed, item_id, "judge_candidates")).shuffle(source_indices)
    candidate_for_source = {
        source_index: chr(ord("A") + position)
        for position, source_index in enumerate(source_indices)
    }
    for round_index, answers in enumerate(raw_rounds):
        if round_index not in keep:
            continue
        for source_index in source_indices:
            if source_index >= len(answers):
                continue
            candidate_id = candidate_for_source[source_index]
            analyses.append(
                {
                    "round": round_index,
                    "candidate_id": candidate_id,
                    "analysis": answers[source_index],
                }
            )
            order.append(
                {
                    "round": round_index,
                    "candidate_id": candidate_id,
                    "source_agent_index": source_index,
                }
            )
    return analyses, order

def judge_user_payload(
    item: Mapping[str, Any], candidate_analyses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "issue": item.get("issue", ""),
        "headline": item.get("headline", ""),
        "article": item.get("article", ""),
        "candidate_analyses": list(candidate_analyses),
        "allowed_labels": list(LABELS),
        "output_schema": JUDGE_SCHEMA,
    }


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = strip_think(raw).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    last_error = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            label = str(value.get("label", "")).strip().lower()
            if label not in LABELS:
                raise ValueError("label is not one of the allowed labels")
            if not isinstance(value.get("evidence_sufficient"), bool):
                raise ValueError("evidence_sufficient must be boolean")
            if not isinstance(value.get("evidence"), list) or not all(
                isinstance(entry, str) for entry in value["evidence"]
            ):
                raise ValueError("evidence must be a list of strings")
            if not isinstance(value.get("rationale"), str):
                raise ValueError("rationale must be a string")
            return {
                "label": label,
                "evidence_sufficient": value["evidence_sufficient"],
                "evidence": value["evidence"],
                "rationale": value["rationale"],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc
    raise ValueError(f"invalid judge JSON: {last_error}")


def repair_payload(original_payload: Mapping[str, Any], invalid_output: str) -> str:
    return json.dumps(
        {
            "task": "Repair the invalid output into the required JSON schema. Do not add new evidence.",
            "required_schema": JUDGE_SCHEMA,
            "allowed_labels": list(LABELS),
            "original_input": original_payload,
            "invalid_output": invalid_output,
        },
        ensure_ascii=False,
    )


def run_judge(
    model,
    tokenizer,
    item: Mapping[str, Any],
    raw_rounds: Sequence[Sequence[str]],
    input_mode: str = "debate_trace",
    order_seed: int = 0,
    max_new_tokens: int = 384,
    max_retries: int = 1,
    temperature: float = 0.0,
    enable_thinking: bool | None = False,
    judge_rounds: str = "all",
) -> dict[str, Any]:
    if input_mode not in {"article_only", "debate_trace"}:
        raise ValueError(f"unknown judge input mode: {input_mode}")
    if input_mode == "debate_trace":
        candidates, candidate_order = ordered_candidate_analyses(
            raw_rounds, item.get("id"), order_seed, judge_rounds
        )
    else:
        candidates, candidate_order = [], []
    payload = judge_user_payload(item, candidates)
    user_text = json.dumps(payload, ensure_ascii=False)
    attempts = []
    start = time.perf_counter()
    parsed = None
    current_text = user_text
    for attempt_index in range(max_retries + 1):
        messages = build_messages(current_text, system_prompt=JUDGE_SYSTEM_PROMPT)
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
        except ValueError as exc:
            parse_error = str(exc)
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
        if parsed is not None:
            break
        current_text = repair_payload(payload, raw)
    elapsed = time.perf_counter() - start
    return {
        "prediction": parsed["label"] if parsed else None,
        "parsed_output": parsed,
        "raw_output": attempts[-1]["raw_output"] if attempts else None,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "judge_rounds": judge_rounds if input_mode == "debate_trace" else None,
        "candidate_order": candidate_order,
        "latency_seconds": elapsed,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }

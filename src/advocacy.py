"""Stance-advocacy debate: one agent per label, then a contrastive judge.

Every label is argued for exactly once, so unlike free-form sampling the gold
label always appears among the candidates. That directly targets the failure
this repository measured on saved runs: most residual errors are items where no
agent ever proposed the gold label, which no aggregation rule can repair.

Model calls arrive through the same lazy shim used by :mod:`src.selective_judge`,
so prompt construction, parsing and the fallback rule stay importable without
torch.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .consensus import LABELS, parse_stance, stable_seed
from .prompts.advocacy import (
    ADVOCATE_SYSTEM_PROMPT,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    REBUTTAL_TEMPLATE,
    STANCE_LABELS,
    SUPPORT_LEVELS,
    advocate_question,
    format_peers,
)
from .selective_judge import count_tokens, parse_judge_json, repair_payload

SUPPORT_RANK = {level: rank for rank, level in enumerate(SUPPORT_LEVELS)}


def build_messages(user_content, system_prompt=None, history=None):
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
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_support_level(text: str) -> str | None:
    """The self-reported strength of the assigned stance, if stated."""
    matches = re.findall(
        r"support\s+for\s+the\s+assigned\s+stance\s*[:\-]?\s*<?\s*(weak|moderate|strong)",
        text,
        flags=re.IGNORECASE,
    )
    return matches[-1].lower() if matches else None


def assigned_stances(item_id: Any, order_seed: int) -> list[str]:
    """Which agent index argues which label. Shuffled so the agent index carries
    no fixed meaning across items."""
    stances = list(STANCE_LABELS)
    random.Random(stable_seed(order_seed, item_id, "advocate_assignment")).shuffle(stances)
    return stances


def peer_order(
    cases: Sequence[Mapping[str, Any]], agent_index: int, item_id: Any, order_seed: int
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    others = [
        (index, case) for index, case in enumerate(cases) if index != agent_index
    ]
    random.Random(
        stable_seed(order_seed, item_id, f"advocate_peers_{agent_index}")
    ).shuffle(others)
    peers = [(case["stance"], case["analysis"]) for _, case in others]
    order = [
        {"position": position, "source_agent_index": index, "stance": case["stance"]}
        for position, (index, case) in enumerate(others, start=1)
    ]
    return peers, order


def run_advocacy(
    model,
    tokenizer,
    item: Mapping[str, Any],
    order_seed: int = 0,
    generation_seed: int = 0,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    enable_thinking: bool | None = False,
    rebuttal: bool = False,
    seed_hook: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """One advocate per label, optionally followed by one rebuttal round."""
    item_id = item.get("id")
    stances = assigned_stances(item_id, order_seed)
    cases: list[dict[str, Any]] = []
    contexts: list[list[dict[str, str]]] = []
    input_tokens = output_tokens = calls = 0
    start = time.perf_counter()

    for agent_index, stance in enumerate(stances):
        question = advocate_question(item, stance)
        messages = build_messages(question, system_prompt=ADVOCATE_SYSTEM_PROMPT)
        agent_seed = stable_seed(generation_seed, item_id, f"advocate_{agent_index}")
        if seed_hook is not None:
            seed_hook(agent_seed)
        reply = strip_think(
            chat(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
        )
        calls += 1
        input_tokens += count_tokens(tokenizer, question) or 0
        output_tokens += count_tokens(tokenizer, reply) or 0
        contexts.append(messages + [{"role": "assistant", "content": reply}])
        cases.append(
            {
                "agent_index": agent_index,
                "stance": stance,
                "analysis": reply,
                "support": parse_support_level(reply),
                # If the advocate volunteers a label of its own, a mismatch with
                # the assigned stance means it did not take the assigned side.
                "stated_label": parse_stance(reply),
                "generation_seed": agent_seed,
            }
        )

    peer_orders = []
    if rebuttal:
        revised = []
        for agent_index, case in enumerate(cases):
            peers, order = peer_order(cases, agent_index, item_id, order_seed)
            prompt = REBUTTAL_TEMPLATE.format(others=format_peers(peers))
            messages = contexts[agent_index] + [{"role": "user", "content": prompt}]
            agent_seed = stable_seed(
                generation_seed, item_id, f"advocate_rebuttal_{agent_index}"
            )
            if seed_hook is not None:
                seed_hook(agent_seed)
            reply = strip_think(
                chat(
                    model,
                    tokenizer,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
            )
            calls += 1
            input_tokens += count_tokens(tokenizer, prompt) or 0
            output_tokens += count_tokens(tokenizer, reply) or 0
            peer_orders.append({"agent_index": agent_index, "peer_order": order})
            revised.append(
                {
                    **case,
                    "analysis": reply,
                    "initial_analysis": case["analysis"],
                    "support": parse_support_level(reply) or case["support"],
                    "initial_support": case["support"],
                    "stated_label": parse_stance(reply),
                }
            )
        cases = revised

    return {
        "assigned_stances": stances,
        "cases": cases,
        "peer_orders": peer_orders,
        "rebuttal": rebuttal,
        "calls": calls,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def judge_candidates(
    cases: Sequence[Mapping[str, Any]], item_id: Any, order_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anonymize the advocates and shuffle their order for the judge."""
    indices = list(range(len(cases)))
    random.Random(stable_seed(order_seed, item_id, "judge_candidates")).shuffle(indices)
    candidates, order = [], []
    for position, source_index in enumerate(indices):
        candidate_id = chr(ord("A") + position)
        case = cases[source_index]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stance_argued": case["stance"],
                "analysis": case["analysis"],
            }
        )
        order.append(
            {
                "candidate_id": candidate_id,
                "source_agent_index": source_index,
                "stance_argued": case["stance"],
            }
        )
    return candidates, order


def judge_user_payload(
    item: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "issue": item.get("issue", ""),
        "headline": item.get("headline", ""),
        "article": item.get("article", ""),
        "analyses": list(candidates),
        "allowed_labels": list(LABELS),
        "output_schema": JUDGE_SCHEMA,
    }


def run_advocacy_judge(
    model,
    tokenizer,
    item: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    order_seed: int = 0,
    max_new_tokens: int = 384,
    max_retries: int = 1,
    temperature: float = 0.0,
    enable_thinking: bool | None = False,
) -> dict[str, Any]:
    candidates, candidate_order = judge_candidates(cases, item.get("id"), order_seed)
    payload = judge_user_payload(item, candidates)
    user_text = json.dumps(payload, ensure_ascii=False)
    attempts = []
    parsed = None
    current_text = user_text
    start = time.perf_counter()
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
    return {
        "prediction": parsed["label"] if parsed else None,
        "parsed_output": parsed,
        "raw_output": attempts[-1]["raw_output"] if attempts else None,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "candidate_order": candidate_order,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }


def support_fallback(
    cases: Sequence[Mapping[str, Any]], item_id: Any, fallback_seed: int = 0
) -> tuple[str, str]:
    """Used only when the judge returns nothing.

    There is no majority to fall back on here, so the label whose advocate
    reported the strongest support wins; ties are broken deterministically.
    """
    ranked = [
        (SUPPORT_RANK.get(case.get("support") or "", -1), case["stance"]) for case in cases
    ]
    best = max(rank for rank, _ in ranked)
    winners = sorted(stance for rank, stance in ranked if rank == best)
    if len(winners) == 1:
        return winners[0], "strongest_declared_support"
    if best < 0:
        reason = "no_declared_support_deterministic_fallback"
    else:
        reason = "tied_declared_support_deterministic_fallback"
    chosen = random.Random(stable_seed(fallback_seed, item_id, "advocacy_fallback")).choice(
        winners
    )
    return chosen, reason


def compliance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Did the advocates argue the side they were given?"""
    stated = [
        case for case in cases if case.get("stated_label") in LABELS
    ]
    mismatched = [
        case for case in stated if case["stated_label"] != case["stance"]
    ]
    return {
        "agents": len(cases),
        "stated_label_present": len(stated),
        "stated_label_mismatch": len(mismatched),
        "mismatched_stances": [case["stance"] for case in mismatched],
        "declared_support": {case["stance"]: case.get("support") for case in cases},
    }

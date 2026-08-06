"""Stance-advocacy debate: one agent per label, R rounds, then a judge.

Every label is argued for exactly once, so unlike free-form sampling the gold
label always appears among the candidates. That directly targets the failure
this repository measured on saved runs: most residual errors are items where no
agent ever proposed the gold label, which no aggregation rule can repair.

Round 0 is the independent case for each assigned label; every later round shows
each agent the other agents' latest cases so it can rebut or concede. Two rounds
is the base configuration, matching PREDICT's fixed two-round debate; the round
count is a parameter so the iteration limit can be raised later.

Consensus is never required or checked. The three agents disagree by
construction and the judge always decides, as in ToC, PREDICT and MAD. The
fallback here is not a consensus rule: it only covers a judge whose output
carries no stance label at all.

Model calls arrive through the same lazy shim used by :mod:`src.selective_judge`,
so prompt construction, parsing and routing stay importable without torch.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .consensus import LABELS, deterministic_fallback, parse_stance, stable_seed
from .prompts.advocacy import (
    JUDGE_SCHEMA,
    STANCE_LABELS,
    advocate_question,
    format_peers,
    get_prompt_style,
)
from .selective_judge import count_tokens, parse_judge_json, repair_payload


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


def assigned_stances(item_id: Any, order_seed: int) -> list[str]:
    """Which agent index argues which label, shuffled so the agent index carries
    no fixed meaning across items."""
    stances = list(STANCE_LABELS)
    random.Random(stable_seed(order_seed, item_id, "advocate_assignment")).shuffle(stances)
    return stances


def peer_order(
    analyses: Sequence[str],
    stances: Sequence[str],
    agent_index: int,
    item_id: Any,
    order_seed: int,
    round_index: int,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """The other agents' latest cases, shuffled reproducibly per round."""
    others = [index for index in range(len(analyses)) if index != agent_index]
    random.Random(
        stable_seed(order_seed, item_id, f"advocate_peers_{round_index}_{agent_index}")
    ).shuffle(others)
    peers = [(stances[index], analyses[index]) for index in others]
    order = [
        {"position": position, "source_agent_index": index, "stance": stances[index]}
        for position, index in enumerate(others, start=1)
    ]
    return peers, order


def run_advocacy(
    model,
    tokenizer,
    item: Mapping[str, Any],
    rounds: int = 2,
    order_seed: int = 0,
    generation_seed: int = 0,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    enable_thinking: bool | None = False,
    prompt_style: str = "toc",
    seed_hook: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """One advocate per label for ``rounds`` rounds."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    style = get_prompt_style(prompt_style)
    item_id = item.get("id")
    stances = assigned_stances(item_id, order_seed)
    contexts = [
        build_messages(
            advocate_question(item, stance, style=prompt_style),
            system_prompt=style["advocate_system"],
        )
        for stance in stances
    ]
    analyses_by_round: list[list[str]] = []
    peer_orders: list[dict[str, Any]] = []
    input_tokens = output_tokens = calls = 0
    start = time.perf_counter()

    for round_index in range(rounds):
        round_analyses = []
        for agent_index, stance in enumerate(stances):
            if round_index > 0:
                peers, order = peer_order(
                    analyses_by_round[round_index - 1], stances, agent_index,
                    item_id, order_seed, round_index,
                )
                prompt = style["rebuttal"].format(others=format_peers(peers))
                contexts[agent_index].append({"role": "user", "content": prompt})
                peer_orders.append(
                    {"round": round_index, "agent_index": agent_index, "peer_order": order}
                )
            input_tokens += count_tokens(tokenizer, contexts[agent_index][-1]["content"]) or 0
            agent_seed = stable_seed(
                generation_seed, item_id, f"advocate_{round_index}_{agent_index}"
            )
            if seed_hook is not None:
                seed_hook(agent_seed)
            reply = strip_think(
                chat(
                    model,
                    tokenizer,
                    contexts[agent_index],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
            )
            contexts[agent_index].append({"role": "assistant", "content": reply})
            round_analyses.append(reply)
            calls += 1
            output_tokens += count_tokens(tokenizer, reply) or 0
        analyses_by_round.append(round_analyses)

    cases = [
        {
            "agent_index": agent_index,
            "stance": stances[agent_index],
            "analysis": analyses_by_round[-1][agent_index],
            # A volunteered label that differs from the assigned one means the
            # agent did not argue the side it was given.
            "stated_label": parse_stance(analyses_by_round[-1][agent_index]),
        }
        for agent_index in range(len(stances))
    ]
    return {
        "prompt_style": prompt_style,
        "rounds": rounds,
        "assigned_stances": stances,
        "analyses_by_round": analyses_by_round,
        "cases": cases,
        "peer_orders": peer_orders,
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
    prompt_style: str = "toc",
) -> dict[str, Any]:
    """Judge the final cases.

    The output contract follows the prompt style: ToC's judge writes prose and
    names the label in its final line, so it is read with the same parser the
    debate methods use; our variant keeps the strict JSON schema. A JSON answer
    that fails validation is still searched for a stance label before the item
    is handed to the fallback, since a truncated object usually carries one.
    """
    style = get_prompt_style(prompt_style)
    judge_system_prompt = style["judge_system"]
    output_format = style["judge_output"]
    candidates, candidate_order = judge_candidates(cases, item.get("id"), order_seed)
    payload = judge_user_payload(item, candidates)
    user_text = json.dumps(payload, ensure_ascii=False)
    attempts: list[dict[str, Any]] = []
    prediction = None
    parsed = None
    recovered = False
    current_text = user_text
    start = time.perf_counter()
    for attempt_index in range(max_retries + 1):
        messages = build_messages(current_text, system_prompt=judge_system_prompt)
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
        if output_format == "final_line":
            prediction = parse_stance(raw)
            if prediction is None:
                parse_error = "no stance label in the final line"
        else:
            try:
                parsed = parse_judge_json(raw)
                prediction = parsed["label"]
            except ValueError as exc:
                parse_error = str(exc)
                salvaged = parse_stance(raw)
                if salvaged is not None:
                    prediction, recovered = salvaged, True
                    parse_error = f"{exc} (label recovered from the text)"
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
        current_text = (
            style["judge_repair"].format(invalid_output=raw)
            if style["judge_repair"]
            else repair_payload(payload, raw)
        )
    return {
        "prediction": prediction,
        "parsed_output": parsed,
        "raw_output": attempts[-1]["raw_output"] if attempts else None,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "output_format": output_format,
        "label_recovered_from_text": recovered,
        "candidate_order": candidate_order,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }


def judge_failure_fallback(item_id: Any, fallback_seed: int = 0) -> tuple[str, str]:
    """Last resort when the judge produced no label in any attempt.

    There is no majority to fall back on: the advocates disagree by design. This
    is a broken generation rather than an unresolved debate, so the label is
    chosen deterministically from the item id and the count is reported. A run
    with a non-trivial fallback count should be treated as invalid.
    """
    return deterministic_fallback(item_id, fallback_seed), "judge_no_label_deterministic"


def compliance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Did the advocates argue the side they were given?"""
    stated = [case for case in cases if case.get("stated_label") in LABELS]
    mismatched = [case for case in stated if case["stated_label"] != case["stance"]]
    return {
        "agents": len(cases),
        "stated_label_present": len(stated),
        "stated_label_mismatch": len(mismatched),
        "mismatched_stances": [case["stance"] for case in mismatched],
    }


def round_label_trajectory(analyses_by_round, stances) -> list[dict[str, Any]]:
    """Per round, the label each agent's text actually reads as.

    Useful when the round limit is raised: it shows whether advocates hold their
    assigned side or drift, without that ever affecting the decision.
    """
    return [
        {
            "round": round_index,
            "stated_labels": [parse_stance(text) for text in round_analyses],
            "held_assigned": [
                parse_stance(text) in (None, stance)
                for text, stance in zip(round_analyses, stances)
            ],
        }
        for round_index, round_analyses in enumerate(analyses_by_round)
    ]

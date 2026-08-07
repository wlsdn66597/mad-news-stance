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
import math
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


def count_message_tokens(tokenizer, messages: Sequence[Mapping[str, Any]]) -> int | None:
    """Tokens the model actually receives for ``messages``.

    Counting only the newest user turn omits the article, the system prompt and
    the agent's own earlier turns, so it understates every round after the first
    by most of the prompt. Rendering the chat template is what generation itself
    does, so it is the only count worth saving.
    """
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = "\n".join(str(message.get("content", "")) for message in messages)
    return count_tokens(tokenizer, text)


STANCE_MARKER = re.compile(
    r"final\s*stance\s*[:=\-]>?\s*[\"'(]?\s*([a-z]+)", re.IGNORECASE
)


def stated_label(text: str) -> tuple[str | None, str | None]:
    """The label an advocate's own text claims, and how firmly.

    ``marker`` means the text carried an explicit "Final stance:" line, which is
    a real statement of position. ``mention`` means no marker was present and
    the label is only the last one named anywhere in the prose -- an advocate
    that dutifully discusses counter-evidence names other labels all the time,
    so a mention is not evidence that it abandoned its assigned side.
    """
    if not isinstance(text, str):
        return None, None
    matches = STANCE_MARKER.findall(text)
    if matches:
        for label in LABELS:
            if matches[-1].lower().startswith(label[:4]):
                return label, "marker"
    lowered = text.lower()
    found = [label for label in LABELS if label in lowered]
    if found:
        return max(found, key=lambda label: lowered.rfind(label)), "mention"
    return None, None


def parse_final_verdict(text: str) -> str | None:
    """The label a prose judgement settles on.

    ToC asks for the stance value in the *final sentence*, after the discussion,
    so the last sentence is read first. Taking the last label mentioned anywhere
    would pick up a label named while weighing a rationale that was then
    rejected.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    marker, source = stated_label(text)
    if source == "marker":
        return marker
    lines = [line for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        lowered = line.lower()
        found = [label for label in LABELS if label in lowered]
        if found:
            return max(found, key=lambda label: lowered.rfind(label))
    return marker


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


def resolve_round_index(analyses_by_round: Sequence[Sequence[str]], judge_round: Any = "last") -> int:
    """Which round's cases the judge reads.

    84.7% of round-1 outputs address another analyst rather than the article,
    and the judge only ever sees the last round, so "which round" is an
    experimental condition rather than an implementation detail.
    """
    total = len(analyses_by_round)
    if total == 0:
        raise ValueError("no advocacy rounds to judge")
    if judge_round in (None, "last", "-1", -1):
        return total - 1
    index = int(judge_round)
    if index < 0:
        index += total
    if not 0 <= index < total:
        raise ValueError(f"judge round {judge_round} is outside the {total} rounds available")
    return index


def cases_from_round(
    analyses_by_round: Sequence[Sequence[str]],
    stances: Sequence[str],
    judge_round: Any = "last",
) -> list[dict[str, Any]]:
    """The per-label cases as they stood at the end of ``judge_round``."""
    round_index = resolve_round_index(analyses_by_round, judge_round)
    analyses = analyses_by_round[round_index]
    cases = []
    for agent_index, stance in enumerate(stances):
        analysis = analyses[agent_index]
        label, source = stated_label(analysis)
        cases.append(
            {
                "agent_index": agent_index,
                "stance": stance,
                "analysis": analysis,
                "stated_label": label,
                "stated_label_source": source,
                "from_round": round_index,
            }
        )
    return cases


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
    cost_by_round: list[dict[str, Any]] = []
    input_tokens = output_tokens = calls = 0
    start = time.perf_counter()

    for round_index in range(rounds):
        round_start = time.perf_counter()
        round_in = round_out = 0
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
            round_in += count_message_tokens(tokenizer, contexts[agent_index]) or 0
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
            round_out += count_tokens(tokenizer, reply) or 0
        analyses_by_round.append(round_analyses)
        input_tokens += round_in
        output_tokens += round_out
        cost_by_round.append(
            {
                "round": round_index,
                "calls": len(round_analyses),
                "latency_seconds": time.perf_counter() - round_start,
                "input_tokens": round_in,
                "output_tokens": round_out,
                "output_chars": [len(text) for text in round_analyses],
            }
        )

    cases = cases_from_round(analyses_by_round, stances)
    return {
        "prompt_style": prompt_style,
        "rounds": rounds,
        "assigned_stances": stances,
        "analyses_by_round": analyses_by_round,
        "cases": cases,
        "peer_orders": peer_orders,
        "cost_by_round": cost_by_round,
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
    item: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    prompt_style: str = "toc",
) -> str | dict[str, Any]:
    """The judge input, in the format its prompt style asks for.

    ToC presents the article and one rationale per stance value as plain text
    and expects prose back, so a JSON output schema has no place in it. Our
    variant keeps the JSON payload this repository's selective judge uses.
    """
    style = get_prompt_style(prompt_style)
    if style["judge_input"] == "text":
        analyses = "\n".join(
            style["judge_analysis"].format(
                stance=candidate["stance_argued"], analysis=candidate["analysis"]
            )
            for candidate in candidates
        )
        return style["judge_user"].format(
            headline=item.get("headline", ""),
            article=item.get("article", ""),
            issue=item.get("issue", ""),
            analyses=analyses,
        )
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
    that fails validation is retried first; only on the last attempt is the text
    searched for a stance label, since a truncated object usually carries one.
    Salvaging before the retries are spent silently accepts malformed output and
    reports the run as clean, which is how a past run showed retries=0 while a
    large share of its labels had never been parsed from valid JSON.
    """
    style = get_prompt_style(prompt_style)
    judge_system_prompt = style["judge_system"]
    output_format = style["judge_output"]
    candidates, candidate_order = judge_candidates(cases, item.get("id"), order_seed)
    payload = judge_user_payload(item, candidates, prompt_style)
    user_text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
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
            prediction = parse_final_verdict(raw)
            if prediction is None:
                parse_error = "no stance value in the closing sentences"
        else:
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
        # the repair turn has to carry the article and the rationales again: a
        # judge asked to "answer again" with only its own invalid output in the
        # context has nothing left to judge.
        current_text = (
            style["judge_repair"].format(original_input=user_text, invalid_output=raw)
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


def run_single_advocate(
    model,
    tokenizer,
    item: Mapping[str, Any],
    stance: str,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    enable_thinking: bool | None = False,
    prompt_style: str = "toc",
    generation_seed: int = 0,
    seed_hook: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """One round-0 case for one label, with no peers.

    Selective advocacy commissions a case only for the labels no free-form agent
    proposed, so it needs the advocate call on its own rather than the full
    three-agent round loop.
    """
    style = get_prompt_style(prompt_style)
    messages = build_messages(
        advocate_question(item, stance, style=prompt_style),
        system_prompt=style["advocate_system"],
    )
    seed = stable_seed(generation_seed, item.get("id"), f"single_advocate_{stance}")
    if seed_hook is not None:
        seed_hook(seed)
    start = time.perf_counter()
    input_tokens = count_message_tokens(tokenizer, messages) or 0
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
    label, source = stated_label(reply)
    return {
        "stance": stance,
        "analysis": reply,
        "stated_label": label,
        "stated_label_source": source,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": count_tokens(tokenizer, reply) or 0,
        },
    }


def position_bias_chi_square(counts: Mapping[str, int]) -> dict[str, Any]:
    """Is the winning candidate position uniform?

    Candidate order is shuffled per item, so a non-uniform winning position is
    the judge preferring a slot rather than an artifact of the layout. With
    three positions the test has two degrees of freedom, where the chi-square
    survival function is exactly exp(-x/2), so no scipy dependency is needed.
    """
    observed = [int(value) for key, value in counts.items() if key]
    total = sum(observed)
    if total == 0 or len(observed) < 2:
        return {"total": total, "chi_square": None, "df": None, "p_value": None}
    expected = total / len(observed)
    chi_square = sum((value - expected) ** 2 / expected for value in observed)
    df = len(observed) - 1
    p_value = math.exp(-chi_square / 2) if df == 2 else None
    return {
        "total": total,
        "chi_square": chi_square,
        "df": df,
        "p_value": p_value,
        "note": None if df == 2 else "p-value only computed for the 3-position case",
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
    """Did the advocates argue the side they were given?

    Only an explicit "Final stance:" line counts as the advocate stating a
    position. A label that merely appears last in the prose is reported
    separately: an advocate discussing counter-evidence names the other labels
    routinely, so counting those as defections wildly overstates the rate.
    """
    declared = [case for case in cases if case.get("stated_label_source") == "marker"]
    defected = [case for case in declared if case["stated_label"] != case["stance"]]
    mentioned = [case for case in cases if case.get("stated_label_source") == "mention"]
    mention_differs = [case for case in mentioned if case["stated_label"] != case["stance"]]
    return {
        "agents": len(cases),
        "declared_label": len(declared),
        "declared_defection": len(defected),
        "defected_stances": [case["stance"] for case in defected],
        "last_mention_only": len(mentioned),
        "last_mention_differs": len(mention_differs),
    }


def round_label_trajectory(analyses_by_round, stances) -> list[dict[str, Any]]:
    """Per round, the label each agent's text actually reads as.

    Useful when the round limit is raised: it shows whether advocates hold their
    assigned side or concede, without that ever affecting the decision. Only an
    explicit declaration counts as conceding; see :func:`compliance`.
    """
    trajectory = []
    for round_index, round_analyses in enumerate(analyses_by_round):
        labels, sources = zip(*(stated_label(text) for text in round_analyses))
        trajectory.append(
            {
                "round": round_index,
                "stated_labels": list(labels),
                "stated_label_sources": list(sources),
                # only an explicit declaration counts as leaving the assigned side
                "held_assigned": [
                    not (source == "marker" and label != stance)
                    for label, source, stance in zip(labels, sources, stances)
                ],
            }
        )
    return trajectory

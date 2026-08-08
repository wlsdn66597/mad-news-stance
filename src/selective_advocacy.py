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
import re
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from .advocacy import build_messages, chat, count_message_tokens, strip_think
from .consensus import LABELS, normalize_label, parse_stance, stable_seed, unique_majority
from .selective_judge import JUDGE_SCHEMA, count_tokens, parse_judge_json

# "a label nobody proposed" sounds selective and is not: with three agents and
# three labels a missing label is the normal case, not the exception. Measured on
# the EXAONE test run, 830 of 1001 items are unanimous (two labels missing) and
# only 9 are 1:1:1, so triggering on a missing label fires on 99% of items and
# the method degenerates into judging everything. The vote strength is the
# selective signal: commission where the agents are split, not where they agree.
TRIGGERS = ("instability", "non_unanimous", "unstable_or_split", "missing_label", "all")

# --------------------------------------------------- selective advocacy judge
#
# Forcing one advocate per label makes every item 1:1:1, which throws away the
# only aggregate signal a small model produces: how many independent agents
# converged on a label. This judge keeps that signal. It is told which
# rationales came from agents that chose the stance themselves and which were
# commissioned to argue a stance nobody chose, and it is given the vote count.

ABLATIONS = ("full", "no_commissioned", "no_votes", "article_only", "article_first")

QUOTE_PATTERN = re.compile(r"[\"“”'‘’「『]([^\"“”'‘’「』」]{6,200})[\"“”'‘’」』]")
VERIFIED_MARK = " [verified: this passage appears in the article]"
UNVERIFIED_MARK = " [unverified: this passage is not in the article]"


def tag_quotes(text: str, article: str) -> tuple[str, int, int]:
    """Mark each quoted passage as present in the article or not.

    Khan et al. (ICML 2024) tag every debater quote verified or unverified
    against the source and tell the judge to trust only the verified ones;
    removing that tool is what lets an incorrect debater build a persuasive
    false narrative. Measured here, only 19.3% of the advocates' Korean quotes
    share even their first twelve characters with the article, so the judge is
    currently reading mostly invented citations with no way to tell.

    The check is exact substring after whitespace normalisation, which is
    strict: a real quote loosened by a dropped particle is marked unverified.
    """
    article_text = " ".join(str(article or "").split())
    verified = unverified = 0
    out, last = [], 0
    for match in QUOTE_PATTERN.finditer(text or ""):
        quote = " ".join(match.group(1).split())
        ok = bool(quote) and quote in article_text
        verified += ok
        unverified += not ok
        out.append(text[last:match.end()])
        out.append(VERIFIED_MARK if ok else UNVERIFIED_MARK)
        last = match.end()
    out.append((text or "")[last:])
    return "".join(out), verified, unverified

_JUDGE_HEADER = """You are an independent adjudicator for news stance classification.

Determine the article's stance toward the specified issue as supportive,
oppositional, or neutral."""

_ORIGINS_BOTH = """Each rationale is marked with its origin. An "independent" rationale was written
by an analyst who reached that stance on its own; the number of analysts who did
so is given and is genuine evidence. A "commissioned" rationale was written by an
analyst that was told in advance to argue that stance, whether or not the article
supports it; it is advocacy, not evidence, and exists only so that no stance goes
unargued."""

_ORIGINS_INDEPENDENT = """Each rationale was written by an analyst who reached that stance on its own, and
the number of analysts who did so is given. Stances no analyst chose have no
rationale, which is itself informative rather than a gap to be filled."""

_ORIGINS_UNLABELLED = """Rationales are provided for some or all of the stances. Treat them as claims to
be checked, not as evidence in themselves."""

_VERIFY = """Verify every claimed passage against the original article."""

_JUDGE_BODY = """Judge the stance expressed by the article's framing, wording, emphasis, and
narrative structure, and distinguish the journalist's framing from opinions
merely quoted from governments, organizations, or individuals."""

_DEFER_TO_VOTE = """Overturn the independent majority only when the article itself, not the force of
a rationale, contradicts it."""

_JUDGE_CLOSE_WITH_RATIONALES = """Do not infer the answer from the order in which the rationales appear. Return one
of the three labels even when all rationales are flawed.

Return valid JSON only."""

_JUDGE_CLOSE_ARTICLE_ONLY = """Decide from the article alone. Return one of the three labels.

Return valid JSON only."""


def selective_judge_system_prompt(ablation: str = "full") -> str:
    """The judge instruction for one ablation.

    The prompt has to match the payload: telling a judge that some rationales
    are commissioned advocacy when none are, or asking it not to be swayed by
    rationale order when it is given no rationales, would make the ablation
    measure the prompt rather than the input.
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}; choose one of {', '.join(ABLATIONS)}")
    parts = [_JUDGE_HEADER]
    if ablation in ("article_only", "article_first"):
        # stage 1 of article_first is exactly the article-only condition, so the
        # two runs start from an identical first turn and stay paired
        parts += [_JUDGE_BODY, _JUDGE_CLOSE_ARTICLE_ONLY]
        return "\n\n".join(parts)
    if ablation == "no_commissioned":
        parts.append(_ORIGINS_INDEPENDENT)
    elif ablation == "no_votes":
        parts.append(_ORIGINS_UNLABELLED)
    else:
        parts.append(_ORIGINS_BOTH)
    parts += [_VERIFY, _JUDGE_BODY]
    if ablation != "no_votes":
        parts.append(_DEFER_TO_VOTE)
    parts.append(_JUDGE_CLOSE_WITH_RATIONALES)
    return "\n\n".join(parts)


SELECTIVE_JUDGE_SYSTEM_PROMPT = selective_judge_system_prompt("full")


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
    round_predictions: Sequence[Sequence[Any]],
    mode: str = "unstable_or_split",
    scope: str = "last",
) -> tuple[bool, str, list[str]]:
    """Should this item get commissioned cases and a judge?

    - ``instability``: a tied final round, or a round-0 majority the final round
      overturned. The trigger the existing selective judge uses.
    - ``non_unanimous``: the agents did not all end on the same label -- a 2:1
      split or a tie. This is where the vote is weak enough to be worth
      questioning.
    - ``unstable_or_split`` (default): the union of the two.
    - ``missing_label``: some label was never proposed. Kept for the ablation
      that shows why it is the wrong trigger; it fires on almost everything.
    - ``all``: judge every item.

    ``missing`` is returned regardless of the mode, because it is what decides
    how many cases have to be commissioned once an item does fire.
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
    final_votes = proposed_labels(round_predictions, "last")
    unanimous = bool(final_votes) and len(final_votes) == 1
    reasons = []
    if unstable:
        reasons.append("final_tie" if final.tied else "round0_final_disagree")
    if not unanimous and not final.tied:
        reasons.append("split_vote")
    if missing:
        reasons.append(f"missing:{'+'.join(missing)}")
    joined = "+".join(reasons)
    if mode == "all":
        return True, joined or "unanimous", missing
    if mode == "instability":
        return unstable, (joined if unstable else "stable"), missing
    if mode == "non_unanimous":
        return not unanimous, (joined if not unanimous else "unanimous"), missing
    if mode == "missing_label":
        return bool(missing), (joined if missing else "all_labels_proposed"), missing
    triggered = unstable or not unanimous
    return triggered, (joined if triggered else "unanimous_and_stable"), missing


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


def ablate_candidates(
    candidates: Sequence[Mapping[str, Any]], ablation: str = "full"
) -> list[dict[str, Any]]:
    """The rationales an ablation lets the judge see."""
    if ablation == "article_only":
        return []
    if ablation == "article_first":
        return [dict(c) for c in candidates]
    if ablation == "no_commissioned":
        return [dict(c) for c in candidates if c.get("origin") != "commissioned"]
    if ablation == "no_votes":
        # the rationales stay, the provenance and the count go
        return [
            {k: v for k, v in c.items() if k not in ("origin", "independent_supporters")}
            for c in candidates
        ]
    return [dict(c) for c in candidates]


def judge_payload(
    item: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    votes: Mapping[str, int],
    ablation: str = "full",
) -> dict[str, Any]:
    """The judge input for one ablation.

    ``full`` is the method. The others isolate what it is actually buying:
    ``no_commissioned`` removes the counterfactuals and leaves the free-form
    rationales, ``no_votes`` removes the vote count and the origin labels, and
    ``article_only`` removes the rationales entirely -- that last one is the
    control for "this is just a stronger model applied to 17% of the items".
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}; choose one of {', '.join(ABLATIONS)}")
    payload = {
        "issue": item.get("issue", ""),
        "headline": item.get("headline", ""),
        "article": item.get("article", ""),
    }
    if ablation not in ("no_votes", "article_only"):
        payload["independent_votes"] = {label: int(votes.get(label, 0)) for label in LABELS}
    if ablation not in ("article_only", "article_first"):
        payload["rationales"] = ablate_candidates(candidates, ablation)
    payload["allowed_labels"] = list(LABELS)
    payload["output_schema"] = JUDGE_SCHEMA
    return payload


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


_REVISION_TASK = """You judged this article on your own above.

Three analyses are now provided. Each analyst was assigned a stance in advance,
so these are advocacy arguments, not evidence. Every quoted passage has already
been checked against the article and marked verified or unverified; an analysis
whose quotes are unverified has nothing behind it.

Reconsider, and change your answer only if the article itself -- not the force
of an analysis -- shows that you were wrong. Say the same label again if nothing
here outweighs what you read.

Return valid JSON only, in the same schema as before."""


def revision_payload(
    candidates: Sequence[Mapping[str, Any]],
    votes: Mapping[str, int],
    include_votes: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task": _REVISION_TASK}
    if include_votes:
        payload["independent_votes"] = {label: int(votes.get(label, 0)) for label in LABELS}
    payload["rationales"] = list(candidates)
    payload["allowed_labels"] = list(LABELS)
    payload["output_schema"] = JUDGE_SCHEMA
    return payload


def run_two_stage_judge(
    model,
    tokenizer,
    item: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    votes: Mapping[str, int],
    first_pass_output: str,
    order_seed: int = 0,
    max_new_tokens: int = 512,
    max_retries: int = 1,
    temperature: float = 0.0,
    enable_thinking: bool | None = False,
    verify_quotes: bool = True,
) -> dict[str, Any]:
    """Judge the article alone first, then reconsider with the analyses.

    Every earlier condition showed the judge the analyses before it had formed
    any view, so its own reading of the article was never the anchor. Here the
    article-only verdict is the first turn -- replayed from a saved
    ``--ablation article_only`` run, which keeps the two conditions paired and
    costs one call per item instead of two -- and the analyses arrive as
    something to weigh against an answer already given.
    """
    candidates, candidate_order = order_candidates(cases, item.get("id"), order_seed)
    quote_stats = {"verified": 0, "unverified": 0}
    if verify_quotes:
        tagged = []
        for candidate in candidates:
            text, ok, bad = tag_quotes(candidate.get("analysis", ""), item.get("article", ""))
            quote_stats["verified"] += ok
            quote_stats["unverified"] += bad
            tagged.append({**candidate, "analysis": text})
        candidates = tagged
    stage1 = judge_payload(item, [], votes, "article_first")
    stage2 = revision_payload(candidates, votes)
    messages = [
        {"role": "system", "content": selective_judge_system_prompt("article_first")},
        {"role": "user", "content": json.dumps(stage1, ensure_ascii=False)},
        {"role": "assistant", "content": first_pass_output},
    ]
    current = json.dumps(stage2, ensure_ascii=False)
    attempts: list[dict[str, Any]] = []
    parsed = None
    prediction = None
    recovered = False
    start = time.perf_counter()
    for attempt_index in range(max_retries + 1):
        raw = strip_think(
            chat(
                model,
                tokenizer,
                messages + [{"role": "user", "content": current}],
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
                "input_tokens": count_message_tokens(
                    tokenizer, messages + [{"role": "user", "content": current}]
                ),
                "output_tokens": count_tokens(tokenizer, raw),
            }
        )
        if prediction is not None:
            break
        current = repair_request(stage2, raw)
    return {
        "prediction": prediction,
        "parsed_output": parsed,
        "raw_output": attempts[-1]["raw_output"] if attempts else None,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "label_recovered_from_text": recovered,
        "ablation": "article_first",
        "verify_quotes": verify_quotes,
        "quote_tagging": quote_stats if verify_quotes else None,
        "first_pass_prediction": parse_stance(first_pass_output),
        "candidate_order": candidate_order,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }


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
    ablation: str = "full",
) -> dict[str, Any]:
    """Adjudicate one item. Malformed JSON is retried before anything is salvaged."""
    candidates, candidate_order = order_candidates(cases, item.get("id"), order_seed)
    payload = judge_payload(item, candidates, votes, ablation)
    system_prompt = selective_judge_system_prompt(ablation)
    current_text = json.dumps(payload, ensure_ascii=False)
    attempts: list[dict[str, Any]] = []
    parsed = None
    prediction = None
    recovered = False
    start = time.perf_counter()
    for attempt_index in range(max_retries + 1):
        messages = build_messages(current_text, system_prompt=system_prompt)
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
        "ablation": ablation,
        "candidate_order": candidate_order,
        "latency_seconds": time.perf_counter() - start,
        "token_usage": {
            "input_tokens": sum(a["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["output_tokens"] or 0 for a in attempts),
        },
    }


def judge_input_tokens(tokenizer, item, cases, votes, order_seed=0, ablation="full"):
    """Prompt size the judge will actually see, for a cost estimate before a run."""
    candidates, _ = order_candidates(cases, item.get("id"), order_seed)
    payload = judge_payload(item, candidates, votes, ablation)
    messages = build_messages(
        json.dumps(payload, ensure_ascii=False),
        system_prompt=selective_judge_system_prompt(ablation),
    )
    return count_message_tokens(tokenizer, messages)

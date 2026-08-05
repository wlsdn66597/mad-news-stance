"""Selective extra reconsideration round for saved stance debate trajectories.

The original debate agents are asked once more, each in its own saved context,
after being shown the anonymized final responses of the other agents.  The
labels are collected and unanimity is decided in code; the prompt never asks the
agents to agree.

This is *not* a re-implementation of a published consensus protocol.  It is
``selective independent reconsideration with unanimity-based acceptance``
applied to the existing MAD pipeline (see docs/selective_consensus.md).

Like :mod:`src.selective_judge`, this module keeps prompt building, parsing and
the unanimity rule importable without torch/transformers; the model call is
resolved lazily.
"""
from __future__ import annotations

import random
import re
import time
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from .consensus import LABELS, normalize_label, parse_stance, stable_seed
from .prompts.stance import PROFILES
from .selective_judge import count_tokens

#: The only textual difference between the two consensus prompt modes.
RECONSIDERATION_INSTRUCTION = (
    "Re-evaluate the stance independently against the original article,\n"
    "without following the majority automatically. Keep or revise your\n"
    "answer based on the article evidence."
)

CONSENSUS_PROMPT_MODES = ("plain_extra_round", "independent_reconsideration")
NO_CONSENSUS_MODE = "none"


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
    # Same rule as src.llm.strip_think, re-implemented so parsing stays importable
    # without torch. The saved debate answers were stripped the same way.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Mirrors src.debate.format_others / format_paper_others.  Duplicated on purpose
# so offline prompt construction never imports the generation runtime.
def format_labeled_agents(answers: Sequence[str]) -> str:
    return "\n\n".join(f"[Agent {index + 1}]\n{answer}" for index, answer in enumerate(answers))


def format_paper_others(answers: Sequence[str]) -> str:
    return "".join(f"\n\n One agent solution: ```{answer}```" for answer in answers)


PEER_FORMATTERS = {
    "labeled_agents": format_labeled_agents,
    "paper_one_agent_solution": format_paper_others,
}


def debate_trace(record: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = record.get("debate_trace")
    return trace if isinstance(trace, Mapping) else {}


def resolve_prompt_profile(record: Mapping[str, Any], meta: Mapping[str, Any] | None = None):
    """Return the stance prompt profile used by the saved run, when known."""
    meta = meta or {}
    name = (
        record.get("prompt_profile")
        or debate_trace(record).get("prompt_profile")
        or meta.get("prompt_profile")
    )
    return PROFILES.get(name), name


def resolve_debate_template(
    record: Mapping[str, Any], meta: Mapping[str, Any] | None = None, override: str | None = None
) -> tuple[str, str]:
    """The debate prompt of the saved run is reused verbatim."""
    if override:
        return override, "cli_override"
    saved = debate_trace(record).get("debate_template")
    if saved:
        return saved, "saved_debate_trace"
    profile, name = resolve_prompt_profile(record, meta)
    if profile and profile.debate_template:
        return profile.debate_template, f"prompt_profile:{name}"
    raise ValueError(
        "no debate template found; the saved result has no debate_trace.debate_template "
        "and no known prompt profile. Pass --consensus-debate-template."
    )


def resolve_peer_formatter(
    record: Mapping[str, Any], meta: Mapping[str, Any] | None = None
) -> tuple[Callable[[Sequence[str]], str], str]:
    saved_format = debate_trace(record).get("other_answers_format")
    if saved_format == "paper_one_agent_solution":
        return PEER_FORMATTERS[saved_format], saved_format
    profile, name = resolve_prompt_profile(record, meta)
    if profile is not None:
        return profile.format_other_answers, f"prompt_profile:{name}"
    return format_labeled_agents, "labeled_agents"


def consensus_prompt(debate_template: str, peer_text: str, mode: str, question: str = "") -> str:
    """The saved debate prompt, optionally plus the independence instruction."""
    if mode not in CONSENSUS_PROMPT_MODES:
        raise ValueError(f"unknown consensus prompt mode: {mode}")
    base = debate_template.format(others=peer_text, question=question)
    if mode == "independent_reconsideration":
        return f"{base}\n\n{RECONSIDERATION_INSTRUCTION}"
    return base


def ordered_peer_answers(
    final_answers: Sequence[str], agent_index: int, item_id: Any, order_seed: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """Shuffle the peer responses reproducibly and report the realized order."""
    others = [(index, answer) for index, answer in enumerate(final_answers) if index != agent_index]
    random.Random(
        stable_seed(order_seed, item_id, f"consensus_peers_{agent_index}")
    ).shuffle(others)
    order = [
        {"position": position, "source_agent_index": source_index}
        for position, (source_index, _) in enumerate(others, start=1)
    ]
    return [answer for _, answer in others], order


def consensus_decision(labels: Sequence[Any]) -> dict[str, Any]:
    """Unanimity is decided here, never by the prompt."""
    normalized = [normalize_label(label) for label in labels]
    counts = Counter(label for label in normalized if label in LABELS)
    if not normalized or len(counts) == 0 or sum(counts.values()) != len(normalized):
        return {
            "consensus_reached": False,
            "consensus_label": None,
            "consensus_reason": "parse_failure",
            "label_counts": dict(counts),
            "parse_failure": True,
        }
    if len(counts) == 1:
        return {
            "consensus_reached": True,
            "consensus_label": next(iter(counts)),
            "consensus_reason": "unanimous",
            "label_counts": dict(counts),
            "parse_failure": False,
        }
    split = "_".join(str(count) for count in sorted(counts.values(), reverse=True))
    return {
        "consensus_reached": False,
        "consensus_label": None,
        "consensus_reason": f"split_{split}",
        "label_counts": dict(counts),
        "parse_failure": False,
    }


def agent_history(
    record: Mapping[str, Any],
    agent_index: int,
    final_answers: Sequence[str],
    meta: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Reuse the saved agent context; rebuild the minimum when it is absent."""
    trace = debate_trace(record)
    contexts = trace.get("agent_contexts")
    if isinstance(contexts, Sequence) and agent_index < len(contexts) and contexts[agent_index]:
        return [dict(message) for message in contexts[agent_index]], "saved_agent_contexts"
    question = trace.get("question") or record.get("prompt") or record.get("question_or_prompt")
    if not question:
        raise ValueError(
            f"item {record.get('id')} has neither agent_contexts nor a saved question; "
            "the consensus round cannot restore the agent context"
        )
    system_prompt = trace.get("system_prompt") or (meta or {}).get("system_prompt")
    history = build_messages(question, system_prompt=system_prompt)
    history.append({"role": "assistant", "content": final_answers[agent_index]})
    return history, "reconstructed_from_question_and_own_answer"


def run_consensus_round(
    model,
    tokenizer,
    record: Mapping[str, Any],
    mode: str,
    meta: Mapping[str, Any] | None = None,
    debate_template: str | None = None,
    order_seed: int = 0,
    generation_seed: int = 0,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    enable_thinking: bool | None = False,
    seed_hook: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Ask every original agent once more, then decide unanimity in code."""
    if mode not in CONSENSUS_PROMPT_MODES:
        raise ValueError(f"unknown consensus prompt mode: {mode}")
    item_id = record.get("id")
    final_answers = list(record["raw_rounds"][-1])
    previous_labels = [parse_stance(answer) for answer in final_answers]
    template, template_source = resolve_debate_template(record, meta, debate_template)
    formatter, formatter_source = resolve_peer_formatter(record, meta)
    question = debate_trace(record).get("question") or record.get("prompt") or ""

    outputs, labels, peer_orders, prompts, agent_seeds = [], [], [], [], []
    context_sources = []
    input_tokens = output_tokens = 0
    start = time.perf_counter()
    for agent_index in range(len(final_answers)):
        peers, order = ordered_peer_answers(final_answers, agent_index, item_id, order_seed)
        prompt = consensus_prompt(template, formatter(peers), mode, question=question)
        history, context_source = agent_history(record, agent_index, final_answers, meta)
        messages = history + [{"role": "user", "content": prompt}]
        agent_seed = stable_seed(generation_seed, item_id, f"consensus_agent_{agent_index}")
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
        outputs.append(reply)
        labels.append(parse_stance(reply))
        peer_orders.append({"agent_index": agent_index, "peer_order": order})
        prompts.append(prompt)
        agent_seeds.append(agent_seed)
        context_sources.append(context_source)
        input_tokens += count_tokens(tokenizer, prompt) or 0
        output_tokens += count_tokens(tokenizer, reply) or 0
    elapsed = time.perf_counter() - start

    decision = consensus_decision(labels)
    return {
        "consensus_prompt_mode": mode,
        "previous_agent_labels": previous_labels,
        "revised_agent_labels": labels,
        "revised_agent_outputs": outputs,
        "consensus_prompts": prompts,
        "peer_orders": peer_orders,
        "agent_context_source": context_sources[0] if context_sources else None,
        "debate_template_source": template_source,
        "peer_format_source": formatter_source,
        "agent_seeds": agent_seeds,
        "calls": len(outputs),
        "latency_seconds": elapsed,
        "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        **decision,
    }

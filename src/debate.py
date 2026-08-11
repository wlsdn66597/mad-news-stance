"""Multi-Agent Debate 구현 (Du et al., 2023 구조 재현).

라운드 0: 각 에이전트 독립 답변. 라운드 r>0: 다른 에이전트들의 직전 답변을 주고 갱신.
debate 프롬프트(debate_template)는 task 가 언어/포맷에 맞게 제공한다(영어=MMLU/GSM8K, 한국어=stance).

agent 수가 커질 때는 shared memory agent가 직전 라운드 전체 응답을 한 번 요약하고,
각 agent가 같은 압축 메모리를 읽는 변형을 선택할 수 있다.
"""
from .llm import build_messages, chat, strip_think

# task 가 debate_template 을 안 주면 쓰는 기본값(한국어 generic).
DEFAULT_DEBATE_TEMPLATE = (
    "다음은 다른 에이전트들이 같은 문제에 대해 내놓은 답변입니다:\n\n{others}\n\n"
    "다른 에이전트들의 추론을 참고하여 당신의 답변을 다시 검토하고, "
    "갱신된 답변을 제시하세요. 마지막 줄에 최종 답을 명확히 쓰세요."
)

DEFAULT_MEMORY_SUMMARY_TEMPLATE = (
    "You are the shared memory agent for a multi-agent debate. Summarize the latest "
    "responses below without solving the original problem yourself. Preserve: "
    "(1) each distinct candidate final answer, (2) the strongest supporting reasoning "
    "or evidence, (3) explicit disagreements or suspected errors, and (4) unresolved "
    "points. Attribute claims to agent labels. Be concise and do not force consensus.\n\n"
    "{responses}"
)

DEFAULT_MEMORY_DEBATE_TEMPLATE = (
    "A shared memory agent compressed the latest responses from all debate agents:\n\n"
    "{memory}\n\n"
    "Use this memory only as additional advice. Re-check the original task and your own "
    "previous reasoning. Do not change your answer merely to follow the majority; change "
    "it only when a concrete argument or piece of evidence warrants it. Give an updated "
    "answer in the format requested by the original task."
)


PEER_MODES = ("peers", "self")


def format_others(other_answers):
    return "\n\n".join(f"[Agent {i + 1}]\n{ans}" for i, ans in enumerate(other_answers))


def format_paper_others(other_answers):
    """Du et al. 공개 코드가 다른 agent 응답을 연결하는 형식."""
    return "".join(
        f"\n\n One agent solution: ```{answer}```" for answer in other_answers
    )


def format_round_answers(round_answers):
    return "\n\n".join(
        f"[Agent {i + 1}]\n{answer}" for i, answer in enumerate(round_answers)
    )


def count_tokens(tokenizer, text):
    """Best-effort token count used only for communication diagnostics."""
    try:
        tokenized = tokenizer(text, add_special_tokens=False)
        ids = tokenized["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)
    except Exception:
        return None


def _sum_if_known(values):
    return sum(values) if values and all(value is not None for value in values) else None


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def run_debate(
    model,
    tokenizer,
    question,
    debate_template=DEFAULT_DEBATE_TEMPLATE,
    system_prompt=None,
    n_agents=3,
    n_rounds=2,
    max_new_tokens=512,
    temperature=0.7,
    enable_thinking=None,
    use_memory=False,
    memory_context=None,
    prompt_profile=None,
    initial_answers=None,
    memory_summary_template=DEFAULT_MEMORY_SUMMARY_TEMPLATE,
    memory_debate_template=DEFAULT_MEMORY_DEBATE_TEMPLATE,
    memory_max_new_tokens=256,
    memory_temperature=0.0,
    other_answers_formatter=None,
    peer_mode="peers",
    self_refine_template=None,
    debate_templates=None,
    debate_protocol="baseline",
):
    if n_agents < 1:
        raise ValueError("n_agents must be at least 1")
    if n_rounds < 1:
        raise ValueError("n_rounds must be at least 1")
    if peer_mode not in PEER_MODES:
        raise ValueError(
            f"unknown peer mode: {peer_mode}; choose one of {', '.join(PEER_MODES)}"
        )
    if peer_mode == "self" and self_refine_template is None:
        # falling back to debate_template would tell the agent its own answer came
        # from someone else, which is the confound this control exists to remove
        raise ValueError("peer_mode='self' needs a self_refine_template")
    if peer_mode == "self" and debate_templates is not None:
        # every protocol prompt talks about the other agents' judgments
        raise ValueError("peer_mode='self' cannot be combined with a debate protocol")
    # one template repeats for every exchange round; a longer list assigns one
    # cognitive task per round and therefore has to match the round count
    round_templates = list(debate_templates or [debate_template])
    if len(round_templates) not in (1, max(1, n_rounds - 1)):
        raise ValueError(
            f"{len(round_templates)} debate templates for {n_rounds} rounds; "
            f"give 1 to repeat it or {n_rounds - 1}, one per exchange round"
        )
    if initial_answers is not None and len(initial_answers) != n_agents:
        raise ValueError("initial_answers must contain exactly n_agents responses")
    if other_answers_formatter is None:
        other_answers_formatter = format_others

    # a list gives each agent its own persona; a single value is shared, which
    # is the historical behaviour and produces an ensemble of near-copies
    agent_prompts = (
        list(system_prompt)
        if isinstance(system_prompt, (list, tuple))
        else [system_prompt] * n_agents
    )
    if len(agent_prompts) != n_agents:
        raise ValueError(f"got {len(agent_prompts)} system prompts for {n_agents} agents")
    agent_contexts = [
        build_messages(question, system_prompt=prompt) for prompt in agent_prompts
    ]
    answers_by_round = []
    memory_by_round = []
    communication_stats_by_round = []

    for round_index in range(n_rounds):
        memory = None
        memory_record = None
        if round_index > 0 and use_memory:
            memory_source = format_round_answers(answers_by_round[round_index - 1])
            memory_prompt = memory_summary_template.format(
                responses=memory_source,
                task_context=memory_context or "",
            )
            memory_messages = build_messages(memory_prompt)
            memory = strip_think(
                chat(
                    model,
                    tokenizer,
                    memory_messages,
                    max_new_tokens=memory_max_new_tokens,
                    temperature=memory_temperature,
                    enable_thinking=enable_thinking,
                )
            )
            memory_record = {
                "round_index": round_index,
                "source_answers": list(answers_by_round[round_index - 1]),
                "source_text": memory_source,
                "summary_prompt": memory_prompt,
                "summary_messages": memory_messages,
                "summary": memory,
                "source_chars": len(memory_source),
                "source_tokens": count_tokens(tokenizer, memory_source),
                "summary_chars": len(memory),
                "summary_tokens": count_tokens(tokenizer, memory),
                "summary_prompt_tokens": count_tokens(tokenizer, memory_prompt),
            }
            memory_by_round.append(memory_record)

        round_answers = []
        direct_contexts = []
        delivered_prompts = []
        for agent_index in range(n_agents):
            if round_index > 0:
                if peer_mode == "self":
                    # same rounds, same generation count, no peer signal: this is
                    # the control that separates rereading from being told what
                    # someone else concluded
                    others = [answers_by_round[round_index - 1][agent_index]]
                    active_template = self_refine_template
                else:
                    others = [
                        answers_by_round[round_index - 1][other_index]
                        for other_index in range(n_agents)
                        if other_index != agent_index
                    ]
                    active_template = round_templates[
                        min(round_index - 1, len(round_templates) - 1)
                    ]
                direct_context = other_answers_formatter(others)
                direct_prompt = active_template.format(
                    others=direct_context, question=question
                )
                direct_contexts.append(direct_prompt)
                if use_memory:
                    debate_prompt = memory_debate_template.format(memory=memory)
                else:
                    debate_prompt = direct_prompt
                delivered_prompts.append(debate_prompt)
                agent_contexts[agent_index].append(
                    {"role": "user", "content": debate_prompt}
                )
            if round_index == 0 and initial_answers is not None:
                reply = initial_answers[agent_index]
            else:
                reply = chat(
                    model,
                    tokenizer,
                    agent_contexts[agent_index],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
                reply = strip_think(reply)
            agent_contexts[agent_index].append(
                {"role": "assistant", "content": reply}
            )
            round_answers.append(reply)
        answers_by_round.append(round_answers)

        direct_chars = [len(text) for text in direct_contexts]
        delivered_chars = [len(text) for text in delivered_prompts]
        direct_tokens = [count_tokens(tokenizer, text) for text in direct_contexts]
        delivered_tokens = [count_tokens(tokenizer, text) for text in delivered_prompts]
        direct_total_tokens = _sum_if_known(direct_tokens)
        delivered_total_tokens = _sum_if_known(delivered_tokens)
        direct_max_tokens = (
            max(direct_tokens)
            if direct_tokens and all(value is not None for value in direct_tokens)
            else None
        )
        delivered_max_tokens = (
            max(delivered_tokens)
            if delivered_tokens and all(value is not None for value in delivered_tokens)
            else None
        )
        summarizer_input_tokens = (
            memory_record["summary_prompt_tokens"] if memory_record is not None else 0
        )
        total_memory_path_tokens = (
            delivered_total_tokens + summarizer_input_tokens
            if delivered_total_tokens is not None and summarizer_input_tokens is not None
            else None
        )
        communication_stats_by_round.append(
            {
                "round_index": round_index,
                "mode": "shared_summary" if use_memory else "direct_concat",
                "direct_equivalent_chars_per_agent": direct_chars,
                "delivered_chars_per_agent": delivered_chars,
                "direct_equivalent_tokens_per_agent": direct_tokens,
                "delivered_tokens_per_agent": delivered_tokens,
                "direct_equivalent_total_chars": sum(direct_chars),
                "delivered_total_chars": sum(delivered_chars),
                "direct_equivalent_total_tokens": direct_total_tokens,
                "delivered_total_tokens": delivered_total_tokens,
                "summarizer_input_tokens": summarizer_input_tokens,
                "total_memory_path_tokens": total_memory_path_tokens,
                "per_agent_context_token_ratio": _ratio(
                    delivered_max_tokens, direct_max_tokens
                ),
                "total_input_token_ratio": _ratio(
                    total_memory_path_tokens, direct_total_tokens
                ),
            }
        )

    logical_agent_generation_calls = n_agents * n_rounds
    reused_initial_answers = n_agents if initial_answers is not None else 0
    executed_agent_generation_calls = logical_agent_generation_calls - reused_initial_answers
    memory_generation_calls = (n_rounds - 1) if use_memory else 0
    return {
        "question": question,
        "debate_template": debate_template,
        "system_prompt": system_prompt,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "peer_mode": peer_mode,
        "debate_protocol": debate_protocol,
        "round_templates": round_templates,
        "communication_mode": "shared_summary" if use_memory else "direct_concat",
        "prompt_profile": prompt_profile,
        "other_answers_format": (
            "paper_one_agent_solution"
            if other_answers_formatter is format_paper_others
            else "labeled_agents"
        ),
        "initial_answer_source": "provided" if initial_answers is not None else "generated",
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        },
        "memory_config": {
            "enabled": use_memory,
            "summary_template": memory_summary_template if use_memory else None,
            "debate_template": memory_debate_template if use_memory else None,
            "max_new_tokens": memory_max_new_tokens if use_memory else None,
            "temperature": memory_temperature if use_memory else None,
            "task_context_included": bool(memory_context) if use_memory else False,
        },
        "call_counts": {
            "agent_generation_calls": logical_agent_generation_calls,
            "executed_agent_generation_calls": executed_agent_generation_calls,
            "reused_initial_answers": reused_initial_answers,
            "memory_generation_calls": memory_generation_calls,
            "total_generation_calls": logical_agent_generation_calls + memory_generation_calls,
            "executed_generation_calls": executed_agent_generation_calls + memory_generation_calls,
        },
        "answers_by_round": answers_by_round,
        "agent_contexts": agent_contexts,
        "memory_by_round": memory_by_round,
        "communication_stats_by_round": communication_stats_by_round,
    }

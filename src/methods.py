"""비교 대상 method 들. 모두 (pred, raw ...) dict 를 돌려준다.

논문 정렬: 모든 method 가 동일한 temperature 를 쓴다.
- vanilla       : 직답 1회
- cot           : 단계별 추론 1회
- majority      : cot 를 k회 샘플링 후 다수결
- debate        : 다른 agent 응답을 직접 연결하는 Multi-Agent Debate
- debate_memory : 직전 round 전체를 shared memory agent가 요약한 뒤 토론
"""
from collections import Counter

from .debate import (
    DEFAULT_DEBATE_TEMPLATE,
    DEFAULT_MEMORY_DEBATE_TEMPLATE,
    DEFAULT_MEMORY_SUMMARY_TEMPLATE,
    format_paper_others,
)
from .debate import run_debate as _debate_engine
from .llm import build_messages, chat, strip_think


def _majority(preds):
    valid = [p for p in preds if p is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def _one(model, tok, task, item, sysp, max_new_tokens, temperature, style, enable_thinking):
    q = task.question(item, style=style)
    messages = build_messages(q, system_prompt=sysp)
    out = strip_think(
        chat(
            model,
            tok,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    )
    return {
        "pred": task.parse(out),
        "raw": [out],
        "prompt": q,
        "messages": messages,
    }


def run_vanilla(
    model, tok, task, item, sysp, max_new_tokens, temperature=0.7, enable_thinking=None
):
    return _one(
        model, tok, task, item, sysp, max_new_tokens, temperature, "vanilla", enable_thinking
    )


def run_cot(
    model, tok, task, item, sysp, max_new_tokens, temperature=0.7, enable_thinking=None
):
    return _one(
        model, tok, task, item, sysp, max_new_tokens, temperature, "cot", enable_thinking
    )


def run_single(
    model,
    tok,
    task,
    item,
    sysp,
    max_new_tokens,
    temperature=0.7,
    enable_thinking=None,
    initial_answers=None,
    initial_style="cot",
):
    """Single-agent baseline, optionally paired with shared Round 0.

    With shared answers, the first agent response becomes the single-agent
    prediction. This avoids another stochastic generation and keeps single,
    majority, and debate paired on the exact same Round 0 population.
    """
    if initial_answers is None:
        return _one(
            model, tok, task, item, sysp, max_new_tokens, temperature,
            "paper", enable_thinking,
        )
    if not initial_answers:
        raise ValueError("initial_answers must contain at least one response")

    question = task.question(item, style=initial_style)
    messages = build_messages(question, system_prompt=sysp)
    answer = initial_answers[0]
    pred = task.parse(answer)
    return {
        "pred": pred,
        "raw": [answer],
        "preds": [pred],
        "prompt": question,
        "messages": messages,
        "single_trace": {
            "agent_index": 0,
            "initial_answer": answer,
            "initial_pred": pred,
            "generation_calls": 0,
            "round0_source": "shared",
        },
    }


def run_reflection(
    model, tok, task, item, sysp, max_new_tokens, temperature=0.7, enable_thinking=None
):
    """논문 Single Agent (Reflection): 최초 답변 뒤 동일 agent가 self-check."""
    question = task.question(item, style="paper")
    reflection_prompt = task.reflection_template
    if not reflection_prompt:
        raise ValueError(f"task {task.name} does not define a reflection prompt")

    messages = build_messages(question, system_prompt=sysp)
    initial = strip_think(
        chat(
            model,
            tok,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    )
    messages.append({"role": "assistant", "content": initial})
    messages.append({"role": "user", "content": reflection_prompt})
    revised = strip_think(
        chat(
            model,
            tok,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    )
    messages.append({"role": "assistant", "content": revised})
    initial_pred = task.parse(initial)
    final_pred = task.parse(revised)
    return {
        "pred": final_pred,
        "raw": [initial, revised],
        "preds": [initial_pred, final_pred],
        "prompt": question,
        "messages": messages,
        "reflection_trace": {
            "initial_answer": initial,
            "initial_pred": initial_pred,
            "reflection_prompt": reflection_prompt,
            "final_answer": revised,
            "final_pred": final_pred,
            "generation_calls": 2,
        },
    }


def run_majority(
    model,
    tok,
    task,
    item,
    sysp,
    max_new_tokens,
    k=5,
    temperature=0.7,
    enable_thinking=None,
    initial_answers=None,
    initial_style="cot",
):
    q = task.question(item, style=initial_style)
    # One system prompt per agent when personas are in use, the same one k times
    # otherwise. Sampling the identical prompt k times produces near-copies:
    # measured on EXAONE-4.0-1.2B, the three agents score 0.4735, 0.4725 and
    # 0.4745, agree pairwise 86.5% of the time at round 0, and all three miss on
    # 45.4% of items where independent failures at those accuracies would miss
    # on 14.6%. That correlation is what caps the candidate pool at 0.5465, and
    # it is why `single` and `majority k=3` score identically.
    prompts = list(sysp) if isinstance(sysp, (list, tuple)) else [sysp] * k
    if len(prompts) != k:
        raise ValueError(f"got {len(prompts)} system prompts for {k} agents")
    messages = build_messages(q, system_prompt=prompts[0])
    outs, preds = [], []
    if initial_answers is not None:
        if len(initial_answers) != k:
            raise ValueError("initial_answers must contain exactly k responses")
        outs = list(initial_answers)
        preds = [task.parse(out) for out in outs]
    else:
        for agent_index in range(k):
            out = strip_think(
                chat(
                    model,
                    tok,
                    build_messages(q, system_prompt=prompts[agent_index]),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
            )
            outs.append(out)
            preds.append(task.parse(out))
    return {
        "pred": _majority(preds),
        "raw": outs,
        "preds": preds,
        "prompt": q,
        "messages": messages,
    }


def generate_initial_answers(
    model,
    tok,
    task,
    item,
    sysp,
    max_new_tokens,
    n_agents,
    temperature=0.7,
    enable_thinking=None,
    initial_style="cot",
):
    """Generate one reusable Round 0 population for paired comparisons."""
    result = run_majority(
        model, tok, task, item, sysp, max_new_tokens, k=n_agents,
        temperature=temperature, enable_thinking=enable_thinking,
        initial_style=initial_style,
    )
    return {
        "raw": result["raw"], "preds": result["preds"],
        "prompt": result["prompt"], "messages": result["messages"],
    }


def _result_from_trace(task, question, trace):
    final = trace["answers_by_round"][-1]
    preds = [task.parse(answer) for answer in final]
    return {
        "pred": _majority(preds),
        "raw": final,
        "preds": preds,
        "prompt": question,
        "debate_trace": trace,
    }


def run_debate(
    model,
    tok,
    task,
    item,
    sysp,
    max_new_tokens,
    n_agents=3,
    n_rounds=2,
    temperature=0.7,
    enable_thinking=None,
    initial_answers=None,
    initial_style="cot",
    peer_mode="peers",
):
    question = task.question(item, style=initial_style)
    paper_template = getattr(task, "paper_debate_template", None)
    template = (
        paper_template
        if initial_style == "paper" and paper_template
        else (task.debate_template or DEFAULT_DEBATE_TEMPLATE)
    )
    trace = _debate_engine(
        model,
        tok,
        question,
        debate_template=template,
        system_prompt=sysp,
        n_agents=n_agents,
        n_rounds=n_rounds,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        prompt_profile=getattr(task, "prompt_profile_name", None),
        initial_answers=initial_answers,
        other_answers_formatter=(
            format_paper_others
            if initial_style == "paper"
            else getattr(task, "format_other_answers", None)
        ),
        peer_mode=peer_mode,
        self_refine_template=getattr(task, "self_refine_template", None),
    )
    return _result_from_trace(task, question, trace)


def run_debate_memory(
    model,
    tok,
    task,
    item,
    sysp,
    max_new_tokens,
    n_agents=5,
    n_rounds=2,
    temperature=0.7,
    enable_thinking=None,
    memory_max_new_tokens=256,
    memory_temperature=0.0,
    initial_answers=None,
):
    question = task.question(item, style="cot")
    template = task.debate_template or DEFAULT_DEBATE_TEMPLATE
    trace = _debate_engine(
        model,
        tok,
        question,
        debate_template=template,
        system_prompt=sysp,
        n_agents=n_agents,
        n_rounds=n_rounds,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        use_memory=True,
        memory_context=(task.memory_context(item) if hasattr(task, "memory_context") else None),
        prompt_profile=getattr(task, "prompt_profile_name", None),
        initial_answers=initial_answers,
        memory_summary_template=(getattr(task, "memory_summary_template", None) or DEFAULT_MEMORY_SUMMARY_TEMPLATE),
        memory_debate_template=(getattr(task, "memory_debate_template", None) or DEFAULT_MEMORY_DEBATE_TEMPLATE),
        memory_max_new_tokens=memory_max_new_tokens,
        memory_temperature=memory_temperature,
    )
    return _result_from_trace(task, question, trace)

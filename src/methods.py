"""비교 대상 method 들. 모두 (pred, raw ...) dict 를 돌려준다.

논문 정렬: 모든 method 가 **동일한 temperature** 를 쓴다(논문은 single·debate 모두 같은 온도).
- vanilla : 직답 1회
- cot     : 단계별 추론 1회
- majority: cot 를 k회 샘플링 후 다수결 (self-consistency)
- debate  : Multi-Agent Debate 후 마지막 라운드 다수결
"""
from collections import Counter

from .debate import DEFAULT_DEBATE_TEMPLATE
from .debate import run_debate as _debate_engine
from .llm import build_messages, chat, strip_think


def _majority(preds):
    valid = [p for p in preds if p is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def _one(model, tok, task, item, sysp, max_new_tokens, temperature, style, enable_thinking):
    q = task.question(item, style=style)
    out = strip_think(chat(model, tok, build_messages(q, system_prompt=sysp),
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           enable_thinking=enable_thinking))
    return {"pred": task.parse(out), "raw": [out]}


def run_vanilla(model, tok, task, item, sysp, max_new_tokens, temperature=0.7, enable_thinking=None):
    return _one(model, tok, task, item, sysp, max_new_tokens, temperature, "vanilla", enable_thinking)


def run_cot(model, tok, task, item, sysp, max_new_tokens, temperature=0.7, enable_thinking=None):
    return _one(model, tok, task, item, sysp, max_new_tokens, temperature, "cot", enable_thinking)


def run_majority(model, tok, task, item, sysp, max_new_tokens, k=5, temperature=0.7, enable_thinking=None):
    q = task.question(item, style="cot")
    outs, preds = [], []
    for _ in range(k):
        o = strip_think(chat(model, tok, build_messages(q, system_prompt=sysp),
                             max_new_tokens=max_new_tokens, temperature=temperature,
                             enable_thinking=enable_thinking))
        outs.append(o)
        preds.append(task.parse(o))
    return {"pred": _majority(preds), "raw": outs, "preds": preds}


def run_debate(model, tok, task, item, sysp, max_new_tokens, n_agents=3, n_rounds=2,
               temperature=0.7, debate_prompt="paper", enable_thinking=None):
    q = task.question(item, style="cot")
    if debate_prompt == "critical":
        template = (getattr(task, "debate_template_critical", None)
                    or task.debate_template or DEFAULT_DEBATE_TEMPLATE)
    else:
        template = task.debate_template or DEFAULT_DEBATE_TEMPLATE
    trace = _debate_engine(model, tok, q, debate_template=template, system_prompt=sysp,
                           n_agents=n_agents, n_rounds=n_rounds, max_new_tokens=max_new_tokens,
                           temperature=temperature, enable_thinking=enable_thinking)
    final = trace["answers_by_round"][-1]
    preds = [task.parse(a) for a in final]
    return {"pred": _majority(preds), "raw": final, "preds": preds}

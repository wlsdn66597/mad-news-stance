"""Multi-Agent Debate 최소 구현 (Du et al., 2023 구조 재현).

라운드 0: 각 에이전트 독립 답변. 라운드 r>0: 다른 에이전트들의 직전 답변을 주고 갱신.
debate 프롬프트(debate_template)는 task 가 언어/포맷에 맞게 제공한다(영어=MMLU/GSM8K, 한국어=stance).
"""
from .llm import build_messages, chat, strip_think

# task 가 debate_template 을 안 주면 쓰는 기본값(한국어 generic).
DEFAULT_DEBATE_TEMPLATE = (
    "다음은 다른 에이전트들이 같은 문제에 대해 내놓은 답변입니다:\n\n{others}\n\n"
    "다른 에이전트들의 추론을 참고하여 당신의 답변을 다시 검토하고, "
    "갱신된 답변을 제시하세요. 마지막 줄에 최종 답을 명확히 쓰세요."
)


def format_others(other_answers):
    return "\n\n".join(f"[Agent {i + 1}]\n{ans}" for i, ans in enumerate(other_answers))


def run_debate(model, tokenizer, question, debate_template=DEFAULT_DEBATE_TEMPLATE,
               system_prompt=None, n_agents=3, n_rounds=2, max_new_tokens=512,
               temperature=0.7, enable_thinking=None):
    agent_contexts = [
        build_messages(question, system_prompt=system_prompt) for _ in range(n_agents)
    ]
    answers_by_round = []

    for r in range(n_rounds):
        round_answers = []
        for a in range(n_agents):
            if r > 0:
                others = [answers_by_round[r - 1][b] for b in range(n_agents) if b != a]
                agent_contexts[a].append(
                    {"role": "user", "content": debate_template.format(others=format_others(others))}
                )
            reply = chat(model, tokenizer, agent_contexts[a], max_new_tokens=max_new_tokens,
                         temperature=temperature, enable_thinking=enable_thinking)
            reply = strip_think(reply)
            agent_contexts[a].append({"role": "assistant", "content": reply})
            round_answers.append(reply)
        answers_by_round.append(round_answers)

    return {
        "question": question,
        "debate_template": debate_template,
        "system_prompt": system_prompt,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        },
        "answers_by_round": answers_by_round,
        "agent_contexts": agent_contexts,
    }

"""Multi-Agent Debate 최소 구현 (Du et al., 2023 구조 재현).

라운드 0: 각 에이전트가 독립적으로 답변.
라운드 r>0: 각 에이전트에게 '다른 에이전트들의 직전 답변'을 주고 답변을 갱신.
task 를 모르는 범용 루프 — Phase 1(MMLU/GSM8K), Phase 2(stance) 에서 재사용한다.
"""
from .llm import build_messages, chat, strip_think

DEBATE_INSTRUCTION = (
    "다음은 다른 에이전트들이 같은 문제에 대해 내놓은 답변입니다:\n\n"
    "{others}\n\n"
    "다른 에이전트들의 추론을 참고하여 당신의 답변을 다시 검토하고, "
    "갱신된 답변을 제시하세요. 마지막 줄에 최종 답을 명확히 쓰세요."
)


def format_others(other_answers):
    return "\n\n".join(
        f"[에이전트 {i + 1}]\n{ans}" for i, ans in enumerate(other_answers)
    )


def run_debate(model, tokenizer, question, system_prompt=None, n_agents=3,
               n_rounds=2, max_new_tokens=256, temperature=0.7, enable_thinking=None):
    """debate 를 수행하고 전체 trace(dict)를 반환한다.

    반환:
      question, n_agents, n_rounds,
      answers_by_round[r][a]  : 각 라운드/에이전트의 답변(‹think› 제거됨),
      agent_contexts[a]       : 에이전트별 전체 대화 메시지(재현/디버깅용).
    """
    agent_contexts = [
        build_messages(question, system_prompt=system_prompt) for _ in range(n_agents)
    ]
    answers_by_round = []

    for r in range(n_rounds):
        round_answers = []
        for a in range(n_agents):
            if r > 0:
                # 다른 에이전트들의 직전 라운드 답변을 user 턴으로 주입
                others = [answers_by_round[r - 1][b] for b in range(n_agents) if b != a]
                agent_contexts[a].append(
                    {"role": "user", "content": DEBATE_INSTRUCTION.format(others=format_others(others))}
                )

            reply = chat(
                model, tokenizer, agent_contexts[a],
                max_new_tokens=max_new_tokens, temperature=temperature,
                enable_thinking=enable_thinking,
            )
            reply = strip_think(reply)  # 다음 라운드/다른 에이전트에 reasoning 오염 방지
            agent_contexts[a].append({"role": "assistant", "content": reply})
            round_answers.append(reply)

        answers_by_round.append(round_answers)

    return {
        "question": question,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "answers_by_round": answers_by_round,
        "agent_contexts": agent_contexts,
    }

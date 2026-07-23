# Phase 1 paper comparison methods

이 저장소의 `--methods paper`는 Du et al., *Improving Factuality and
Reasoning in Language Models through Multiagent Debate*의 표와 공개 코드를
기준으로 task별 비교군을 선택한다.

| Task | Methods |
|---|---|
| GSM8K | `single`, `reflection`, `majority`, `debate` |
| MMLU | `single`, `reflection`, `debate` |

## Method mapping

- `single`: 공개 코드의 최초 reasoning prompt로 응답을 한 번 생성한다.
- `reflection`: 최초 응답을 같은 대화 문맥에 넣고, 공개 코드의
  `Can you double check...` self-check prompt로 두 번째 응답을 생성한다.
  최종 두 번째 응답을 채점한다.
- `majority`: 세 에이전트가 독립적으로 생성한 최초 응답을 다수결한다.
  논문 Table 1의 GSM8K 비교군에만 기본 포함한다.
- `debate`: 세 에이전트가 독립적으로 최초 응답을 만든 뒤 다른 두
  에이전트의 응답을 보고 한 번 갱신한다(3 agents, 2 total rounds).

기존 `vanilla`와 `cot`는 과거 결과 호환을 위해 유지한다. 특히 기존
`cot`는 단일 호출의 chain-of-thought prompt이며, 두 번 호출하는 논문
`Single Agent (Reflection)`과 같지 않다.

## Reproduction details

`single`, `reflection`, `majority`, `debate`는 같은 task별 최초
prompt를 사용한다. GSM8K의 boxed-answer prompt와 MMLU의 `(X)` answer
prompt, reflection prompt, 다른 agent 응답 연결 형식은 공개 코드 문구를
사용한다. 결과 JSON에는 reflection의 최초/최종 응답과 prompt,
debate의 전체 trace가 저장된다.

Sources:

- Paper: https://arxiv.org/abs/2305.14325
- Official code: https://github.com/composable-models/llm_multiagent_debate
- GSM8K implementation: https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/gen_gsm.py
- MMLU implementation: https://github.com/composable-models/llm_multiagent_debate/blob/main/mmlu/gen_mmlu.py

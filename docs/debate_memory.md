# Agent scaling with summarization-memory

## 근거와 범위

Du et al., *Improving Factuality and Reasoning in Language Models through
Multiagent Debate*는 본 실험에서 주로 3 agents x 2 rounds를 사용했다.
논문 Section 3.3은 agent 수가 증가하면 다른 agent 응답을 직접 이어 붙일 때
context length 문제가 발생하므로, 5명 이상에서는 응답을 먼저 요약해서
전달했다고 설명한다. Figure 13은 이 요약 방식이 직접 연결보다 성능도
개선할 수 있다고 보고한다. Figure 13은 1-5 agents에서 concat과 summarize를 비교하고, Appendix Figure 25는 4-agent arithmetic 예시의 instruction에 요약 응답을 전달한다.

- Paper: https://arxiv.org/abs/2305.14325
- Public code: https://github.com/composable-models/llm_multiagent_debate

공개 코드는 3 agents x 2 rounds의 direct concatenation만 포함하며, 논문의
summarization 구현은 공개되어 있지 않다. 이 저장소의 debate_memory는 논문에서
설명한 동작을 재현한 로컬 오픈웨이트 adaptation이다.

## 구현

debate_memory는 각 갱신 round 직전에 shared memory agent를 한 번 호출한다.

1. N개 agent의 직전 답변을 agent label과 함께 수집한다.
2. memory 호출이 후보 정답, 근거, 반박, 미해결 쟁점을 압축한다.
3. 모든 agent가 같은 shared summary를 읽고 자기 답을 갱신한다.
4. 최종 답은 기존 debate와 동일하게 마지막 round agent 예측의 다수결이다.

memory prompt는 다수 의견에 무조건 동조하지 말고 구체적인 근거와 반박을
보존하도록 요구한다. 최종 raw 답변, memory 입력/출력, agent context와
communication 통계는 모두 debate_trace에 저장된다.

## 호출 수

2 rounds 기준 문항당 생성 호출 수:

| Method | N agents일 때 호출 수 |
|---|---:|
| majority | N |
| debate | 2N |
| debate_memory | 2N + 1 |

shared summary는 agent마다 따로 만들지 않고 round마다 한 번 생성한다. 따라서
direct concatenation의 agent별 context 증가를 줄이면서 추가 호출을 최소화한다.

## 저장 지표

communication_stats_by_round에는 다음이 기록된다.

- direct 방식이었다면 각 agent가 받았을 prompt token 수
- 실제 전달된 direct 또는 summary prompt token 수
- memory summarizer 입력 token 수
- agent별 peak context 압축 비율
- summarizer까지 포함한 전체 입력 token 비율

정확도뿐 아니라 context 감소와 추가 계산량도 함께 비교할 수 있다.

## GPU 없는 검증

    python -m py_compile src/debate.py src/methods.py scripts/run_phase1.py scripts/run_phase1_repeated.py scripts/run_agent_scaling.py
    python scripts/run_agent_scaling.py --config config/phase1.yaml --tasks gsm8k,mmlu --model qwen --agent-counts 3,5,7 --n 10 --dry-run

## RTX 4090: Qwen3-8B smoke

서버에서 만들어 둔 config/phase1_qwen8b.yaml을 그대로 사용한다.

    python scripts/run_agent_scaling.py \
      --config config/phase1_qwen8b.yaml \
      --tasks gsm8k,mmlu \
      --model qwen \
      --methods majority,debate,debate_memory \
      --agent-counts 3,5,7 \
      --n-rounds 2 \
      --n 3 \
      --data-seed 0 \
      --run-seed 2000 \
      --temperature 1.0 \
      --memory-max-new-tokens 256 \
      --memory-temperature 0.0 \
      --tag-prefix qwen8b_agent_scaling_memory_smoke

## RTX 4090: 탐색 실험

smoke가 끝나면 먼저 n=10으로 trace와 시간, 압축률을 확인한다.

    nohup python scripts/run_agent_scaling.py \
      --config config/phase1_qwen8b.yaml \
      --tasks gsm8k,mmlu \
      --model qwen \
      --methods majority,debate,debate_memory \
      --agent-counts 3,5,7 \
      --n-rounds 2 \
      --n 10 \
      --data-seed 0 \
      --run-seed 2000 \
      --temperature 1.0 \
      --memory-max-new-tokens 256 \
      --memory-temperature 0.0 \
      --tag-prefix qwen8b_agent_scaling_memory_n10 \
      > phase1_agent_scaling_memory.log 2>&1 &

진행 확인:

    tail -f phase1_agent_scaling_memory.log

완료 후 요약:

    cat results/phase1/agent_scaling_summary_gsm8k-mmlu_qwen_n10_rounds2_qwen8b_agent_scaling_memory_n10.md

같은 명령을 다시 실행하면 완료된 문항은 건너뛴다. 실제 agent 수 효과를
판단할 때는 정확도, debate_memory - debate, peak context token 비율,
호출 수와 실행시간을 함께 보고한다.

# Phase 1 반복실험

## 실험 원칙

- `data_seed=0`을 모든 반복에서 고정해 같은 GSM8K/MMLU 문항을 평가한다.
- `run_seed`만 반복마다 변경해 LLM 샘플링 변동을 측정한다.
- 각 task/model/method의 반복별 accuracy를 먼저 계산한 뒤 다음 값을 보고한다.
  - mean accuracy
  - sample standard deviation (`ddof=1`)
  - standard error: `sample_std / sqrt(completed_runs)`
- 기본 반복 수는 3회다. stderr는 5회 이상보다 불안정할 수 있으므로 해석 시 함께 명시한다.

## 빠른 확인

실제 장시간 실행 전에 Qwen, GSM8K 5문항으로 명령과 저장 경로를 확인한다.

```bash
python scripts/run_phase1_repeated.py \
  --tasks gsm8k \
  --models qwen \
  --methods vanilla,debate \
  --n 5 \
  --repeats 2 \
  --tag-prefix smoke_repeat
```

## 본실험

두 task와 두 모델을 각각 100문항, 3회 반복한다. 기존
`scripts/run_n100.sh`의 faithful 설정과 맞추기 위해 temperature는 1.0으로 지정한다.
GPU 메모리 충돌을 피하도록 Qwen 완료 후 EXAONE을 순차 실행한다.

```bash
nohup bash scripts/run_repeated_n100.sh qwen > phase1_repeat_qwen.log 2>&1 &
nohup bash scripts/run_repeated_n100.sh exaone > phase1_repeat_exaone.log 2>&1 &
```

진행 상황은 다음 명령으로 확인한다.

```bash
tail -f phase1_repeat_qwen.log
tail -f phase1_repeat_exaone.log
```

프로세스가 중단되면 같은 본실험 명령을 다시 실행한다. `run_phase1.py`가 이미 저장된 문항을
건너뛰며, 문항별 generation seed가 고정되어 있어 재개 위치가 달라도 이후 생성 seed는 바뀌지 않는다.

한 모델씩 나누어 실행하려면 `--models qwen` 또는 `--models exaone`을 사용한다. 기존 결과와
파일명을 구분하려면 실행 설정마다 서로 다른 `--tag-prefix`를 사용한다.

## 저장되는 입력과 debate trace

정성분석과 실행 재현을 위해 각 문항 결과에 다음 입력 정보를 함께 저장한다.

- 모든 방법: `prompt`, `messages`
- Debate: 기존 마지막 round의 `raw`, `preds`와 함께 `debate_trace` 저장
- `debate_trace`: `answers_by_round`, agent별 전체 `agent_contexts`, debate template,
  system prompt, agent/round 수, generation 설정

기존 `pred`, `raw`, `preds`, `gold`, `correct`, `generation_seed` 필드는 유지하므로
기존 결과 분석 코드와 호환된다.

## 결과 파일

반복별 raw 결과:

```text
results/phase1/gsm8k_qwen_n100_faithful_repeat_trace_r01_s1000.json
results/phase1/gsm8k_qwen_n100_faithful_repeat_trace_r02_s1001.json
results/phase1/gsm8k_qwen_n100_faithful_repeat_trace_r03_s1002.json
```

자동 집계 결과:

```text
results/phase1/repeat_summary_gsm8k-mmlu_qwen-exaone_n100_r3_faithful_repeat_trace.csv
results/phase1/repeat_summary_gsm8k-mmlu_qwen-exaone_n100_r3_faithful_repeat_trace.json
results/phase1/repeat_summary_gsm8k-mmlu_qwen-exaone_n100_r3_faithful_repeat_trace.md
```

실험은 끝났지만 집계 파일만 다시 만들고 싶다면 다음과 같이 실행한다.

```bash
python scripts/run_phase1_repeated.py \
  --tasks gsm8k,mmlu \
  --models qwen,exaone \
  --n 100 \
  --repeats 3 \
  --data-seed 0 \
  --base-run-seed 1000 \
  --temperature 1.0 \
  --tag-prefix faithful_repeat_trace \
  --summarize-only
```

`completed_runs`가 `5/5`가 아닌 행은 누락되거나 아직 끝나지 않은 반복이 있다는 뜻이다.
불완전한 반복은 mean과 stderr 계산에서 제외된다.

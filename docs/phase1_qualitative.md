# Qwen3-14B trace 정성분석

14B의 GSM8K/MMLU 각 50문항에서 Majority→Debate 전환을 집계하고,
수동 검토용 50문항(GSM8K 20 + MMLU 30)을 만든다. GPU는 사용하지 않는다.

## 실행

```bash
python scripts/analyze_phase1_qualitative.py
```

결과:

```text
results/phase1/qualitative_14b/summary.md
results/phase1/qualitative_14b/gsm8k_transition_index.csv
results/phase1/qualitative_14b/mmlu_transition_index.csv
results/phase1/qualitative_14b/qualitative_sample_50.csv
results/phase1/qualitative_14b/qualitative_sample_50_ids.txt
```

특정 문항의 모든 prompt/trace 확인:

```bash
python scripts/analyze_phase1_qualitative.py --case mmlu:문항ID | less
python scripts/analyze_phase1_qualitative.py --case gsm8k:문항ID | less
```

## 해석

- GSM8K: `corrected=degraded=0`이면 같은 46문항 정답이므로 천장 효과와 양립한다.
  둘이 같은 양수라면 교정과 퇴행이 상쇄된 동률이다.
- MMLU: `corrected-degraded`가 순향상 문항 수다. 교정 사례에서 round 0의
  정답 소수 agent가 근거를 통해 다수를 바꿨는지 확인한다.
- 수동 코딩: 초기 정답 분포, 답변 변경 방향, 구체적 근거, 단순 동조,
  정답/오답 수렴을 기록한다.

# 실험 기록 (Experiment Log)

## 공통 setup
- 모델: `Qwen/Qwen3-4B` (fp16, non-thinking)  ※ EXAONE는 transformers 버전 이슈로 보류
- GPU: RTX 4090 24GB / CUDA 12.2
- transformers: `<채우기: python -c "import transformers; print(transformers.__version__)">`
- seed: 0 (config)
- 코드/설정 버전은 각 실험의 **git commit** 으로 고정된다.

## 기록 규칙 (실험 1건 = 아래 한 블록)
- 날짜 / git commit / 실행 커맨드
- config 요약 (n, split, max_new_tokens, temperature, debate n_agents×n_rounds, majority k)
- 결과 표 (method별 accuracy; stance는 + macro-F1, confusion matrix)
- 관찰 & 해석
- **바꾼 것 + 이유** (다음 실험으로 연결)
- 원칙: **한 번에 변수 하나만** 바꾼다. 그래야 개선 원인을 특정할 수 있다.

---

## [E1] GSM8K 재현 (n=50)
- date: 2026-07-06 | commit: 896fdb0 | `python scripts/run_phase1.py --task gsm8k --n 50`
- config: n=50, split=test, max_new_tokens=512, temp=0.7, debate 3×2, majority k=5

| method | acc |
|---|---|
| vanilla | 0.900 |
| cot | 0.880 |
| majority | 0.840 |
| debate | 0.880 |

- 관찰: debate가 vanilla를 못 이김. 방법 간 차이 1~3문제(노이즈 수준), majority만 약간 낮음.
- 해석: Qwen3-4B가 GSM8K를 이미 잘함(near-ceiling) + vanilla/cot는 greedy, majority/debate는 sampling(불리한 출발). **강한 모델·쉬운 과제라 debate 이득 없음(예상됨).**

## [E2] MMLU 재현 (n=50)
- date: 2026-07-06 | commit: 896fdb0 | `python scripts/run_phase1.py --task mmlu --n 50`
- config: 위와 동일 (max_new_tokens=512)

| method | acc |
|---|---|
| vanilla | 0.860 |
| cot | 0.560 |
| majority | 0.600 |
| debate | 0.660 |

- 관찰: cot가 vanilla보다 30%p 급락(0.86→0.56). 비정상적으로 큼. cot는 13.8s/item(≈512토큰 꽉 참), vanilla는 4.4s/item.
- 해석(가설): **max_new_tokens=512 truncation.** CoT 추론이 길어 마지막 `정답:` 줄 전에 잘림 → 파서가 추론 중간의 엉뚱한 글자를 집음. vanilla는 짧아 무사. → 방법 비교가 무효.
- 검증: `results` json에서 `정답` 마커 없는 cot 항목 수 확인.
- **바꾼 것 → [E3]: max_new_tokens 512→1024** (한 변수만). 캐시 파일은 `mmlu_qwen_n50_maxtok512.json`으로 보존 후 재실행.

## [E3] MMLU 재현 재실행 (max_new_tokens=1024)  ← 진행 예정
- date: | commit: | `python scripts/run_phase1.py --task mmlu --n 50`
- config: 위와 동일하되 **max_new_tokens=1024**

| method | acc |
|---|---|
| vanilla |  |
| cot |  |
| majority |  |
| debate |  |

- 관찰:
- 해석: (cot가 회복되면 E2의 truncation 가설 확정)
- 바꾼 것 →:

---

## (예정) Phase 2 — K-News-Stance
- `python scripts/run_phase2.py --n 30` → method별 accuracy / macro-F1 / confusion matrix 기록.
- 분석: neutral 혼동, 인용문 오독 대표 사례.

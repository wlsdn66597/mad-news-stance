# mad-news-stance

**Multi-Agent Debate (MAD) 재현 → 한국어 뉴스 입장 탐지(K-News-Stance) 적용**

- 논문: *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (Du et al., ICML 2024)
- 참고 구현: https://github.com/composable-models/llm_multiagent_debate
- 사용 모델(로컬 오픈웨이트): `LGAI-EXAONE/EXAONE-4-2B`, `Qwen/Qwen3-4B`

## 레포 구조

```
mad-news-stance/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── smoke.yaml          # 모델 id, 4-bit, mem 상한, debate 파라미터, 예시 질문
├── src/
│   ├── llm.py              # 4-bit 로딩 + chat() (모델별 chat template 처리)
│   └── debate.py           # task 무관 debate 루프 (라운드0 독립 → 라운드r 상호검토)
├── scripts/
│   └── smoke_test.py       # 로딩 → 단일 답변 → 1문항 debate trace 저장
├── data/                   # (gitignore) 데이터셋은 원격에 별도 배치
└── results/                # (gitignore) 생성 결과/trace
```


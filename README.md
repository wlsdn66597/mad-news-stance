# mad-news-stance

**Multi-Agent Debate (MAD) 재현 → 한국어 뉴스 입장 탐지(K-News-Stance) 적용** 프로젝트.

- 논문: *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (Du et al., ICML 2024)
- 참고 구현: https://github.com/composable-models/llm_multiagent_debate
- 사용 모델(로컬 오픈웨이트): `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`, `Qwen/Qwen3-4B`

> **현재 단계: Phase 0 — 스모크 테스트.**
> 성능 측정이 아니라 "파이프라인이 도는지"만 확인한다:
> ① 4-bit 로딩 OK  ② 모델 생성 OK  ③ debate(에이전트×라운드) trace 저장 OK.

---

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

모델 가중치·데이터셋·결과물은 **Git 에 올리지 않는다**(용량). 원격 서버에서 받아서 쓴다.

---

## 원격 GPU 서버에서 실행하는 법

> 환경: Ubuntu / RTX 4090 24GB / CUDA 12.2 / 공용 `user` 계정 가정.
> **공용 GPU 주의:** 다른 사람의 학습 잡이 VRAM 을 점유 중일 수 있다. 무거운 실험은 GPU 가 빌 때 돌리고,
> 스모크는 4-bit + 메모리 상한(`mem_fraction`)으로 작게 돌린다.

### 0) (최초 1회) 코드 받기 — git clone

```bash
cd ~/proj                       # 자기 작업 폴더
git clone https://github.com/wlsdn66597/mad-news-stance.git
cd mad-news-stance
```

이후 코드 수정분 반영은 **git pull**:

```bash
cd ~/proj/mad-news-stance
git pull
```

### 1) 파이썬 환경 + 라이브러리

```bash
python -m venv .venv
source .venv/bin/activate

# torch 는 CUDA 에 맞춰 먼저 설치 (CUDA 12.x → cu121 휠, 드라이버 535 와 호환)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2) HuggingFace 캐시 경로 지정 (공용 계정이라 내 경로로 격리)

```bash
export HF_HOME=~/hf_cache_jinwoo          # .bashrc 에도 추가해두면 편함
export HF_HUB_ENABLE_HF_TRANSFER=1        # (선택) 다운로드 가속
```

### 3) 모델 다운로드

`smoke_test.py` 첫 실행 시 **자동으로 다운로드**되어 `HF_HOME` 에 캐시된다(한 번만).
미리 받아두고 싶으면:

```bash
huggingface-cli download LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
huggingface-cli download Qwen/Qwen3-4B
```

- 대략 EXAONE ~5GB, Qwen3-4B ~8GB. 네트워크에 따라 수 분~20분.
- 비공개 게이팅이 걸려 있으면 `huggingface-cli login` 후 재시도.

### 4) 스모크 테스트 실행

```bash
# GPU 파편화 완화(권장)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EXAONE-2.4B (기본) — 한국어 확인
python scripts/smoke_test.py

# Qwen3-4B — thinking 모드 on/off 확인
python scripts/smoke_test.py --model qwen              # thinking ON
python scripts/smoke_test.py --model qwen --no-think   # thinking OFF
```

정상이면 콘솔에 단일 답변 + 라운드별 각 에이전트 답변이 찍히고
`results/smoke/trace_*.json` 이 저장된다. (3-4-5 삼각형 넓이 = 6 으로 수렴하면 성공)

### 5) 공용 GPU 여유가 없을 때 (예약 실행)

다른 잡이 GPU 를 물고 있으면, 여유 VRAM 이 생길 때 자동 실행하도록 걸어둔다:

```bash
cat > ~/gpu_wait_run.sh << 'EOF'
#!/usr/bin/env bash
set -u
GPU=0; NEED_MB=20000
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  echo "[$(date)] free=${FREE}MB"; [ "$FREE" -ge "$NEED_MB" ] && break; sleep 60
done
sleep 20
cd ~/proj/mad-news-stance && source .venv/bin/activate
python scripts/smoke_test.py            # ← 원하는 실제 커맨드로 교체
EOF
chmod +x ~/gpu_wait_run.sh
nohup ~/gpu_wait_run.sh > ~/gpu_wait_run.log 2>&1 &   # 로그아웃해도 유지
tail -f ~/gpu_wait_run.log
```

---

## 데이터셋 (Phase 2 에서 사용)

`K-News-Stance` (2,000 기사 / 47 이슈 / supportive·oppositional·neutral) JSON 은
용량·라이선스 때문에 Git 에 넣지 않는다. 노트북에서 서버로 한 번 복사:

```bash
scp k-news-stance_nosegment.json user@<서버>:~/proj/mad-news-stance/data/
```

GSM8K / MMLU (Phase 1 재현용)는 코드에서 `datasets` 로 자동 다운로드 예정.

---

## 로드맵

- **Phase 0 (현재):** 스모크 테스트 — 파이프라인 점검.
- **Phase 1:** 논문 재현 — MMLU / GSM8K 소량 서브셋에서 `single / CoT / majority / debate` 비교.
- **Phase 2:** K-News-Stance 적용 — 단일 LLM vs MAD (Accuracy / Macro-F1 / confusion matrix).
- **Phase 3 (확장):** 역할기반 에이전트(Evidence/Stance/Critic/Judge) + memory agent.

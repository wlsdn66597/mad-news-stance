#!/usr/bin/env bash
# The qualitative half of the results section, appended to the same file the
# tables went into. Offline.
#
#   bash scripts/report_siw_tables.sh && bash scripts/report_siw_cases.sh
#   CASES=8 CHARS=1200 bash scripts/report_siw_cases.sh
#
# The tables say the exchange gains items. This says what a peer wrote that
# changed a mind, and whether the passage it quoted is in the article at all.
set -uo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-6000}"
MODEL="${MODEL:-exaone}"
SPLIT="${SPLIT:-test}"
N="${N:-1001}"
PROFILE="${PROFILE:-stance_minimal_en}"
DATA_SEED="${DATA_SEED:-0}"
MIX="${MIX:-sourcing-issue-wording}"
DATA="${DATA:-data/k-news-stance_nosegment.json}"
CASES="${CASES:-5}"
CHARS="${CHARS:-900}"

B="results/phase2/stance_${MODEL}_${SPLIT}_n${N}_${PROFILE}_d${DATA_SEED}_s${SEED}_a3"
# the historical trio carries no mix tag, so its five conditions live under
# different names; everything downstream is identical
if [ "$MIX" = "mixed" ] || [ "$MIX" = "fsw" ]; then
  MAJ="${B}_r1_personas.json"
  SELF="${B}_r4_personas_selfrefine-full.json"
  FULL="${B}_r4_personas_reasoned_exchange_full.json"
else
  MAJ="${B}_r1_personas_mix-${MIX}.json"
  SELF="${B}_r4_personas_selfrefine-full_mix-${MIX}.json"
  FULL="${B}_r4_personas_mix-${MIX}_reasoned_exchange_full.json"
fi
if [ "$MIX" = "mixed" ] || [ "$MIX" = "fsw" ]; then
  JUDGE_R0="results/judge_round0_only"
else
  JUDGE_R0="results/judge_mix_${MIX}"
fi

OUT="docs/results_${MODEL}_s${SEED}_${MIX}.md"
mkdir -p docs
[ -f "$OUT" ] || : > "$OUT"

say () { echo "$@" | tee -a "$OUT"; }
run () {
  say ""
  say "### $1"
  say ""
  say '```'
  "${@:2}" 2>&1 | tee -a "$OUT"
  say '```'
}

say ""
say "---"
say ""
say "# 정성 분석"
say ""
say "생성일 $(date '+%Y-%m-%d %H:%M'). \`o\` = 인용이 기사에 있음, \`x\` = 없음, \`-\` = 판단 불가."

# --- grounding ---------------------------------------------------------------
say ""
say "## 인용 접지 — 근거가 기사에 실제로 존재하는가"
say ""
say "advocacy 진단에서는 한국어 인용의 81%가 기사에 없었다. 근거가 허구라면"
say "동료가 무엇으로 설득하고 있는지가 달라지므로, 이 비율이 교환 이득의 상한을 정한다."
for f in "$FULL" "$SELF"; do
  [ -f "$f" ] || { say ""; say "**없음** — \`$f\`"; continue; }
  run "접지율 · $(basename "$f" | sed 's/.*_personas//;s/\.json//')" \
    python scripts/inspect_cases.py --after "$f" --data-path "$DATA" --no-cases
done

# --- buckets -----------------------------------------------------------------
if [ -f "$FULL" ] && [ -f "$MAJ" ]; then
  say ""
  say "## 사례 — 교환이 각 문항에 무엇을 했는가"
  say ""
  say "| 버킷 | 논문에서의 역할 |"
  say "|---|---|"
  say "| wrong_to_correct | H2의 서술 근거 — 동료가 무슨 근거를 줘서 바뀌었는지 |"
  say "| correct_to_wrong | 교환의 비용, conformity 사례 |"
  say "| consensus_error | 셋이 합의하고 틀린 경우 — H4 |"
  say "| neutral_missed | 근거 형식이 neutral을 억제하는 사례 |"

  for bucket in wrong_to_correct correct_to_wrong consensus_error neutral_missed; do
    run "사례 · $bucket (peer exchange)" \
      python scripts/inspect_cases.py --after "$FULL" --before "$MAJ" \
        --data-path "$DATA" --bucket "$bucket" -n "$CASES" --chars "$CHARS"
  done
fi

# --- peers vs rereading ------------------------------------------------------
if [ -f "$SELF" ] && [ -f "$MAJ" ]; then
  say ""
  say "## 동료 vs 자기 재검토 — 같은 형식, 다른 상대"
  say ""
  say "self-refine이 깨뜨린 문항을 위 wrong_to_correct와 나란히 읽으면, 동료가 하는 일이"
  say "숫자가 아니라 문장으로 설명된다."
  for bucket in correct_to_wrong wrong_to_correct; do
    run "사례 · $bucket (self-refine)" \
      python scripts/inspect_cases.py --after "$SELF" --before "$MAJ" \
        --data-path "$DATA" --bucket "$bucket" -n "$CASES" --chars "$CHARS"
  done
fi

# --- judge -------------------------------------------------------------------
say ""
say "## Judge가 실제로 무엇을 근거로 내세우는가"
say ""
say "이전 확인에서 judge가 \`short article-grounded evidence\`처럼 지시문을 되뇌는 사례가"
say "있었다. 빈도가 높으면 §5에 각주가 필요하다."
say ""
say '```'
JUDGE_DIR="$JUDGE_R0" CASES="$CASES" python - 2>&1 <<'PY' | tee -a "$OUT"
import glob, json, os, random, re
d = os.environ["JUDGE_DIR"]
files = sorted(glob.glob(os.path.join(d, "*.items.json")))
if not files:
    print(f"{d}: 없음")
else:
    rows = [r for r in json.load(open(files[0], encoding="utf-8")) if r.get("triggered")]
    quoted = sum(1 for r in rows if re.search(r'["“‘].{6,}["”’]',
                                              str(r.get("judge_raw_output") or "")))
    print(f"{os.path.basename(d)}  triggered={len(rows)}  "
          f"인용부호가 있는 근거를 낸 비율={quoted / max(1, len(rows)):.1%}")
    for r in random.Random(0).sample(rows, min(int(os.environ["CASES"]), len(rows))):
        print(f"\n=== item {r['item_id']}  gold={r['gold']}  r0={r.get('round0_agent_preds')}"
              f"  judge={r.get('judge_prediction')}")
        print(str(r.get("judge_raw_output"))[:500])
PY
say '```'

say ""
say "---"
say ""
say "본문에 넣을 사례는 wrong_to_correct에서 동료의 근거가 \`[o]\`로 표시된 것을 고른다."
say "\`[x]\`가 붙은 근거로 답이 바뀐 사례가 있다면 그것도 실을 값어치가 있다 —"
say "허구의 인용으로 설득이 성립했다는 뜻이기 때문이다."

echo
echo "[appended] $OUT"

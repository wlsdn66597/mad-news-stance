#!/usr/bin/env bash
# Every table the results section needs, for one seed, from runs already on disk.
# Offline: reads saved traces and judge outputs, calls no model.
#
#   bash scripts/report_siw_tables.sh                    # s6000, EXAONE
#   SEED=6001 bash scripts/report_siw_tables.sh
#   MODEL=qwen CONFIGN=1001 bash scripts/report_siw_tables.sh
#
# Writes docs/results_{model}_s{seed}_{mix}.md and prints the same to stdout.
# Missing conditions are reported as missing rather than crashing the run, so
# this is safe to call while the generating jobs are still going.
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
  JUDGE_FULL="results/judge_full"
else
  JUDGE_R0="results/judge_mix_${MIX}"
  JUDGE_FULL="results/judge_${MIX}_full"
fi

OUT="docs/results_${MODEL}_s${SEED}_${MIX}.md"
mkdir -p docs

say () { echo "$@" | tee -a "$OUT"; }
run () {  # run <heading> <command...>
  say ""
  say "### $1"
  say ""
  say '```'
  "${@:2}" 2>&1 | tee -a "$OUT"
  say '```'
}
have () {
  if [ -f "$1" ]; then return 0; fi
  say ""
  say "### $2"
  say ""
  say "**없음** — \`$1\`"
  return 1
}

: > "$OUT"
say "# Results — ${MODEL}, ${SPLIT} n=${N}, seed ${SEED}, mix ${MIX}"
say ""
say "생성일 $(date '+%Y-%m-%d %H:%M'). 저장된 실행 결과만 읽으며 모델을 부르지 않는다."
say "누락된 조건은 실패 대신 '없음'으로 표시되므로, 생성이 도는 중에도 돌릴 수 있다."

# --- 1. main table -----------------------------------------------------------
run "1. 메인 표 — 전 조건" \
  python scripts/collect_report_table.py --seeds "$SEED" --model "$MODEL" \
    --split "$SPLIT" --n "$N" --profile "$PROFILE" --data-seed "$DATA_SEED"

# --- 2. H1: role differentiation --------------------------------------------
say ""
say "## H1 — 역할 분화"
say ""
say "pool은 다수결이 넘을 수 없는 천장이다. 단독 역할과 조합을 나란히 둔다."
for f in "${B}"_r1_personas_only-*.json "${B}"_r1_personas_mix-*.json "${B}"_r2_personas.json; do
  [ -e "$f" ] || continue
  name=$(basename "$f" | sed 's/.*_personas//;s/^_//;s/\.json//')
  run "H1 · ${name:-mixed(F/S/W)}" \
    python scripts/analyze_agent_diversity.py --result "$f"
done

# --- 3. H2: the three-way comparison inside one mix -------------------------
say ""
say "## H2 — 같은 역할 조합 안에서의 동료 효과"
say ""
say "Majority → Self-refine → Peer exchange. 셋이 같은 Round 0을 공유해야 성립한다."
specs=()
[ -f "$MAJ" ]  && specs+=("$MAJ:majority=siw_majority")
[ -f "$SELF" ] && specs+=("$SELF:debate=siw_selfrefine_full")
[ -f "$FULL" ] && specs+=("$FULL:debate=siw_reasoned_full")
if [ "${#specs[@]}" -ge 2 ]; then
  run "H2 · paired comparison (exact McNemar)" \
    python scripts/compare_methods.py "${specs[@]}"
else
  say ""
  say "**비교 불가** — 세 조건 중 ${#specs[@]}개만 존재한다."
  say ""
  say "- majority: \`$MAJ\`"
  say "- self-refine full: \`$SELF\`"
  say "- reasoned full: \`$FULL\`"
fi

for pair in "$SELF:self-refine full" "$FULL:reasoned exchange full"; do
  f="${pair%%:*}"; label="${pair#*:}"
  have "$f" "H2 · $label vs majority" || continue
  run "H2 · $label — 라운드별 지표와 라벨 분포" \
    python scripts/analyze_rounds.py "$f"
done

# --- 4. information flow -----------------------------------------------------
flow=()
for f in "$MAJ" "$SELF" "$FULL"; do [ -f "$f" ] && flow+=("$f"); done
if [ "${#flow[@]}" -ge 1 ]; then
  say ""
  say "## 정보 흐름"
  say ""
  say "coverage가 천장, selection이 회수. 후보를 살리는 개입은 회수를 낮춘다."
  run "정보 흐름 · debate 트레이스" \
    python scripts/analyze_information_flow.py "${flow[@]}"
  [ -f "$MAJ" ] && run "정보 흐름 · Round 0 (majority 트레이스)" \
    python scripts/analyze_information_flow.py "$MAJ" --method majority
fi

# --- 5. H3: selective judge --------------------------------------------------
say ""
say "## H3 — 불일치 문항과 selective judge"
jspecs=()
for d in "$JUDGE_R0" "$JUDGE_FULL" results/judge_round0_only results/judge_freemad_first; do
  for f in "$d"/*.items.json; do
    [ -e "$f" ] && jspecs+=("$f:final_prediction=$(basename "$d")")
  done
done
if [ "${#jspecs[@]}" -ge 2 ]; then
  run "H3 · judge 조건 비교" python scripts/compare_methods.py "${jspecs[@]}"
else
  say ""
  say "**judge 출력이 2개 미만** — \`$JUDGE_R0\`, \`$JUDGE_FULL\` 확인."
fi

say ""
say "### H3 · trigger subset (판정 대상 문항의 전후)"
say ""
say '```'
python - "$JUDGE_R0" "$JUDGE_FULL" 2>&1 <<'PY' | tee -a "$OUT"
import json, glob, os, sys
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "*.summary.json")))
    if not files:
        print(f"{os.path.basename(d):40} 없음")
        continue
    s = json.load(open(files[0], encoding="utf-8"))
    sub = s.get("trigger_subset", {})
    tr = sub.get("transitions", {})
    print(f"{os.path.basename(d):40} trigger={sub.get('triggered_total')}"
          f" ({sub.get('trigger_ratio', 0):.1%})"
          f"  before={sub.get('baseline_accuracy', float('nan')):.4f}"
          f"  after={sub.get('judge_accuracy', float('nan')):.4f}"
          f"  W2C={tr.get('wrong_to_correct')}  C2W={tr.get('correct_to_wrong')}"
          f"  net={tr.get('net_improvement'):+d}"
          f"  fallback={s.get('failures', {}).get('fallback_count')}")
PY
say '```'

# --- 6. H4: rounds and over-agreement ---------------------------------------
if [ -f "$FULL" ]; then
  say ""
  say "## H4 — 라운드별 합의와 정확도"
  say ""
  say "합의가 정확도보다 빨리 오르면 conformity다. 만장일치 열과 net을 같이 읽는다."
  run "H4 · 라운드 전이" python scripts/analyze_rounds.py "$FULL" --compare "$MAJ"
  [ -f "$MAJ" ] && run "H4 · 라운드0 일치도별 이득" \
    python scripts/analyze_when_debate_helps.py --result "$MAJ" \
      --candidate-result "$FULL" --data-path "$DATA"
fi

# --- 7. channel ---------------------------------------------------------------
say ""
say "## 채널 — 동료에게 실제로 무엇이 전달되는가"
say ""
say '```'
python - "$MAJ" "$SELF" "$FULL" 2>&1 <<'PY' | tee -a "$OUT"
import json, os, statistics as st, sys
print(f"{'run':44} {'round':>5} {'label-only':>11} {'no-label':>9} {'median':>7}")
for path in sys.argv[1:]:
    if not os.path.exists(path):
        print(f"{os.path.basename(path)[-44:]:44} 없음")
        continue
    data = json.load(open(path, encoding="utf-8"))
    block = data.get("debate") or data.get("majority") or {}
    rows = [v for v in block.values() if (v.get("debate_trace") or {}).get("answers_by_round")]
    if not rows:
        rows = [{"debate_trace": {"answers_by_round": [v["raw"]]}}
                for v in block.values() if v.get("raw")]
    if not rows:
        print(f"{os.path.basename(path)[-44:]:44} 트레이스 없음")
        continue
    n_rounds = max(len(r["debate_trace"]["answers_by_round"]) for r in rows)
    name = os.path.basename(path)[-44:]
    for i in range(n_rounds):
        answers = [a for r in rows for a in r["debate_trace"]["answers_by_round"][i]]
        bare = sum(1 for a in answers if len(a.strip()) <= 40)
        missing = sum(1 for a in answers if "final stance" not in a.lower())
        print(f"{name if i == 0 else '':44} {i:>5} {bare / len(answers):10.1%} "
              f"{missing:9d} {st.median(len(a) for a in answers):6.0f}자")
PY
say '```'

say ""
say "---"
say ""
say "생성 비용: Round 0 majority 3회/item, self-refine full·reasoned full 각 12회/item"
say "(Round 0을 심으면 9회), judge는 비만장일치 문항에만 1회."
say ""
say "시드 변동은 1001문항에서 11~16건이다. 시드 간 절대 정확도를 비교하지 말고"
say "페어드 델타를 보고할 것. \`A -> B\`의 net은 B로 갈 때의 변화이며 음수면 B가 나쁘다."

echo
echo "[saved] $OUT"

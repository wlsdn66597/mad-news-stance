"""저장된 결과에서 '정답 표기 마커'가 없는(=잘림/포맷 이상 의심) 출력 개수를 method별로 집계.

  python scripts/check_trunc.py
많으면 truncation → --max-new-tokens 상향 후 재실행.
"""
import glob
import json
import re
from pathlib import Path

# 과업별 최종답 마커 (이게 없으면 파싱이 fallback으로 밀려 오답 위험)
# stance는 프롬프트 프로파일에 따라 한국어(legacy/v2_ko) 또는 영어(v2_en/minimal_en) 마커를 쓴다.
MARK = {
    "mmlu": r"\([ABCD]\)",
    "gsm8k": r"[Aa]nswer\s*[:：]",
    "stance": r"(최종\s*입장|[Ff]inal\s*stance)",
}

print(f"{'file':40} {'method':9} {'마커없음':>8}")
print("-" * 62)
for f in sorted(glob.glob("results/phase1/*.json") + glob.glob("results/phase2/*.json")):
    name = Path(f).stem
    pat = MARK.get(name.split("_")[0])
    if not pat:
        continue
    data = json.load(open(f, encoding="utf-8"))
    for method, items in data.items():
        # `_meta`, `_shared_round0` 등 method 결과가 아닌 최상위 키는 건너뛴다.
        if method.startswith("_") or not isinstance(items, dict):
            continue
        scored = {
            key: value
            for key, value in items.items()
            if isinstance(value, dict) and isinstance(value.get("raw"), list)
        }
        if not scored:
            continue
        miss = sum(not re.search(pat, " ".join(v["raw"])) for v in scored.values())
        flag = "  <-- 확인" if miss / len(scored) >= 0.2 else ""
        print(f"{name:40} {method:9} {miss:>4}/{len(scored):<3}{flag}")

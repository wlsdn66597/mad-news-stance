"""results/phase1(+phase2)의 모든 결과 JSON을 읽어 정확도 표로 한 번에 출력.

재실행 없이 저장된 결과만 집계한다.
  python scripts/report.py
"""
import glob
import json
import re
from pathlib import Path


def acc_of(items):
    return sum(v["correct"] for v in items.values()) / len(items) if items else 0.0


rows = []
for f in sorted(glob.glob("results/phase1/*.json") + glob.glob("results/phase2/*.json")):
    name = Path(f).stem                        # 예: mmlu_qwen_n50_h2_critical
    parts = name.split("_")
    task = parts[0]
    model = next((p for p in parts if p in ("qwen", "exaone")), "?")
    m = re.search(r"_n\d+_(.+)$", name)
    variant = m.group(1) if m else "baseline"  # tag 없으면 baseline
    data = json.load(open(f, encoding="utf-8"))
    for method, items in data.items():
        if items:
            rows.append((model, task, method, variant, len(items), acc_of(items)))

order = {"vanilla": 0, "cot": 1, "majority": 2, "debate": 3}
rows.sort(key=lambda r: (r[0], r[1], order.get(r[2], 9), r[3]))

print(f"{'model':8} {'task':6} {'method':9} {'variant':13} {'n':>4} {'acc':>7}")
print("-" * 56)
last = None
for model, task, method, variant, n, acc in rows:
    if last and last != (model, task):
        print()
    print(f"{model:8} {task:6} {method:9} {variant:13} {n:>4} {acc:7.3f}")
    last = (model, task)

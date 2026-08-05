"""results/phase1(+phase2)의 모든 결과 JSON을 읽어 정확도 표로 한 번에 출력.

재실행 없이 저장된 결과만 집계한다.
  python scripts/report.py
"""
import glob
import json
import re
from pathlib import Path


def scored_items(items):
    """Only per-item method results. Skips `_meta`, `_shared_round0`, etc."""
    if not isinstance(items, dict):
        return {}
    return {
        key: value
        for key, value in items.items()
        if isinstance(value, dict) and "correct" in value
    }


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
        if method.startswith("_"):
            continue
        scored = scored_items(items)
        if scored:
            rows.append((model, task, method, variant, len(scored), acc_of(scored)))

order = {
    "single": 0,
    "reflection": 1,
    "majority": 2,
    "debate": 3,
    "vanilla": 4,
    "cot": 5,
}
rows.sort(key=lambda r: (r[0], r[1], order.get(r[2], 9), r[3]))

print(f"{'model':8} {'task':6} {'method':9} {'variant':13} {'n':>4} {'acc':>7}")
print("-" * 56)
last = None
for model, task, method, variant, n, acc in rows:
    if last and last != (model, task):
        print()
    print(f"{model:8} {task:6} {method:9} {variant:13} {n:>4} {acc:7.3f}")
    last = (model, task)

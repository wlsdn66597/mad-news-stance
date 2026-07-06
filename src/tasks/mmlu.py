"""MMLU (4지선다 지식) — 논문 재현의 factuality 과업. stance(3지선다)와 구조가 가장 유사."""
import random
import re

from datasets import load_dataset

from .base import Task

LETTERS = ["A", "B", "C", "D"]


class MMLU(Task):
    name = "mmlu"

    def load(self, split="test", n=50, seed=0):
        ds = load_dataset("cais/mmlu", "all", split=split)
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        items = []
        for i in idx[:n]:
            row = ds[i]
            items.append({
                "id": int(i),
                "q": row["question"],
                "choices": list(row["choices"]),
                "gold": int(row["answer"]),  # 0~3
            })
        return items

    def question(self, item, style="cot"):
        opts = "\n".join(f"{LETTERS[j]}) {c}" for j, c in enumerate(item["choices"]))
        base = f"{item['q']}\n{opts}"
        if style == "cot":
            return (f"다음 질문에 최대한 정확히 답하라.\n{base}\n\n"
                    "단계별로 근거를 설명한 뒤, 마지막 줄에 반드시 '정답: (X)' 형식으로 "
                    "A/B/C/D 중 하나를 써라.")
        return (f"다음 질문에 답하라.\n{base}\n\n"
                "마지막 줄에 반드시 '정답: (X)' 형식으로 A/B/C/D 중 하나만 써라.")

    def parse(self, text):
        m = re.findall(r"정답\s*[:：]?\s*\(?\s*([ABCD])\s*\)?", text)
        if m:
            return m[-1]
        m2 = re.findall(r"\(([ABCD])\)", text)
        if m2:
            return m2[-1]
        m3 = re.findall(r"\b([ABCD])\b", text)
        return m3[-1] if m3 else None

    def correct(self, pred, item):
        if pred in LETTERS:
            return LETTERS.index(pred) == item["gold"]
        return False

"""MMLU (4지선다 지식) — 논문 재현의 factuality 과업.

프롬프트는 Du et al. 원문 그대로(영어). stance(3지선다)와 구조가 가장 유사.
"""
import random
import re

from datasets import load_dataset

from .base import Task

LETTERS = ["A", "B", "C", "D"]


class MMLU(Task):
    name = "mmlu"
    # 논문 gen_mmlu.py construct_message 원문
    debate_template = (
        "These are the solutions to the problem from other agents:\n\n{others}\n\n"
        "Using the reasoning from other agents as additional advice, can you give an "
        "updated answer? Examine your solution and that of other agents step by step. "
        "Put your answer in the form (X) at the end of your response."
    )
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
        opts = ", ".join(f"{LETTERS[j]}) {c}" for j, c in enumerate(item["choices"]))
        base = f"{item['q']}: {opts}"
        if style == "cot":
            # 논문 원문 프롬프트 (설명 포함 = single-agent baseline)
            return (f"Can you answer the following question as accurately as possible? {base} "
                    "Explain your answer, putting the answer in the form (X) at the end of your response.")
        return (f"Answer the following question. {base} "
                "Put the answer in the form (X) at the end of your response.")

    def parse(self, text):
        m = re.findall(r"\(([ABCD])\)", text)          # 논문 포맷 (X)
        if m:
            return m[-1]
        m2 = re.findall(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?\b", text, re.IGNORECASE)
        if m2:
            return m2[-1]
        m3 = re.findall(r"\b([ABCD])\b", text)
        return m3[-1] if m3 else None

    def correct(self, pred, item):
        return pred in LETTERS and LETTERS.index(pred) == item["gold"]

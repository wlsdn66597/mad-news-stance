"""GSM8K (초등 수학 추론) — 논문 재현의 reasoning 과업. 프롬프트 영어."""
import random
import re

from datasets import load_dataset

from .base import Task


class GSM8K(Task):
    name = "gsm8k"
    debate_template = (
        "These are the solutions to the problem from other agents:\n\n{others}\n\n"
        "Using the reasoning from other agents as additional advice, can you give an "
        "updated answer? Examine your solution and that of other agents step by step. "
        "Put your final numeric answer after 'Answer:'."
    )
    # H2: 비판적 프롬프트 (동조 억제, 오류 지적 유도)
    debate_template_critical = (
        "These are solutions from other agents:\n\n{others}\n\n"
        "Critically examine each agent's reasoning for mistakes — do NOT simply agree. "
        "If another agent's reasoning is flawed, identify the specific error. Only revise "
        "your answer if you find a genuine mistake. Then reason step by step and put your "
        "final numeric answer after 'Answer:'."
    )

    def load(self, split="test", n=50, seed=0):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        items = []
        for i in idx[:n]:
            row = ds[i]
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"id": int(i), "q": row["question"], "gold": gold})
        return items

    def question(self, item, style="cot"):
        base = item["q"]
        if style == "cot":
            return (f"{base}\n\nExplain your reasoning step by step, then give the final "
                    "numeric answer after 'Answer:'.")
        return f"{base}\n\nGive the final numeric answer after 'Answer:'."

    def parse(self, text):
        m = re.findall(r"[Aa]nswer\s*[:：]?\s*\$?\s*(-?\d[\d,]*\.?\d*)", text)
        if m:
            return m[-1].replace(",", "").rstrip(".")
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "").rstrip(".") if nums else None

    def correct(self, pred, item):
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(item["gold"])) < 1e-4
        except ValueError:
            return str(pred).strip() == str(item["gold"]).strip()

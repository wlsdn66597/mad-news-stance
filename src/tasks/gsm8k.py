"""GSM8K (초등 수학 추론) — 논문 재현의 reasoning 과업."""
import random
import re

from datasets import load_dataset

from .base import Task


class GSM8K(Task):
    name = "gsm8k"

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
            return (f"{base}\n\n단계별로 풀이 과정을 쓰고, "
                    "마지막 줄에 반드시 '정답: <숫자>' 형식으로 답하라.")
        return f"{base}\n\n마지막 줄에 반드시 '정답: <숫자>' 형식으로 답만 써라."

    def parse(self, text):
        # 1) '정답:' 뒤의 첫 숫자
        for pat in (r"정답\s*[:：]\s*([^\n]+)", r"answer\s*[:：]\s*([^\n]+)"):
            m = re.findall(pat, text, flags=re.IGNORECASE)
            if m:
                nums = re.findall(r"-?\d[\d,]*\.?\d*", m[-1])
                if nums:
                    return nums[0].replace(",", "").rstrip(".")
        # 2) fallback: 본문 마지막 숫자
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "").rstrip(".") if nums else None

    def correct(self, pred, item):
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(item["gold"])) < 1e-4
        except ValueError:
            return str(pred).strip() == str(item["gold"]).strip()

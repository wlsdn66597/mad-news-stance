"""GSM8K (초등 수학 추론) — 논문 재현의 reasoning 과업. 프롬프트 영어."""
import random
import re

from datasets import load_dataset

from .base import Task


class GSM8K(Task):
    name = "gsm8k"
    # composable-models/llm_multiagent_debate gsm/gen_gsm.py 원문.
    reflection_template = (
        "Can you double check that your answer is correct. Please reiterate your "
        "answer, with your final answer a single numerical number, in the form "
        "\\boxed{answer}."
    )
    paper_debate_template = (
        "These are the solutions to the problem from other agents: {others}"
        "\n\n Using the solutions from other agents as additional information, can "
        "you provide your answer to the math problem? \n The original math problem "
        "is {question}. Your final answer should be a single numerical number, in "
        "the form \\boxed{{answer}}, at the end of your response."
    )
    debate_template = (
        "These are the solutions to the problem from other agents:\n\n{others}\n\n"
        "Using the reasoning from other agents as additional advice, can you give an "
        "updated answer? Examine your solution and that of other agents step by step. "
        "Put your final numeric answer after 'Answer:'."
    )
    def load(self, split="test", n=50, seed=0, sampling_protocol="uniform"):
        if sampling_protocol not in {"uniform", "paper"}:
            raise ValueError(f"unknown sampling protocol: {sampling_protocol}")
        ds = load_dataset("openai/gsm8k", "main", split=split)
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        items = []
        for i in idx[:n]:
            row = ds[i]
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"id": int(i), "source_index": int(i), "q": row["question"], "gold": gold})
        return items

    def question(self, item, style="cot"):
        base = item["q"]
        if style == "paper":
            return (
                f"Can you solve the following math problem? {base} Explain your "
                "reasoning. Your final answer should be a single numerical number, "
                "in the form \\boxed{{answer}}, at the end of your response. "
            )
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

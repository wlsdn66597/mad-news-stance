"""Korean news stance detection task (K-News-Stance).

Input: issue + headline + article -> supportive / oppositional / neutral.
Prompt wording is selected by a versioned profile so paper-aligned experiments and
application-oriented prompt ablations do not overwrite each other.
"""
import json
import random
import re

from ..prompts.stance import get_stance_prompt_profile
from .base import Task

LABELS = ["supportive", "oppositional", "neutral"]


class Stance(Task):
    name = "stance"

    def __init__(
        self,
        data_path="data/k-news-stance_nosegment.json",
        prompt_profile="legacy_ko",
    ):
        self.data_path = data_path
        self.prompt_profile = get_stance_prompt_profile(prompt_profile)
        self.prompt_profile_name = self.prompt_profile.name
        self.debate_template = self.prompt_profile.debate_template
        self.self_refine_template = self.prompt_profile.self_refine_template
        self.debate_protocols = self.prompt_profile.debate_protocols or {}
        self.memory_summary_template = self.prompt_profile.memory_summary_template
        self.memory_debate_template = self.prompt_profile.memory_debate_template

    def load(self, split="validation", n=None, seed=0):
        with open(self.data_path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = [row for row in data if row.get("split") == split]
        random.Random(seed).shuffle(rows)
        if n:
            rows = rows[:n]
        return [
            {
                "id": row["id"],
                "issue": row["issue"],
                "headline": row.get("haedline") or row.get("headline", ""),
                "article": row["article"],
                "gold": row["stance"],
                "genre": row.get("genre"),
            }
            for row in rows
        ]

    def question(self, item, style="cot"):
        return self.prompt_profile.question(item, style=style)

    def memory_context(self, item):
        return self.prompt_profile.memory_context(item)

    def format_other_answers(self, answers):
        return self.prompt_profile.format_other_answers(answers)

    def parse(self, text):
        lowered = text.lower()
        final_patterns = [
            r"final\s*stance\s*[:：]?\s*[\"'(]?\s*([a-z]+)",
            r"최종\s*입장\s*[:：]?\s*[\"'(]?\s*([a-z]+)",
        ]
        for pattern in final_patterns:
            matches = re.findall(pattern, lowered)
            if matches:
                for label in LABELS:
                    if matches[-1].startswith(label[:4]):
                        return label

        found = [label for label in LABELS if label in lowered]
        if found:
            return max(found, key=lambda label: lowered.rfind(label))
        if any(keyword in text for keyword in ["지지", "찬성", "우호"]):
            return "supportive"
        if any(keyword in text for keyword in ["반대", "비판", "부정적"]):
            return "oppositional"
        if "중립" in text:
            return "neutral"
        return None

    def correct(self, pred, item):
        return pred == item["gold"]

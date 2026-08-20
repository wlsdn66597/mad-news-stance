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
        segment_labels_path=None,
    ):
        self.data_path = data_path
        # JOA-ICL's article-level stage reads the same article with a predicted
        # stance attached to each journalism-guided segment. Swapping the text
        # at load time rather than adding a method keeps every method, profile
        # and protocol working on it unchanged, and keeps the item ids, so the
        # runs stay paired with the plain-article ones for exact McNemar.
        self.segment_labels_path = segment_labels_path
        self.prompt_profile = get_stance_prompt_profile(prompt_profile)
        self.prompt_profile_name = self.prompt_profile.name
        self.debate_template = self.prompt_profile.debate_template
        self.self_refine_template = self.prompt_profile.self_refine_template
        self.reflection_template = self.prompt_profile.reflection_template
        self.self_refine_reasoned_template = (
            self.prompt_profile.self_refine_reasoned_template)
        self.self_refine_full_template = (
            self.prompt_profile.self_refine_full_template)
        self.debate_protocols = self.prompt_profile.debate_protocols or {}
        self.memory_summary_template = self.prompt_profile.memory_summary_template
        self.memory_debate_template = self.prompt_profile.memory_debate_template

    def segment_labeled_articles(self):
        """Segment-labelled headline and body, keyed by item id.

        Only the text is taken. The issue and the gold label keep coming from
        the dataset, so a segment-label file can never introduce a label the
        run is then scored against.
        """
        if not self.segment_labels_path:
            return {}
        with open(self.segment_labels_path, encoding="utf-8") as handle:
            rows = json.load(handle)
        overlay = {}
        for row in rows:
            headline = row.get("title_joa_icl")
            article = row.get("main_body_joa_icl")
            if not article:
                continue
            overlay[str(row["id"])] = (headline or "", article)
        if not overlay:
            raise ValueError(
                f"{self.segment_labels_path} carries no title_joa_icl/"
                "main_body_joa_icl rows"
            )
        return overlay

    def load(self, split="validation", n=None, seed=0):
        with open(self.data_path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = [row for row in data if row.get("split") == split]
        random.Random(seed).shuffle(rows)
        if n:
            rows = rows[:n]
        overlay = self.segment_labeled_articles()
        items = []
        for row in rows:
            item = {
                "id": row["id"],
                "issue": row["issue"],
                "headline": row.get("haedline") or row.get("headline", ""),
                "article": row["article"],
                "gold": row["stance"],
                "genre": row.get("genre"),
            }
            if overlay:
                # a silently unlabelled item would be compared against labelled
                # ones as if the condition were the same, so refuse instead
                labelled = overlay.get(str(row["id"]))
                if labelled is None:
                    raise ValueError(
                        f"item {row['id']} has no entry in "
                        f"{self.segment_labels_path}; the split would be a mix "
                        "of labelled and unlabelled articles"
                    )
                item["headline"], item["article"] = labelled
                item["segment_labeled"] = True
            items.append(item)
        return items

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

"""한국어 뉴스 입장 탐지 (K-News-Stance).

입력: issue + headline + article  →  supportive / oppositional / neutral (3지선다)
프롬프트는 한국어(데이터가 한국어). debate 프롬프트는 논문 문구를 한국어로 옮긴 것.
"""
import json
import random
import re

from .base import Task

LABELS = ["supportive", "oppositional", "neutral"]


class Stance(Task):
    name = "stance"
    # 논문 debate 프롬프트의 한국어 대응 (MMLU/GSM8K 영어판과 병렬)
    debate_template = (
        "다음은 다른 에이전트들이 같은 기사에 대해 내놓은 판단입니다:\n\n{others}\n\n"
        "다른 에이전트들의 추론을 추가 조언으로 참고하여, 당신의 판단과 다른 에이전트들의 판단을 "
        "단계별로 검토한 뒤 갱신된 답을 제시하라. "
        "마지막 줄에 반드시 '최종 입장: <supportive|oppositional|neutral>' 형식으로 답하라."
    )

    def __init__(self, data_path="data/k-news-stance_nosegment.json"):
        self.data_path = data_path

    def load(self, split="validation", n=None, seed=0):
        data = json.load(open(self.data_path, encoding="utf-8"))
        rows = [d for d in data if d.get("split") == split]
        random.Random(seed).shuffle(rows)
        if n:
            rows = rows[:n]
        items = []
        for d in rows:
            items.append({
                "id": d["id"],
                "issue": d["issue"],
                "headline": d.get("haedline") or d.get("headline", ""),  # 원본 필드 오타 'haedline'
                "article": d["article"],
                "gold": d["stance"],
            })
        return items

    def question(self, item, style="cot"):
        header = (f"이슈: {item['issue']}\n"
                  f"제목: {item['headline']}\n"
                  f"기사:\n{item['article']}")
        desc = ("위 기사가 해당 이슈에 대해 취하는 입장을 판단하라. "
                "supportive(지지/찬성), oppositional(반대/비판), neutral(중립) 중 하나다. "
                "인용문 화자의 입장이 아니라 기사 자체의 논조를 기준으로 판단하라.")
        fmt = ("마지막 줄에 반드시 '최종 입장: <supportive|oppositional|neutral>' "
               "형식으로 셋 중 하나만 영어로 써라.")
        if style == "cot":
            return f"{header}\n\n{desc}\n\n먼저 핵심 근거 문장을 짚은 뒤 판단하고, {fmt}"
        return f"{header}\n\n{desc}\n\n{fmt}"

    def parse(self, text):
        t = text.lower()
        m = re.findall(r"최종\s*입장\s*[:：]?\s*[\"'(]?\s*([a-z]+)", t)
        if m:
            for lab in LABELS:
                if m[-1].startswith(lab[:4]):
                    return lab
        found = [lab for lab in LABELS if lab in t]
        if found:
            return max(found, key=lambda lab: t.rfind(lab))
        if any(k in text for k in ["지지", "찬성", "우호"]):
            return "supportive"
        if any(k in text for k in ["반대", "비판", "부정적"]):
            return "oppositional"
        if "중립" in text:
            return "neutral"
        return None

    def correct(self, pred, item):
        return pred == item["gold"]

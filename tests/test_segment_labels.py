import json
import tempfile
import unittest
from pathlib import Path

from src.prompts.stance import get_stance_prompt_profile
from src.tasks.stance import Stance


DATA = [
    {"id": 1, "split": "test", "issue": "의대 정원", "haedline": "정원 확대 발표",
     "article": "의회는 예산안을 통과시켰다.", "stance": "neutral", "genre": "analysis"},
    {"id": 2, "split": "test", "issue": "의대 정원", "haedline": "반발 확산",
     "article": "업계는 반발했다.", "stance": "oppositional", "genre": "analysis"},
]

JOA = [
    {"id": 1, "event_name": "의대 정원",
     "title_joa_icl": '<제목 입장="중립적">정원 확대 발표</제목>',
     "main_body_joa_icl": '<도입부 입장="중립적">의회는 예산안을 통과시켰다.</도입부>'},
    {"id": 2, "event_name": "의대 정원",
     "title_joa_icl": '<제목 입장="비판적">반발 확산</제목>',
     "main_body_joa_icl": '<도입부 입장="비판적">업계는 반발했다.</도입부>'},
]


def write(directory, name, payload):
    path = Path(directory) / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


class SegmentLabelOverlayTest(unittest.TestCase):
    def test_without_the_file_nothing_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Stance(write(directory, "data.json", DATA), prompt_profile="stance_minimal_en")
            items = task.load(split="test", n=2, seed=0)
            self.assertEqual({item["id"] for item in items}, {1, 2})
            for item in items:
                self.assertNotIn("segment_labeled", item)
                self.assertNotIn("입장=", item["article"])
            self.assertNotIn("입장=", task.question(items[0]))

    def test_the_overlay_replaces_the_text_and_keeps_issue_and_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Stance(
                write(directory, "data.json", DATA),
                prompt_profile="stance_minimal_en",
                segment_labels_path=write(directory, "joa.json", JOA),
            )
            items = {item["id"]: item for item in task.load(split="test", n=2, seed=0)}
            self.assertIn('입장="중립적"', items[1]["article"])
            self.assertIn('입장="중립적"', items[1]["headline"])
            self.assertTrue(items[1]["segment_labeled"])
            # a segment-label file must never be able to supply a label the run
            # is then scored against
            self.assertEqual(items[1]["gold"], "neutral")
            self.assertEqual(items[2]["gold"], "oppositional")
            self.assertEqual(items[1]["issue"], "의대 정원")

    def test_a_missing_item_is_refused_rather_than_left_unlabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Stance(
                write(directory, "data.json", DATA),
                prompt_profile="stance_minimal_en",
                segment_labels_path=write(directory, "joa.json", JOA[:1]),
            )
            with self.assertRaises(ValueError):
                task.load(split="test", n=2, seed=0)


class SegmentNoteTest(unittest.TestCase):
    def test_the_note_appears_only_for_a_labelled_item(self):
        profile = get_stance_prompt_profile("stance_minimal_en")
        plain = {"issue": "i", "headline": "h", "article": "a"}
        labelled = {**plain, "segment_labeled": True}
        self.assertNotIn("입장=", profile.article_block(plain))
        self.assertIn("입장=", profile.article_block(labelled))
        self.assertIn("supportive", profile.article_block(labelled))

    def test_a_korean_profile_gets_the_korean_note(self):
        profile = get_stance_prompt_profile("legacy_ko")
        labelled = {"issue": "i", "headline": "h", "article": "a", "segment_labeled": True}
        block = profile.article_block(labelled)
        self.assertIn("참고 단서이며 정답이 아니다", block)

    def test_the_note_says_the_labels_are_cues_rather_than_the_answer(self):
        # the span stances routinely disagree with the article stance, so a
        # prompt that reads as "here is the answer" would measure obedience
        profile = get_stance_prompt_profile("stance_minimal_en")
        block = profile.article_block(
            {"issue": "i", "headline": "h", "article": "a", "segment_labeled": True}
        )
        self.assertIn("not as the answer", block)


if __name__ == "__main__":
    unittest.main()

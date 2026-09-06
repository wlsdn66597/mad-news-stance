import unittest

from src.joa_zero_shot import (
    annotate_markup,
    extract_segments,
    prepare_segment_row,
    remove_segment_labels,
    segment_prompt,
)


class JoaZeroShotPreparationTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "id": 7,
            "event_name": "정책 확대",
            "title_joa_icl": '<제목 입장="중립적">정책 확대 발표</제목>',
            "main_body_joa_icl": (
                '<도입부 입장="지지적">정부는 <직접 인용구 입장="지지적">'
                '필요하다</직접 인용구>고 밝혔다.</도입부>\n본문\n'
                '<결론부 입장="비판적">시민단체는 반대했다.</결론부>'
            ),
        }

    def test_removes_labels_but_keeps_boundaries(self):
        cleaned = remove_segment_labels(self.row["main_body_joa_icl"])
        self.assertNotIn("입장=", cleaned)
        self.assertIn("<도입부>", cleaned)
        self.assertIn("<직접 인용구>", cleaned)

    def test_extracts_outer_and_nested_segments_once(self):
        prepared = prepare_segment_row(self.row)
        self.assertEqual(
            [segment["type"] for segment in prepared["segments"]],
            ["headline", "lead", "quotation", "conclusion"],
        )
        lead = prepared["segments"][1]
        self.assertEqual(lead["text"], "정부는 필요하다고 밝혔다.")
        self.assertNotIn("입장=", str(prepared))

    def test_annotates_each_opening_tag_with_new_prediction(self):
        prepared = prepare_segment_row(self.row)
        predictions = [
            {**segment, "pred": label}
            for segment, label in zip(
                prepared["segments"],
                ["neutral", "supportive", "oppositional", "neutral"],
            )
        ]
        body = annotate_markup(
            prepared["main_body_joa_unlabeled"], "body", predictions
        )
        self.assertIn('<도입부 입장="지지적">', body)
        self.assertIn('<직접 인용구 입장="비판적">', body)
        self.assertIn('<결론부 입장="중립적">', body)

    def test_prompt_contains_no_segment_type_or_previous_label(self):
        prompt = segment_prompt("정책 확대", "정부가 발표했다.")
        self.assertIn("Issue: 정책 확대", prompt)
        self.assertIn("News segment:\n정부가 발표했다.", prompt)
        self.assertNotIn("Headline", prompt)
        self.assertNotIn("입장=", prompt)

    def test_unclosed_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_segments("<도입부>끝나지 않음", "body")


if __name__ == "__main__":
    unittest.main()

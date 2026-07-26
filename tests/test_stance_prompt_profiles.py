import unittest

from src.prompts.stance import PROFILES
from src.tasks.stance import Stance


ITEM = {
    "id": 1,
    "issue": "의대 정원 확대",
    "headline": "정부, 확대 필요성 강조",
    "article": "정부는 확대가 필요하다고 밝혔다. 기자는 쟁점이 남았다고 설명했다.",
    "gold": "neutral",
}


class StancePromptProfileTest(unittest.TestCase):
    def test_profiles_keep_korean_article_and_are_versioned(self):
        english = Stance(prompt_profile="stance_v2_en")
        korean = Stance(prompt_profile="stance_v2_ko")
        self.assertIn(ITEM["article"], english.question(ITEM, style="cot"))
        self.assertIn(ITEM["article"], korean.question(ITEM, style="cot"))
        self.assertIn("Authorial evidence", english.question(ITEM, style="cot"))
        self.assertIn("기사 자체의 근거", korean.question(ITEM, style="cot"))

    def test_generic_memory_control_changes_only_memory_prompts(self):
        structured = PROFILES["stance_v2_en"]
        generic = PROFILES["stance_v2_en_generic_memory"]
        self.assertEqual(structured.cot_instruction, generic.cot_instruction)
        self.assertEqual(structured.debate_template, generic.debate_template)
        self.assertIsNotNone(structured.memory_summary_template)
        self.assertIsNone(generic.memory_summary_template)

    def test_structured_memory_receives_article_context(self):
        task = Stance(prompt_profile="stance_v2_en")
        prompt = task.memory_summary_template.format(
            task_context=task.memory_context(ITEM),
            responses="[Agent 1]\nFinal stance: neutral",
        )
        self.assertIn(ITEM["article"], prompt)
        self.assertIn("[Authorial evidence]", prompt)
        self.assertIn("Do not decide the final label", prompt)

    def test_minimal_english_profile_matches_slide_format(self):
        task = Stance(prompt_profile="stance_minimal_en")
        prompt = task.question(ITEM, style="cot")
        self.assertTrue(prompt.startswith("Classify this article's stance"))
        self.assertLess(prompt.index("Final stance:"), prompt.index("Issue:"))
        self.assertIn("Article:\n" + ITEM["article"], prompt)
        self.assertNotIn("Article (Korean):", prompt)
        formatted = task.format_other_answers(["First", "Second"])
        self.assertEqual(formatted, "Agent 1:\nFirst\n\nAgent 2:\nSecond")
        debate = task.debate_template.format(others=formatted)
        self.assertLess(debate.index("Final stance:"), debate.index("Agent 1:"))

    def test_parser_accepts_both_final_markers(self):
        task = Stance(prompt_profile="stance_v2_en")
        self.assertEqual(task.parse("Final stance: supportive"), "supportive")
        self.assertEqual(task.parse("최종 입장: oppositional"), "oppositional")


if __name__ == "__main__":
    unittest.main()

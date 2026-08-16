import unittest

from src.prompts.stance import (
    LABEL_LINE_EN,
    PROFILES,
    STANCE_PERSONAS,
    get_stance_prompt_profile,
    stance_personas,
)
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


class NeutralGateProfileTest(unittest.TestCase):
    """`stance_minimal_en` asks for one of three labels and EXAONE-4.0-1.2B then
    recovers 25% of neutral articles while picking the right polarity on 74% of
    the ones that do take a side. These profiles ask the same question as two
    decisions so that the neutral call stops competing with the polarity call."""

    ITEM = {
        "issue": "김포 서울 편입",
        "headline": "편입 논의 본격화",
        "article": "기사 본문입니다.",
    }

    def test_both_are_registered(self):
        for name in ("stance_twostep_en", "stance_gate_en"):
            self.assertEqual(get_stance_prompt_profile(name).name, name)

    def test_the_output_contract_is_unchanged(self):
        """The parser and every saved comparison depend on this line, so a new
        framing must not change how the answer is read."""
        for name in ("stance_minimal_en", "stance_twostep_en", "stance_gate_en"):
            profile = get_stance_prompt_profile(name)
            for style in ("cot", "vanilla"):
                self.assertIn(LABEL_LINE_EN, profile.question(self.ITEM, style), name)
            self.assertIn(LABEL_LINE_EN, profile.debate_template, name)

    def test_the_gate_comes_before_the_polarity(self):
        for name in ("stance_twostep_en", "stance_gate_en"):
            text = get_stance_prompt_profile(name).cot_instruction.lower()
            self.assertLess(text.index("side"), text.index("oppositional"), name)
            self.assertIn("no side", text, name)

    def test_only_the_gate_variant_defines_taking_no_side(self):
        gate = get_stance_prompt_profile("stance_gate_en").cot_instruction.lower()
        twostep = get_stance_prompt_profile("stance_twostep_en").cot_instruction.lower()
        self.assertIn("only reports", gate)
        self.assertNotIn("only reports", twostep)

    def test_the_article_still_reaches_the_prompt(self):
        for name in ("stance_twostep_en", "stance_gate_en"):
            profile = get_stance_prompt_profile(name)
            self.assertIn(self.ITEM["article"], profile.question(self.ITEM, "cot"), name)


class AgentPersonaTest(unittest.TestCase):
    """Three samples of one prompt are near-copies: measured on EXAONE-4.0-1.2B
    the agents score 0.4735/0.4725/0.4745, agree 86.5% of the time at round 0,
    and all three miss on 45.4% of items where independent failures at those
    accuracies would miss on 14.6%."""

    def test_off_returns_none_so_old_runs_are_untouched(self):
        self.assertIsNone(stance_personas(3, enabled=False))

    def test_one_prompt_per_agent_and_all_distinct(self):
        personas = stance_personas(3)
        self.assertEqual(len(personas), 3)
        self.assertEqual(len(set(personas)), 3)

    def test_asking_for_more_agents_than_roles_is_refused(self):
        with self.assertRaises(ValueError):
            stance_personas(len(STANCE_PERSONAS) + 1)

    def test_the_roles_decompose_the_judgement_the_prompt_already_asks_for(self):
        framing, sourcing, wording = stance_personas(3)
        self.assertIn("framing", framing.lower())
        self.assertIn("quoted", sourcing.lower())
        self.assertIn("wording", wording.lower())

    def test_no_persona_names_a_stance_to_argue_for(self):
        """These diversify how the article is read; they do not assign a side.
        Assigning sides was measured separately and lost to majority everywhere."""
        for persona in stance_personas(3):
            lowered = persona.lower()
            self.assertNotIn("argue", lowered)
            self.assertNotIn("supportive", lowered)
            self.assertNotIn("oppositional", lowered)


class MinimalEnBaselineTemplatesTest(unittest.TestCase):
    """The rows the plan's baseline column needs, and a fair memory test."""

    def setUp(self):
        from src.prompts.stance import PROFILES
        self.profile = PROFILES["stance_minimal_en"]

    def test_reflection_prompt_ends_where_the_parser_looks(self):
        template = self.profile.reflection_template
        self.assertIsNotNone(template)
        self.assertTrue(template.rstrip().endswith(
            "Final stance: <supportive|oppositional|neutral>"))

    def test_memory_uses_the_stance_prompts_not_the_generic_fallback(self):
        from src.prompts.stance import PROFILES
        v2 = PROFILES["stance_v2_en"]
        self.assertEqual(self.profile.memory_summary_template,
                         v2.memory_summary_template)
        self.assertEqual(self.profile.memory_debate_template,
                         v2.memory_debate_template)
        # the property the memory variant is being tested for
        self.assertIn("Preserve minority evidence",
                      self.profile.memory_summary_template)

    def test_reflection_is_not_the_cost_matched_control(self):
        # self_refine_template runs three agents for as many rounds as the
        # debate; reflection is one agent checking itself once
        self.assertNotEqual(self.profile.reflection_template,
                            self.profile.self_refine_template)

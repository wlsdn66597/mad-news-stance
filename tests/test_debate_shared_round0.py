import sys
import types
import unittest
from unittest.mock import patch


fake_llm = types.ModuleType("src.llm")
fake_llm.build_messages = lambda user_content, system_prompt=None, history=None: (
    ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    + (list(history) if history else [])
    + [{"role": "user", "content": user_content}]
)
fake_llm.chat = lambda *args, **kwargs: ""
fake_llm.strip_think = lambda text: text
sys.modules.setdefault("src.llm", fake_llm)

from src import debate  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class SharedRoundZeroTest(unittest.TestCase):
    def test_each_agent_can_receive_a_distinct_initial_question(self):
        seen = []

        def fake_chat(model, tokenizer, messages, max_new_tokens, temperature,
                      enable_thinking):
            seen.append(messages[-1]["content"])
            return "Final stance: neutral"

        with patch.object(debate, "chat", side_effect=fake_chat):
            trace = debate.run_debate(
                object(), FakeTokenizer(), "shared trace label",
                n_agents=3, n_rounds=1,
                agent_questions=["headline input", "lead input", "quote input"],
            )

        self.assertEqual(seen, ["headline input", "lead input", "quote input"])
        self.assertEqual(trace["agent_questions"], seen)

    def test_agent_question_count_must_match_agents(self):
        with self.assertRaisesRegex(ValueError, "agent_questions"):
            debate.run_debate(
                object(), FakeTokenizer(), "question",
                n_agents=3, n_rounds=1, agent_questions=["only one"],
            )

    def test_reuse_and_article_grounded_memory(self):
        calls = []

        def fake_chat(model, tokenizer, messages, max_new_tokens, temperature,
                      enable_thinking):
            prompt = messages[-1]["content"]
            calls.append(prompt)
            return "Evidence ledger" if prompt.startswith("MEMORY") else "Final stance: neutral"

        with patch.object(debate, "chat", side_effect=fake_chat):
            trace = debate.run_debate(
                object(),
                FakeTokenizer(),
                "Original Korean article",
                n_agents=3,
                n_rounds=2,
                use_memory=True,
                memory_context="ARTICLE CONTEXT",
                prompt_profile="stance_v2_en",
                initial_answers=[
                    "Final stance: supportive",
                    "Final stance: neutral",
                    "Final stance: oppositional",
                ],
                memory_summary_template="MEMORY\n{task_context}\n{responses}",
                memory_debate_template="LEDGER\n{memory}",
                memory_temperature=0.0,
            )

        self.assertEqual(trace["initial_answer_source"], "provided")
        self.assertEqual(trace["prompt_profile"], "stance_v2_en")
        self.assertEqual(trace["call_counts"]["reused_initial_answers"], 3)
        self.assertEqual(trace["call_counts"]["executed_agent_generation_calls"], 3)
        self.assertEqual(trace["call_counts"]["memory_generation_calls"], 1)
        self.assertEqual(trace["call_counts"]["executed_generation_calls"], 4)
        self.assertIn("ARTICLE CONTEXT", trace["memory_by_round"][0]["summary_prompt"])
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()

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


class DebateMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def fake_chat(
        self,
        model,
        tokenizer,
        messages,
        max_new_tokens,
        temperature,
        enable_thinking,
    ):
        prompt = messages[-1]["content"]
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        if prompt.startswith("You are the shared memory agent"):
            return "Agent 1 supports Answer: 1; other agents agree; no unresolved issue."
        return (
            "Detailed independent reasoning with repeated evidence "
            + "evidence " * 40
            + "Final Answer: 1"
        )

    def test_shared_memory_adds_one_summary_call_and_compresses_context(self):
        with patch.object(debate, "chat", side_effect=self.fake_chat):
            trace = debate.run_debate(
                object(),
                self.tokenizer,
                "Solve the problem. Answer after 'Answer:'.",
                n_agents=5,
                n_rounds=2,
                use_memory=True,
                memory_max_new_tokens=64,
                memory_temperature=0.0,
            )

        self.assertEqual(trace["communication_mode"], "shared_summary")
        self.assertEqual(trace["call_counts"]["agent_generation_calls"], 10)
        self.assertEqual(trace["call_counts"]["memory_generation_calls"], 1)
        self.assertEqual(trace["call_counts"]["total_generation_calls"], 11)
        self.assertEqual(len(trace["memory_by_round"]), 1)
        self.assertEqual(len(trace["memory_by_round"][0]["source_answers"]), 5)
        self.assertEqual([len(row) for row in trace["answers_by_round"]], [5, 5])

        stats = trace["communication_stats_by_round"][1]
        self.assertLess(stats["per_agent_context_token_ratio"], 1.0)
        self.assertTrue(
            all(
                "shared memory agent compressed" in context[-2]["content"]
                for context in trace["agent_contexts"]
            )
        )

    def test_direct_mode_preserves_original_call_count(self):
        with patch.object(debate, "chat", side_effect=self.fake_chat):
            trace = debate.run_debate(
                object(),
                self.tokenizer,
                "Solve the problem.",
                n_agents=5,
                n_rounds=2,
            )

        self.assertEqual(trace["communication_mode"], "direct_concat")
        self.assertEqual(trace["call_counts"]["total_generation_calls"], 10)
        self.assertEqual(trace["memory_by_round"], [])
        self.assertEqual(len(self.calls), 10)


if __name__ == "__main__":
    unittest.main()

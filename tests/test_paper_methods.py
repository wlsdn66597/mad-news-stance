import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

fake_llm = types.ModuleType("src.llm")
fake_llm.build_messages = lambda question, system_prompt=None: [
    {"role": "user", "content": question}
]
fake_llm.chat = lambda *args, **kwargs: ""
fake_llm.strip_think = lambda text: text

with patch.dict(sys.modules, {"src.llm": fake_llm}):
    from src import methods


class FakeTask:
    name = "fake"
    reflection_template = "DOUBLE CHECK"
    debate_template = "OTHER {others}"
    paper_debate_template = "PAPER {others} QUESTION {question}"

    def question(self, item, style="cot"):
        return f"{style.upper()} QUESTION"

    def parse(self, text):
        return text[-1]


class PaperMethodsTest(unittest.TestCase):
    def test_single_uses_paper_initial_prompt(self):
        seen = []

        def fake_chat(model, tokenizer, messages, **kwargs):
            seen.append(deepcopy(messages))
            return "answer A"

        with patch.object(methods, "chat", side_effect=fake_chat):
            result = methods.run_single(None, None, FakeTask(), {}, None, 32)

        self.assertEqual(result["pred"], "A")
        self.assertEqual(result["prompt"], "PAPER QUESTION")
        self.assertEqual(len(seen), 1)

    def test_reflection_reuses_context_and_scores_second_answer(self):
        seen = []
        replies = iter(["initial A", "revised B"])

        def fake_chat(model, tokenizer, messages, **kwargs):
            seen.append(deepcopy(messages))
            return next(replies)

        with patch.object(methods, "chat", side_effect=fake_chat):
            result = methods.run_reflection(
                None, None, FakeTask(), {}, None, 32, temperature=1.0
            )

        self.assertEqual(result["pred"], "B")
        self.assertEqual(result["preds"], ["A", "B"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(
            seen[1],
            [
                {"role": "user", "content": "PAPER QUESTION"},
                {"role": "assistant", "content": "initial A"},
                {"role": "user", "content": "DOUBLE CHECK"},
            ],
        )
        self.assertEqual(result["reflection_trace"]["generation_calls"], 2)

    def test_paper_debate_uses_paper_prompt_and_formatter(self):
        captured = {}

        def fake_engine(*args, **kwargs):
            captured.update(kwargs)
            return {
                "answers_by_round": [["initial A"], ["final B"]],
                "question": args[2],
            }

        with patch.object(methods, "_debate_engine", side_effect=fake_engine):
            result = methods.run_debate(
                None,
                None,
                FakeTask(),
                {},
                None,
                32,
                n_agents=1,
                n_rounds=2,
                initial_style="paper",
            )

        self.assertEqual(result["prompt"], "PAPER QUESTION")
        self.assertEqual(captured["debate_template"], FakeTask.paper_debate_template)
        self.assertIs(captured["other_answers_formatter"], methods.format_paper_others)


    def test_generate_initial_answers_uses_requested_style(self):
        seen = []

        def fake_chat(model, tokenizer, messages, **kwargs):
            seen.append(deepcopy(messages))
            return "answer A"

        with patch.object(methods, "chat", side_effect=fake_chat):
            result = methods.generate_initial_answers(
                None, None, FakeTask(), {}, None, 32, n_agents=2,
                initial_style="paper",
            )

        self.assertEqual(result["prompt"], "PAPER QUESTION")
        self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()

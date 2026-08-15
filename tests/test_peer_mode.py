"""The self-refine control: same rounds, same calls, no peer signal."""
import re
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


ROUND0 = ["answer from A", "answer from B", "answer from C"]


def run(peer_mode, **kwargs):
    prompts = []

    def fake_chat(model, tokenizer, messages, max_new_tokens, temperature,
                  enable_thinking):
        prompts.append(messages[-1]["content"])
        return "Final stance: neutral"

    with patch.object(debate, "chat", side_effect=fake_chat):
        trace = debate.run_debate(
            object(),
            FakeTokenizer(),
            "the article",
            debate_template="PEERS\n{others}",
            n_agents=3,
            n_rounds=2,
            initial_answers=list(ROUND0),
            peer_mode=peer_mode,
            **kwargs,
        )
    return trace, prompts


class PeerModeTest(unittest.TestCase):
    def test_self_shows_only_the_agent_its_own_answer(self):
        _, prompts = run("self", self_refine_template="SELF\n{others}")

        self.assertEqual(len(prompts), 3)
        for agent_index, prompt in enumerate(prompts):
            self.assertTrue(prompt.startswith("SELF"))
            own = ROUND0[agent_index]
            self.assertIn(own, prompt)
            for other in ROUND0:
                if other != own:
                    self.assertNotIn(other, prompt)

    def test_peers_still_shows_the_other_two(self):
        _, prompts = run("peers")

        for agent_index, prompt in enumerate(prompts):
            self.assertTrue(prompt.startswith("PEERS"))
            self.assertNotIn(ROUND0[agent_index], prompt)
            for other_index in range(3):
                if other_index != agent_index:
                    self.assertIn(ROUND0[other_index], prompt)

    def test_the_control_costs_the_same_as_the_debate(self):
        peers, _ = run("peers")
        selfref, _ = run("self", self_refine_template="SELF\n{others}")

        self.assertEqual(peers["call_counts"], selfref["call_counts"])

    def test_the_run_records_which_mode_it_was(self):
        trace, _ = run("self", self_refine_template="SELF\n{others}")
        self.assertEqual(trace["peer_mode"], "self")
        self.assertEqual(run("peers")[0]["peer_mode"], "peers")

    def test_self_without_a_template_is_refused(self):
        # the debate template would tell the agent its own answer came from
        # someone else, which is the confound this control removes
        with self.assertRaises(ValueError):
            run("self")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            run("telepathy")


class PeerThinkTest(unittest.TestCase):
    """A model can reason at length and still tell its peers only the answer."""

    def build(self, peer_think):
        seen = []

        def fake_chat(model, tokenizer, messages, max_new_tokens, temperature,
                      enable_thinking):
            seen.append(messages[-1]["content"])
            return "<think>long private reasoning</think>Final stance: neutral"

        # the module-level src.llm stub makes strip_think a no-op, and this
        # class is about exactly what it removes
        def real_strip(text):
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        with patch.object(debate, "chat", side_effect=fake_chat), \
                patch.object(debate, "strip_think", real_strip):
            trace = debate.run_debate(
                object(),
                FakeTokenizer(),
                "the article",
                debate_template="PEERS\n{others}",
                n_agents=3,
                n_rounds=2,
                peer_think=peer_think,
            )
        return trace, seen

    def test_strip_keeps_the_thinking_out_of_the_channel(self):
        trace, seen = self.build("strip")
        self.assertNotIn("long private reasoning", seen[-1])
        self.assertEqual(trace["answers_by_round"][0][0], "Final stance: neutral")
        self.assertIsNone(trace["peer_answers_by_round"])

    def test_keep_hands_the_peers_the_reasoning(self):
        trace, seen = self.build("keep")
        self.assertIn("long private reasoning", seen[-1])
        # what is stored and scored is unchanged either way
        self.assertEqual(trace["answers_by_round"][0][0], "Final stance: neutral")
        self.assertIn("long private reasoning",
                      trace["peer_answers_by_round"][0][0])

    def test_the_run_records_the_mode(self):
        self.assertEqual(self.build("keep")[0]["peer_think"], "keep")
        self.assertEqual(self.build("strip")[0]["peer_think"], "strip")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.build("telepathy")




class PeerContentTest(unittest.TestCase):
    """Cut the channel majority pressure runs on, rather than forbid its use."""

    def run_with(self, peer_content):
        prompts = []

        def fake_chat(model, tokenizer, messages, max_new_tokens, temperature,
                      enable_thinking):
            prompts.append(messages[-1]["content"])
            return "Evidence: some quoted wording\nFinal stance: neutral"

        with patch.object(debate, "chat", side_effect=fake_chat):
            trace = debate.run_debate(
                object(),
                FakeTokenizer(),
                "the article",
                debate_template="PEERS\n{others}",
                n_agents=3,
                n_rounds=3,
                peer_content=peer_content,
            )
        return trace, prompts

    def test_full_shows_the_answer_line(self):
        _, prompts = self.run_with("full")
        self.assertIn("Final stance: neutral", prompts[-1])

    def test_evidence_only_hides_it_and_keeps_the_evidence(self):
        trace, prompts = self.run_with("evidence_only")
        # round 2 peers come from round 1, which the fake model generated
        self.assertNotIn("Final stance", prompts[-1])
        self.assertIn("some quoted wording", prompts[-1])
        # what is stored and scored is untouched
        self.assertIn("Final stance: neutral", trace["answers_by_round"][-1][0])
        self.assertEqual(trace["peer_content"], "evidence_only")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_with("telepathy")


if __name__ == "__main__":
    unittest.main()

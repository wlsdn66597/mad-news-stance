"""Evidence-gated and round-specific exchanges, against the baseline."""
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
from src.prompts.stance import (  # noqa: E402
    STANCE_MINIMAL_EN,
    STANCE_MINIMAL_EN_PROTOCOLS,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


ROUND0 = ["answer from A", "answer from B", "answer from C"]


def run(n_rounds=4, **kwargs):
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
            debate_template="BASELINE\n{others}",
            n_agents=3,
            n_rounds=n_rounds,
            initial_answers=list(ROUND0),
            **kwargs,
        )
    # one prompt per agent per exchange round, grouped by round
    return trace, [prompts[i:i + 3] for i in range(0, len(prompts), 3)]


class DebateProtocolTest(unittest.TestCase):
    def test_baseline_repeats_one_template(self):
        _, rounds = run()
        self.assertEqual(len(rounds), 3)
        for round_prompts in rounds:
            for prompt in round_prompts:
                self.assertTrue(prompt.startswith("BASELINE"))

    def test_evidence_gated_repeats_its_own_template(self):
        gate = STANCE_MINIMAL_EN_PROTOCOLS["evidence_gated"]
        _, rounds = run(debate_templates=gate, debate_protocol="evidence_gated")

        self.assertEqual(len(rounds), 3)
        for round_prompts in rounds:
            for prompt in round_prompts:
                self.assertIn("only if another agent provides stronger", prompt)
                self.assertIn("[Agent 1]", prompt)
        # round 1 still carries the peers' round 0, two of the three of them
        for agent_index, prompt in enumerate(rounds[0]):
            self.assertNotIn(ROUND0[agent_index], prompt)
            self.assertEqual(2, sum(answer in prompt for answer in ROUND0))

    def test_round_specific_gives_each_round_its_own_task(self):
        steps = STANCE_MINIMAL_EN_PROTOCOLS["round_specific"]
        _, rounds = run(debate_templates=steps, debate_protocol="round_specific")

        self.assertEqual(len(rounds), 3)
        for prompt in rounds[0]:
            self.assertIn("check their claims against the original article", prompt)
        for prompt in rounds[1]:
            self.assertIn("test your current stance against", prompt)
        for prompt in rounds[2]:
            self.assertIn("Make your final judgment", prompt)

    def test_every_protocol_costs_what_the_baseline_costs(self):
        baseline, _ = run()
        for name, templates in STANCE_MINIMAL_EN_PROTOCOLS.items():
            trace, _ = run(debate_templates=templates, debate_protocol=name)
            self.assertEqual(trace["call_counts"], baseline["call_counts"], name)

    def test_the_run_records_which_protocol_it_was(self):
        steps = STANCE_MINIMAL_EN_PROTOCOLS["round_specific"]
        trace, _ = run(debate_templates=steps, debate_protocol="round_specific")
        self.assertEqual(trace["debate_protocol"], "round_specific")
        self.assertEqual(trace["round_templates"], list(steps))
        self.assertEqual(run()[0]["debate_protocol"], "baseline")

    def test_round_specific_refuses_the_wrong_round_count(self):
        steps = STANCE_MINIMAL_EN_PROTOCOLS["round_specific"]
        with self.assertRaises(ValueError):
            run(n_rounds=2, debate_templates=steps, debate_protocol="round_specific")

    def test_a_protocol_cannot_run_without_peers(self):
        gate = STANCE_MINIMAL_EN_PROTOCOLS["evidence_gated"]
        with self.assertRaises(ValueError):
            run(
                debate_templates=gate,
                debate_protocol="evidence_gated",
                peer_mode="self",
                self_refine_template="SELF\n{others}",
            )

    def test_protocols_keep_the_baseline_answer_format(self):
        for templates in STANCE_MINIMAL_EN_PROTOCOLS.values():
            for template in templates:
                self.assertIn("{others}", template)
                self.assertIn(
                    "Final stance: <supportive|oppositional|neutral>", template
                )

    def test_v2_drops_the_default_that_froze_the_gate(self):
        v1 = STANCE_MINIMAL_EN_PROTOCOLS["evidence_gated"][0]
        v2 = STANCE_MINIMAL_EN_PROTOCOLS["evidence_gated_v2"][0]
        self.assertIn("keep your previous stance", v1)
        self.assertNotIn("keep your previous stance", v2)
        self.assertIn("quote the one passage", v2)
        # the anti-conformity clause is the part that must survive
        self.assertIn("merely to agree", v2)

    def test_v2_says_what_neutral_is_in_every_round_that_decides(self):
        steps = STANCE_MINIMAL_EN_PROTOCOLS["round_specific_v2"]
        for template in steps[1:]:
            self.assertIn("Neutral is not a tie-breaker", template)
        for template in STANCE_MINIMAL_EN_PROTOCOLS["round_specific"]:
            self.assertNotIn("Neutral is not a tie-breaker", template)

    def test_v2_gives_the_first_round_something_to_produce(self):
        v1, v2 = (STANCE_MINIMAL_EN_PROTOCOLS[name][0]
                  for name in ("round_specific", "round_specific_v2"))
        self.assertNotIn("as written", v1)
        self.assertIn("Quote the single strongest passage", v2)
        self.assertIn("as written", v2)

    def test_reasoned_exchange_asks_last_and_steers_nothing(self):
        template = STANCE_MINIMAL_EN_PROTOCOLS["reasoned_exchange"][0]
        # the demand has to sit after the peers, where generation begins
        self.assertLess(template.index("{others}"), template.index("Evidence:"))
        self.assertLess(template.index("{others}"),
                        template.index("Final stance:"))
        # and it must not tell the agent which way to decide
        for steer in ("keep your previous stance", "merely to agree",
                      "Neutral is not a tie-breaker", "majority"):
            self.assertNotIn(steer, template)

    def test_the_profile_carries_the_protocols(self):
        self.assertEqual(
            STANCE_MINIMAL_EN.debate_protocols, STANCE_MINIMAL_EN_PROTOCOLS
        )


if __name__ == "__main__":
    unittest.main()

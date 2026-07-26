import unittest
from types import SimpleNamespace
import sys
import types

fake_torch = types.ModuleType("torch")
fake_torch.no_grad = lambda: (lambda function: function)
fake_torch.bfloat16 = object()
fake_torch.cuda = types.SimpleNamespace()
sys.modules.setdefault("torch", fake_torch)

fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoModelForCausalLM = object()
fake_transformers.AutoTokenizer = object()
fake_transformers.BitsAndBytesConfig = object()
sys.modules.setdefault("transformers", fake_transformers)


from src.llm import model_context_window, validate_context_budget


class ContextBudgetTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=40960)
        )
        self.tokenizer = SimpleNamespace(model_max_length=10**30)

    def test_uses_finite_model_context_window(self):
        self.assertEqual(model_context_window(self.model, self.tokenizer), 40960)

    def test_accepts_prompt_within_budget(self):
        self.assertEqual(
            validate_context_budget(self.model, self.tokenizer, 8000, 1024),
            40960,
        )

    def test_rejects_prompt_that_would_overflow(self):
        with self.assertRaisesRegex(ValueError, "input is not truncated"):
            validate_context_budget(self.model, self.tokenizer, 40500, 1024)


if __name__ == "__main__":
    unittest.main()

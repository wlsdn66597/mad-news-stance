import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_llm_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.cuda = types.SimpleNamespace(set_per_process_memory_fraction=Mock())
    fake_torch.no_grad = lambda: (lambda function: function)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=Mock())
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=Mock())
    fake_transformers.BitsAndBytesConfig = Mock()

    path = Path(__file__).resolve().parents[1] / "src" / "llm.py"
    spec = importlib.util.spec_from_file_location("exaone35_llm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        spec.loader.exec_module(module)
    return module


llm = load_llm_module()


class Exaone35CompatibilityTest(unittest.TestCase):
    def test_translates_new_mask_signature_and_restores_original(self):
        calls = []

        def create_causal_mask(config, inputs_embeds, attention_mask=None):
            calls.append((config, inputs_embeds, attention_mask))
            return "mask"

        masking_module = types.SimpleNamespace(create_causal_mask=create_causal_mask)
        state = llm._install_exaone35_causal_mask_compat(masking_module)

        result = masking_module.create_causal_mask(
            config="config",
            input_embeds="embeddings",
            attention_mask="attention",
            cache_position="removed",
        )

        self.assertEqual(result, "mask")
        self.assertEqual(calls, [("config", "embeddings", "attention")])
        llm._restore_causal_mask(state)
        self.assertIs(masking_module.create_causal_mask, create_causal_mask)

    def test_legacy_mask_signature_needs_no_patch(self):
        def create_causal_mask(config, input_embeds, cache_position=None):
            return config, input_embeds, cache_position

        masking_module = types.SimpleNamespace(create_causal_mask=create_causal_mask)
        state = llm._install_exaone35_causal_mask_compat(masking_module)

        self.assertIsNone(state)
        self.assertIs(masking_module.create_causal_mask, create_causal_mask)

    def test_patch_is_scoped_to_exaone35(self):
        tokenizer_loader = llm.AutoTokenizer.from_pretrained
        model_loader = llm.AutoModelForCausalLM.from_pretrained
        tokenizer_loader.reset_mock()
        model_loader.reset_mock()
        tokenizer_loader.return_value = Mock()
        model_loader.return_value = Mock()

        with patch.object(llm, "_install_exaone35_causal_mask_compat") as install_patch:
            with patch.object(llm, "_restore_causal_mask") as restore_patch:
                install_patch.return_value = "state"

                llm.load_model("Qwen/Qwen3-4B", load_in_4bit=False)
                install_patch.assert_not_called()
                restore_patch.assert_called_once_with(None)

                install_patch.reset_mock()
                restore_patch.reset_mock()
                llm.load_model(
                    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
                    load_in_4bit=False,
                    trust_remote_code=True,
                )
                install_patch.assert_called_once_with()
                restore_patch.assert_called_once_with("state")


if __name__ == "__main__":
    unittest.main()

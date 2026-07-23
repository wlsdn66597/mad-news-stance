import random
import sys
import types
import unittest
from unittest.mock import patch

if "datasets" not in sys.modules:
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = None
    sys.modules["datasets"] = datasets

from src.tasks.gsm8k import GSM8K
from src.tasks.mmlu import MMLU


class PaperSamplingTest(unittest.TestCase):
    def test_gsm8k_paper_uses_seeded_shuffle_then_prefix(self):
        dataset = [
            {"question": f"q{i}", "answer": f"work #### {i}"}
            for i in range(8)
        ]
        expected = list(range(len(dataset)))
        random.Random(0).shuffle(expected)

        with patch("src.tasks.gsm8k.load_dataset", return_value=dataset):
            items = GSM8K().load(n=4, seed=0, sampling_protocol="paper")

        self.assertEqual([item["source_index"] for item in items], expected[:4])
        self.assertEqual([item["gold"] for item in items], [str(i) for i in expected[:4]])

    def test_mmlu_paper_samples_subject_then_row_with_replacement(self):
        dataset = [
            {"subject": "zeta", "question": "z0", "choices": ["a", "b", "c", "d"], "answer": 0},
            {"subject": "alpha", "question": "a0", "choices": ["a", "b", "c", "d"], "answer": 1},
            {"subject": "zeta", "question": "z1", "choices": ["a", "b", "c", "d"], "answer": 2},
            {"subject": "alpha", "question": "a1", "choices": ["a", "b", "c", "d"], "answer": 3},
        ]
        by_subject = {"alpha": [1, 3], "zeta": [0, 2]}
        subjects = sorted(by_subject)
        rng = random.Random(7)
        expected = []
        for _ in range(10):
            subject = rng.choice(subjects)
            expected.append(rng.choice(by_subject[subject]))

        with patch("src.tasks.mmlu.load_dataset", return_value=dataset):
            items = MMLU().load(n=10, seed=7, sampling_protocol="paper")

        self.assertEqual([item["id"] for item in items], list(range(10)))
        self.assertEqual([item["source_index"] for item in items], expected)
        self.assertEqual([item["subject"] for item in items], [dataset[i]["subject"] for i in expected])

    def test_mmlu_uniform_preserves_legacy_global_shuffle(self):
        dataset = [
            {"subject": "s", "question": str(i), "choices": ["a", "b", "c", "d"], "answer": i % 4}
            for i in range(8)
        ]
        expected = list(range(len(dataset)))
        random.Random(3).shuffle(expected)

        with patch("src.tasks.mmlu.load_dataset", return_value=dataset):
            items = MMLU().load(n=5, seed=3, sampling_protocol="uniform")

        self.assertEqual([item["source_index"] for item in items], expected[:5])
        self.assertEqual([item["id"] for item in items], expected[:5])

    def test_unknown_protocol_is_rejected(self):
        with patch("src.tasks.gsm8k.load_dataset", return_value=[]):
            with self.assertRaises(ValueError):
                GSM8K().load(sampling_protocol="bad")
        with patch("src.tasks.mmlu.load_dataset", return_value=[]):
            with self.assertRaises(ValueError):
                MMLU().load(sampling_protocol="bad")


if __name__ == "__main__":
    unittest.main()

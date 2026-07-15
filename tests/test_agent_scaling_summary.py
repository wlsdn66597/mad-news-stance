import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_agent_scaling.py"
SPEC = importlib.util.spec_from_file_location("run_agent_scaling", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AgentScalingSummaryTest(unittest.TestCase):
    def test_summary_reports_accuracy_context_and_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.REPO_ROOT = root
            MODULE.RESULTS_DIR = root / "results" / "phase1"
            args = SimpleNamespace(
                tasks=["gsm8k"],
                model="qwen",
                methods=["majority", "debate_memory"],
                agent_counts=[5],
                n_rounds=2,
                n=2,
                data_seed=0,
                run_seed=2000,
                temperature=1.0,
                memory_max_new_tokens=256,
                memory_temperature=0.0,
                tag_prefix="test_memory",
            )
            result_path = MODULE.result_path(args, "gsm8k", 5)
            result_path.parent.mkdir(parents=True)
            trace = {
                "communication_mode": "shared_summary",
                "call_counts": {"total_generation_calls": 11},
                "communication_stats_by_round": [
                    {"round_index": 0},
                    {
                        "round_index": 1,
                        "direct_equivalent_tokens_per_agent": [400] * 5,
                        "delivered_tokens_per_agent": [120] * 5,
                        "per_agent_context_token_ratio": 0.3,
                        "total_input_token_ratio": 0.55,
                    },
                ],
            }
            result_path.write_text(
                json.dumps(
                    {
                        "majority": {
                            "1": {"correct": True},
                            "2": {"correct": False},
                        },
                        "debate_memory": {
                            "1": {"correct": True, "debate_trace": trace},
                            "2": {"correct": True, "debate_trace": trace},
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows = MODULE.summarize(args)

            memory = next(row for row in rows if row["method"] == "debate_memory")
            self.assertEqual(memory["accuracy"], 1.0)
            self.assertEqual(memory["mean_peak_delivered_tokens"], 120)
            self.assertEqual(memory["mean_per_agent_context_token_ratio"], 0.3)
            self.assertEqual(memory["mean_generation_calls_per_item"], 11)
            self.assertTrue(
                any(MODULE.RESULTS_DIR.glob("agent_scaling_summary_*.md"))
            )


if __name__ == "__main__":
    unittest.main()

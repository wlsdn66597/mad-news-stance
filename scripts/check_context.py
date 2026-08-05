"""Check whether a model's context window fits the stance debate prompts.

Downloads only the tokenizer and the config (no weights, no GPU), tokenizes
every prompt of the selected split, and projects the widest debate context.

    python scripts/check_context.py \
      --config config/phase2_exaone_stance_minimal_en.yaml \
      --model exaone --split test --n 1001
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context_check import project_budgets  # noqa: E402
from src.prompts.stance import PROFILES  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2.yaml")
    parser.add_argument("--model", default="exaone")
    parser.add_argument("--split", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--prompt-profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--output")
    return parser.parse_args()


def context_window(model_config, tokenizer):
    """Same rule as src.llm.model_context_window, without loading weights."""
    candidates = [
        value
        for value in (
            getattr(model_config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        )
        if isinstance(value, int) and 0 < value < 1_000_000
    ]
    return min(candidates) if candidates else None


def count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model_cfg = cfg["models"][args.model]
    model_id = model_cfg["id"]
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    profile = args.prompt_profile or cfg.get("prompt_profile", "legacy_ko")
    split = args.split or cfg["split"]
    n = args.n or cfg["n"]
    data_seed = cfg["seed"] if args.data_seed is None else args.data_seed
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    n_agents = args.n_agents or cfg["debate"]["n_agents"]
    n_rounds = args.n_rounds or cfg["debate"]["n_rounds"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    window = context_window(model_config, tokenizer)

    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=n, seed=data_seed)
    # The template is sent with the peer answers substituted in; count it empty
    # so the peer answers are not double counted.
    template_tokens = count(tokenizer, task.debate_template.format(others="", question=""))
    question_tokens = {item["id"]: count(tokenizer, task.question(item)) for item in items}

    report = project_budgets(
        question_tokens,
        max_new_tokens=max_new_tokens,
        context_window=window,
        n_agents=n_agents,
        template_tokens=template_tokens,
        n_rounds=n_rounds,
    )
    report["model_id"] = model_id
    report["prompt_profile"] = profile
    report["split"] = split
    report["chat_template_overhead_not_counted"] = True

    print(f"[model] {model_id}")
    print(f"[window] context_window={window} max_new_tokens={max_new_tokens} "
          f"agents={n_agents} rounds={n_rounds}")
    lengths = report["question_tokens"]
    print(f"[prompt tokens] min={lengths['min']} p50={lengths['p50']} "
          f"p95={lengths['p95']} max={lengths['max']} (n={report['items']})")
    peak = report["peak_required"]
    print(f"[peak required] p50={peak['p50']} p95={peak['p95']} max={peak['max']}")
    if window:
        print(f"[headroom] {report['headroom_at_worst_item']} tokens at the worst item")
    if report["fits"]:
        print("[ok] every item fits with the current settings")
    else:
        print(f"[over budget] {len(report['over_budget_items'])} items exceed the window")
        for row in report["over_budget_items"][:10]:
            print(f"  item={row['item_id']} prompt={row['question_tokens']} "
                  f"peak={row['peak_required']}")
        print(f"[hint] max_new_tokens <= {report['max_new_tokens_that_would_fit']} "
              "would fit the longest item")
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()

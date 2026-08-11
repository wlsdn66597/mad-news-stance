"""What does an agent actually write, before anything is stripped from it?

Saved runs are stripped: `run_debate` calls `strip_think` and stores the
result, so the file cannot say whether an agent reasoned inside a <think>
block that was then discarded before its peers and the judge ever saw it. On
test s6000, round-0 answers have a median of 24 characters -- the length of
"Final stance: supportive" -- and rounds 2 and 3 return to that after the one
round where the shared prompt produces prose. This prints the raw generation
next to the stripped one so the difference, if any, is visible.

    python scripts/probe_raw_answer.py --config config/phase2_exaone_stance_minimal_en.yaml
    python scripts/probe_raw_answer.py --n 5 --personas --round1

Reads the dataset and loads the model; writes nothing.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.debate import format_others  # noqa: E402
from src.llm import build_messages, chat, load_model, strip_think  # noqa: E402
from src.prompts.stance import stance_personas  # noqa: E402
from src.tasks.stance import Stance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase2_exaone_stance_minimal_en.yaml")
    parser.add_argument("--model", default="exaone")
    parser.add_argument("--n", type=int, default=3, help="items to probe")
    parser.add_argument("--split", default=None)
    parser.add_argument("--personas", action="store_true")
    parser.add_argument("--round1", action="store_true",
                        help="also probe one exchange round, fed the round-0 answers")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def show(label, raw):
    stripped = strip_think(raw)
    print(f"  [{label}] raw {len(raw)} chars, stripped {len(stripped)} chars"
          + ("   <- strip_think removed content" if len(stripped) != len(raw.strip()) else ""))
    print(f"    raw      : {raw[:400]!r}")
    if stripped != raw.strip():
        print(f"    stripped : {stripped[:400]!r}")
    return stripped


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["models"][args.model]
    profile = cfg.get("prompt_profile", "stance_minimal_en")
    split = args.split or cfg["split"]
    temperature = (args.temperature if args.temperature is not None
                   else cfg["sampling"]["temperature"])
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    n_agents = cfg["debate"]["n_agents"]

    task = Stance(cfg["data_path"], prompt_profile=profile)
    items = task.load(split=split, n=args.n, seed=cfg["seed"])
    prompts = (stance_personas(n_agents) if args.personas
               else [model_cfg.get("system_prompt")] * n_agents)

    print(f"config={args.config} profile={profile} split={split} "
          f"temperature={temperature} max_new_tokens={max_new_tokens} "
          f"personas={bool(args.personas)}")
    model, tokenizer = load_model(
        model_cfg["id"],
        load_in_4bit=bool(cfg["load_in_4bit"]),
        gpu_index=cfg["gpu_index"],
        mem_fraction=cfg["mem_fraction"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    for item in items:
        question = task.question(item, style="cot")
        print(f"\n===== item {item['id']}  gold={item.get('gold')}")
        contexts, answers = [], []
        for index, system_prompt in enumerate(prompts):
            messages = build_messages(question, system_prompt=system_prompt)
            raw = chat(model, tokenizer, messages, max_new_tokens=max_new_tokens,
                       temperature=temperature, enable_thinking=None)
            answers.append(show(f"round0 agent{index}", raw))
            messages.append({"role": "assistant", "content": answers[-1]})
            contexts.append(messages)

        if not args.round1:
            continue
        for index, messages in enumerate(contexts):
            others = [a for other, a in enumerate(answers) if other != index]
            messages.append({"role": "user", "content": task.debate_template.format(
                others=format_others(others), question=question)})
            raw = chat(model, tokenizer, messages, max_new_tokens=max_new_tokens,
                       temperature=temperature, enable_thinking=None)
            show(f"round1 agent{index}", raw)


if __name__ == "__main__":
    main()

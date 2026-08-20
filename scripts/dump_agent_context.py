"""One item's exact agent contexts, for a slide or a figure. Offline, no GPU.

`debate_trace.agent_contexts` stores the message list each agent actually
received -- system prompt, the article and task, its own round-0 answer, the
peer answers it was then shown, its revision. Retyping any of that onto a slide
invites a transcription error in the one place a reader will read closely, so
this prints it verbatim, split by role, with the article trimmed to whatever
length fits.

Picking the item matters as much as printing it. A slide is illustrating a
mechanism, so the useful item is one where the mechanism ran: the agents
disagreed at round 0 and the exchange landed on the gold label. That is
`--pick split_then_correct`, the default, and it needs no second file. `--pick
fixed` narrows it further to items a baseline got wrong, which needs
`--before`.

For the opposite case -- what an exchange destroys -- `--pick gold_lost` finds
items where an agent named the gold label at round 0 and none still held it at
the end, and `--pick unanimous_wrong` finds the ones the agents agreed on and
got wrong. Under the published template those are the same failure seen twice:
87% of what an agent hands its peers is a label line, so the exchange carries
votes rather than reasons and a correct minority is outnumbered rather than
answered.

    python scripts/dump_agent_context.py results/phase2/<run>.json
    python scripts/dump_agent_context.py results/phase2/<run>.json --item 412 --agent 0
    python scripts/dump_agent_context.py results/phase2/<run>.json \
        --before results/phase2/<majority run>.json --pick fixed --article-chars 400
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import LABELS, parse_stance  # noqa: E402

PICKS = ("split_then_correct", "fixed", "gold_lost", "unanimous_wrong", "any")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="saved phase2 result with a debate trace")
    parser.add_argument("--method", default="debate")
    parser.add_argument("--before", help="a second result, for --pick fixed")
    parser.add_argument("--before-method", default="majority")
    parser.add_argument("--item", help="item id; overrides --pick")
    parser.add_argument("--pick", default="split_then_correct", choices=PICKS)
    parser.add_argument("--agent", type=int, help="only this agent index")
    parser.add_argument(
        "--article-chars", type=int, default=0,
        help="trim the article inside the round-0 user message to this many "
             "characters (0 keeps it whole). The trim is marked in the output.",
    )
    parser.add_argument(
        "--candidates", type=int, default=8,
        help="how many matching item ids to list before printing one",
    )
    return parser.parse_args()


def load_block(path, method):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data.get(method)
    if not isinstance(block, dict) or not block:
        raise SystemExit(f"{path} has no '{method}' block")
    return block


def rounds_of(item):
    trace = item.get("debate_trace") or {}
    return trace.get("answers_by_round") or [item.get("raw") or []]


def round_labels(item, index):
    rounds = rounds_of(item)
    if index >= len(rounds):
        return []
    return [parse_stance(answer) for answer in rounds[index]]


def matching(block, before, pick):
    """Item ids where the mechanism a slide is illustrating actually ran."""
    keys = []
    for key, item in block.items():
        gold, pred = item.get("gold"), item.get("pred")
        if gold not in LABELS:
            continue
        first = round_labels(item, 0)
        if pick == "any":
            keys.append(key)
        elif pick == "split_then_correct":
            if len(set(first)) > 1 and pred == gold:
                keys.append(key)
        elif pick == "gold_lost":
            # someone had the answer at round 0 and the exchange talked it away.
            # This is the failure the published template produces: 87% of what
            # an agent hands its peers is a label line, so the exchange carries
            # votes rather than reasons and the minority is outnumbered rather
            # than answered.
            final = round_labels(item, len(rounds_of(item)) - 1)
            if gold in first and gold not in final:
                keys.append(key)
        elif pick == "unanimous_wrong":
            final = round_labels(item, len(rounds_of(item)) - 1)
            if final and len(set(final)) == 1 and pred != gold:
                keys.append(key)
        elif pick == "fixed":
            was = (before.get(key) or {}).get("pred")
            if was is not None and was != gold and pred == gold:
                keys.append(key)
    return sorted(keys, key=lambda k: (len(str(k)), str(k)))


def trim_article(text, limit):
    if not limit or len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} characters trimmed for the slide]"


def main():
    args = parse_args()
    block = load_block(args.result, args.method)
    before = load_block(args.before, args.before_method) if args.before else {}
    if args.pick == "fixed" and not before:
        raise SystemExit("--pick fixed needs --before, the run being improved on")

    if args.item is not None:
        key = str(args.item)
        if key not in block:
            raise SystemExit(f"item {key} is not in the {args.method} block")
        keys = [key]
    else:
        keys = matching(block, before, args.pick)
        if not keys:
            raise SystemExit(f"no item matches --pick {args.pick}")
        print(f"[{args.pick}] {len(keys)} items; showing the first. Others: "
              + " ".join(str(k) for k in keys[1:1 + args.candidates]))

    key = str(keys[0])
    item = block[key]
    trace = item.get("debate_trace") or {}
    contexts = trace.get("agent_contexts")
    if not contexts:
        raise SystemExit(
            f"item {key} has no debate_trace.agent_contexts; this run predates "
            "the field, so the contexts would have to be rebuilt rather than read"
        )

    print(f"\nitem {key}   gold={item.get('gold')}   final={item.get('pred')}"
          + (f"   before={(before.get(key) or {}).get('pred')}" if before else ""))
    print(f"protocol={trace.get('debate_protocol')}   rounds={trace.get('n_rounds')}   "
          f"agents={trace.get('n_agents')}   peer_mode={trace.get('peer_mode')}")
    for index, answers in enumerate(rounds_of(item)):
        print(f"  round {index}: " + " ".join(str(parse_stance(a)) for a in answers))

    for agent_index, messages in enumerate(contexts):
        if args.agent is not None and agent_index != args.agent:
            continue
        print(f"\n{'=' * 78}\nAGENT {agent_index}\n{'=' * 78}")
        # the first user turn carries the article; every later one is a peer
        # exchange, which is the box a slide usually wants side by side with it
        user_turns = 0
        for message in messages:
            role = message.get("role", "?")
            content = str(message.get("content", ""))
            if role == "user":
                user_turns += 1
                if user_turns == 1:
                    content = trim_article(content, args.article_chars)
                    label = "user  (article + task)"
                else:
                    label = f"user  (peer answers, round {user_turns - 1})"
            elif role == "assistant":
                label = f"assistant  (round {user_turns - 1})"
            else:
                label = role
            print(f"\n[{label}]\n{content}")


if __name__ == "__main__":
    main()

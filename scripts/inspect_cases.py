"""What the agents actually said, and whether their quotes are in the article.

Offline. The tables say a peer exchange gains items; they cannot say what a peer
said that changed a mind, or whether the passage it quoted exists. Two things
here, both from saved traces:

**Quote grounding.** Every protocol that asks for evidence produces an
"Evidence:" line. This checks each one against the article it claims to quote,
after normalising whitespace and quotation marks, and reports the share that is
actually there. The advocacy diagnostic found 81% of its Korean quotes absent
from the article, so this is not a formality -- and a quote that is not in the
article is what the peers are being persuaded by.

**Case sampling.** Items are bucketed by what the exchange did to them, and the
transcripts are printed with the article so a paragraph of the paper can quote
one. The buckets are the ones the argument turns on:

    wrong_to_correct   the debate fixed it
    correct_to_wrong   the debate broke it
    consensus_error    all three agreed, all three wrong
    neutral_missed     gold is neutral, the system said otherwise

    python scripts/inspect_cases.py --before <majority>.json --after <debate>.json
    python scripts/inspect_cases.py --after <debate>.json --bucket consensus_error -n 5
"""
import argparse
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consensus import parse_stance  # noqa: E402

BUCKETS = ("wrong_to_correct", "correct_to_wrong", "consensus_error", "neutral_missed")
EVIDENCE_PREFIXES = ("evidence", "new from others", "decisive", "근거")
GROUNDED_RATIO = 0.8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", required=True, help="the run being inspected")
    parser.add_argument("--before", help="the run it is compared against, for the flip buckets")
    parser.add_argument("--method", default="debate")
    parser.add_argument("--before-method", default="majority")
    parser.add_argument("--data-path", default="data/k-news-stance_nosegment.json")
    parser.add_argument("--bucket", choices=BUCKETS, help="print only this bucket")
    parser.add_argument("-n", "--cases", type=int, default=3, help="cases per bucket")
    parser.add_argument("--chars", type=int, default=700, help="article characters to show")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cases", action="store_true", help="grounding audit only")
    return parser.parse_args()


def normalise(text):
    text = re.sub(r"[\"'“”‘’`]", "", str(text))
    return re.sub(r"\s+", "", text)


def evidence_lines(answer):
    return [
        line.split(":", 1)[1].strip()
        for line in str(answer).splitlines()
        if ":" in line and line.strip().lower().startswith(EVIDENCE_PREFIXES)
    ]


def grounded(quote, article):
    """Is the quote in the article, allowing for whitespace and quote marks?"""
    q, a = normalise(quote), normalise(article)
    if len(q) < 8:
        return None  # too short to judge; NONE and one-word answers land here
    if q in a:
        return True
    match = SequenceMatcher(None, q, a).find_longest_match(0, len(q), 0, len(a))
    return match.size / len(q) >= GROUNDED_RATIO


def load_runs(path, method):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = data.get(method)
    if not isinstance(block, dict) or not block:
        raise SystemExit(f"{path} has no '{method}' block")
    return block


def rounds_of(item):
    trace = item.get("debate_trace") or {}
    return trace.get("answers_by_round") or [item.get("raw") or []]


def main():
    args = parse_args()
    after = load_runs(args.after, args.method)
    before = load_runs(args.before, args.before_method) if args.before else {}
    articles = {
        str(row["id"]): row
        for row in json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    }

    # ---- quote grounding ----------------------------------------------------
    print(f"[grounding] Evidence lines checked against the article they quote "
          f"(>= {GROUNDED_RATIO:.0%} of the quote found contiguously)")
    print(f"  {'round':>5} {'lines':>7} {'grounded':>9} {'not found':>10} {'too short':>10}")
    n_rounds = max(len(rounds_of(v)) for v in after.values())
    for index in range(n_rounds):
        found = missing = short = 0
        for key, item in after.items():
            article = (articles.get(str(key)) or {}).get("article", "")
            rounds = rounds_of(item)
            if index >= len(rounds) or not article:
                continue
            for answer in rounds[index]:
                for quote in evidence_lines(answer):
                    verdict = grounded(quote, article)
                    if verdict is None:
                        short += 1
                    elif verdict:
                        found += 1
                    else:
                        missing += 1
        total = found + missing + short
        if not total:
            print(f"  {index:>5} {'-':>7} {'(no evidence lines)':>32}")
            continue
        print(f"  {index:>5} {total:7d} {found / total:8.1%} {missing / total:9.1%} "
              f"{short / total:9.1%}")
    print("  a quote that is not in the article is what the peers are being "
          "persuaded by, so this bounds what the exchange can be worth")

    if args.no_cases:
        return

    # ---- buckets ------------------------------------------------------------
    shared = sorted(set(after) & set(before)) if before else sorted(after)
    buckets = {name: [] for name in BUCKETS}
    for key in shared:
        item = after[key]
        gold, pred = item.get("gold"), item.get("pred")
        final = [parse_stance(a) for a in rounds_of(item)[-1]]
        if before:
            was = before[key].get("pred")
            if was != gold and pred == gold:
                buckets["wrong_to_correct"].append(key)
            elif was == gold and pred != gold:
                buckets["correct_to_wrong"].append(key)
        if len(set(final)) == 1 and pred != gold:
            buckets["consensus_error"].append(key)
        if gold == "neutral" and pred != gold:
            buckets["neutral_missed"].append(key)

    print()
    print("[buckets] " + "  ".join(
        f"{name}={len(keys)}" for name, keys in buckets.items() if keys or not args.before
    ))

    rng = random.Random(args.seed)
    wanted = [args.bucket] if args.bucket else list(BUCKETS)
    for name in wanted:
        keys = buckets[name]
        if not keys:
            continue
        print(f"\n{'=' * 78}\n[{name}] {len(keys)} items, showing {min(args.cases, len(keys))}")
        for key in rng.sample(keys, min(args.cases, len(keys))):
            item = after[key]
            row = articles.get(str(key), {})
            print(f"\n{'-' * 78}")
            print(f"item {key}   gold={item.get('gold')}   final={item.get('pred')}"
                  + (f"   before={before[key].get('pred')}" if before else ""))
            print(f"issue: {row.get('issue', '')}")
            print(f"headline: {row.get('headline') or row.get('haedline', '')}")
            print(f"article[:{args.chars}]: {str(row.get('article', ''))[:args.chars]}")
            for index, answers in enumerate(rounds_of(item)):
                print(f"  -- round {index}")
                for agent, answer in enumerate(answers):
                    marks = "".join(
                        "o" if grounded(q, row.get("article", "")) else
                        "-" if grounded(q, row.get("article", "")) is None else "x"
                        for q in evidence_lines(answer)
                    )
                    print(f"    a{agent}{' [' + marks + ']' if marks else ''}: "
                          f"{str(answer).strip()[:400]}")

    print("\no = quote found in the article, x = not found, - = too short to judge")


if __name__ == "__main__":
    main()

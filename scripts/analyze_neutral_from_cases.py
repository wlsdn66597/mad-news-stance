"""Can the two polar cases tell you the article is neutral? Offline. No GPU.

A polar debate with abstention only works if the two advocates' relative
strength carries the neutral decision: both sides weak means the article takes
no side, one side clearly stronger means it does. The saved three-label
advocacy run already contains a supportive case and an oppositional case for
every item, so that premise can be tested before any of it is built.

Each feature below is computed on the two polar cases and scored by AUC against
"gold is neutral" -- the probability that a randomly chosen neutral article
ranks above a randomly chosen non-neutral one. AUC needs no threshold, so
nothing is fitted to this split. For reference, the model's own neutral
detection on this run reaches precision 0.458 at recall 0.245 (EXAONE) and
precision 0.469 at recall 0.661 (Qwen); an AUC near 0.5 means these cases carry
less than that, and the design has no signal to run on.

    python scripts/analyze_neutral_from_cases.py results/advocacy/<run>.items.json \
      --data-path data/k-news-stance_nosegment.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUOTE_PATTERN = re.compile(r"[\"“”'‘’「『]([^\"“”'‘’「』」]{6,200})[\"“”'‘’」』]")
HEDGES = (
    "little support", "does not support", "no explicit", "weak", "not strongly",
    "hardly", "insufficient", "difficult to argue", "does not clearly",
    "cannot be said", "at best", "arguably", "however", "although", "on the other hand",
    "근거가 부족", "보기 어렵", "단정하기", "그러나", "다만",
)
POLAR = ("supportive", "oppositional")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("items", help="an advocacy .items.json with per-label cases")
    parser.add_argument("--round", default="0",
                        help="'saved' for the cases the judge read, or a round index; "
                             "round 0 is where the surface signal lives")
    parser.add_argument("--data-path", help="dataset, to verify quotes against the article")
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def normalize(text):
    return " ".join(str(text or "").split())


def case_strength(text, article=None):
    """How well this advocate did, by the surface measures that ranked gold."""
    lowered = (text or "").lower()
    quotes = QUOTE_PATTERN.findall(text or "")
    verified = (
        sum(1 for q in quotes if normalize(q) in article) if article is not None else 0
    )
    return {
        "hedges": float(sum(lowered.count(cue) for cue in HEDGES)),
        "quotes": float(len(quotes)),
        "verified_quotes": float(verified),
        "chars": float(len(text or "")),
    }


def auc(scores, positive):
    """Mann-Whitney AUC, ties counted as half."""
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return None
    wins = ties = 0
    neg_sorted = sorted(neg)
    import bisect
    for value in pos:
        wins += bisect.bisect_left(neg_sorted, value)
        ties += bisect.bisect_right(neg_sorted, value) - bisect.bisect_left(neg_sorted, value)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    args = parse_args()
    rows = json.loads(Path(args.items).read_text(encoding="utf-8"))
    articles = {}
    if args.data_path:
        for row in json.loads(Path(args.data_path).read_text(encoding="utf-8")):
            articles[str(row["id"])] = normalize(
                f"{row.get('haedline') or row.get('headline', '')} {row.get('article', '')}"
            )

    samples = []
    for row in rows:
        if args.round == "saved":
            cases = {c["stance"]: c["analysis"] for c in row["advocate_cases"]}
        else:
            analyses = row["analyses_by_round"][int(args.round)]
            cases = {s: analyses[i] for i, s in enumerate(row["assigned_stances"])}
        if not set(POLAR) <= set(cases):
            continue
        article = articles.get(str(row["item_id"]))
        s = case_strength(cases["supportive"], article)
        o = case_strength(cases["oppositional"], article)
        samples.append({"gold": row["gold"], "supportive": s, "oppositional": o})

    if not samples:
        raise SystemExit("no item carried both polar cases")
    neutral = [sample["gold"] == "neutral" for sample in samples]
    print(f"[items] {len(samples)}   neutral {sum(neutral)}  "
          f"non-neutral {len(samples) - sum(neutral)}   (round={args.round})")

    # both-weak means neutral; one-side-clearly-stronger means it is not
    derived = {
        "both hedge a lot (min hedges)": lambda s, o: min(s["hedges"], o["hedges"]),
        "both hedge a lot (total hedges)": lambda s, o: s["hedges"] + o["hedges"],
        "sides are evenly matched (|hedge gap|, inverted)":
            lambda s, o: -abs(s["hedges"] - o["hedges"]),
        "neither cites much (min quotes, inverted)":
            lambda s, o: -min(s["quotes"], o["quotes"]),
        "neither verifies much (min verified, inverted)":
            lambda s, o: -min(s["verified_quotes"], o["verified_quotes"]),
        "evenly matched on verified quotes (|gap|, inverted)":
            lambda s, o: -abs(s["verified_quotes"] - o["verified_quotes"]),
        "both cases are short (min chars, inverted)":
            lambda s, o: -min(s["chars"], o["chars"]),
    }

    print(f"\n  {'signal':52} {'AUC':>6}")
    report = {"items": len(samples), "neutral": sum(neutral), "auc": {}}
    for name, fn in derived.items():
        value = auc([fn(x["supportive"], x["oppositional"]) for x in samples], neutral)
        if value is None:
            continue
        print(f"  {name:52} {value:6.3f}")
        report["auc"][name] = value

    best = max(report["auc"].values()) if report["auc"] else 0.5
    print(f"\n0.500 is chance. The model's own neutral detection on the debate run "
          f"sits near precision 0.46; a signal below about 0.55 AUC here is weaker "
          f"than what the model already does unaided, and the abstention design "
          f"would have nothing to decide on.")
    print(f"best signal: {best:.3f}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

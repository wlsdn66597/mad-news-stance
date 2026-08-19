"""What the rebuttal round does to the advocacy cases. Offline, no GPU.

Judging round 0 instead of round 1 is worth +19 items (p = 0.016), and the
offline oracle says the same independently: ranking the three cases by hedging
picks the gold case 42.4% of the time at round 0 and 32.8% at round 1, quoted
passage density goes from p = 0.0053 to p = 0.84, and mentions of "Analyst"
decide 12 items at round 0 against 815 at round 1. That establishes *that* the
exchange destroys the signal. It does not say how.

`advocacy_oracle.py` answers "are the cases separable" one round at a time.
This script answers "what changed between the rounds", from the same saved
`.items.json`, and reports the four mechanisms that could produce the loss:

- **the ballot.** Advocacy's whole justification is that the gold label is
  argued on every item. That holds by construction at round 0 only. If the
  advocate assigned gold concedes during the rebuttal, gold leaves the ballot
  and the loss is in the candidate set rather than in the judge. Concession is
  read the strict way the rest of the repository reads it -- only an explicit
  "Final stance:" line naming another label counts, because an advocate that
  discusses counter-evidence names other labels constantly. The gold advocate's
  hold rate is compared against the other two advocates' on the same items, so
  a general drop in compliance is not mistaken for a directional one.
- **homogenisation.** If the three cases converge on each other's wording, no
  judge can separate them whatever it reads. This is the same quantity that
  made `single` and `majority k=3` identical for the free-form ensemble, asked
  of the advocacy cases: pairwise token overlap, shared quoted passages, and
  verbatim sentence reuse, per round.
- **grounding.** With `--data-path`, how much of each case is still traceable
  to the article: quote verification at three strictnesses, the share of the
  case's own character shingles that appear in the article, and how often it
  addresses a peer rather than the text.
- **truncation.** Round 1 outputs are shorter than round 0 (1187 vs 1489
  chars). That reads as convergence, but a round-1 prompt carries the article,
  the agent's own turn and two peer cases, so it could equally be the article
  being pushed out of the context or the answer being cut off. Cases that end
  without sentence-final punctuation are counted as a proxy. It is only a
  proxy: the real check needs the tokenizer and lives in
  `scripts/check_context.py`.

`analyze_information_flow.py` asks the coverage / retention / selection question
of a phase2 debate trace and cannot read an advocacy `.items.json`; the ballot
section here is the same question asked of the advocacy cases, in the same
words, so the two can sit in one table. `inspect_cases.py` audits `Evidence:`
lines in a debate trace; the grounding section audits advocacy cases, which have
no such line.

Everything is computed from `analyses_by_round`, which is saved in full, so no
model runs and a rejudge file works as well as the original.

    python scripts/analyze_advocacy_rounds.py results/advocacy/<run>.items.json
    python scripts/analyze_advocacy_rounds.py results/advocacy/<run>.items.json \
        --data-path data/k-news-stance_nosegment.json --json out.json
"""
import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# the surface-feature vocabulary is the oracle's, deliberately: the two scripts
# must call the same thing a quote and the same thing a hedge, or their numbers
# cannot be quoted in one paragraph.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from advocacy_oracle import (  # noqa: E402
    HEDGES,
    NON_CONTENT,
    PEER_PATTERN,
    QUOTE_PATTERN,
    articles_by_id,
    squeeze,
)

from src.advocacy import position_bias_chi_square, stated_label  # noqa: E402
from src.consensus import LABELS, mcnemar_exact  # noqa: E402

SENTENCE_END = ".!?…\"'”’)]}」』。"
DECLARATION = re.compile(r"^\s*final\s*stance\s*[:=\-]", re.IGNORECASE)
# a quoted passage can be as short as six characters, so a twelve-character
# shingle cannot land inside one and the overlap reads zero on cases that quote
# accurately. Eight characters is long enough that agreement is not chance in
# either script, and every fourth position is sampled to keep the pass cheap.
SHINGLE = 8
SHINGLE_STEP = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("items", nargs="+", help="one or more advocacy .items.json")
    parser.add_argument(
        "--data-path",
        help="the dataset, so the cases can be checked against the article. "
             "Without it the grounding section is skipped.",
    )
    parser.add_argument(
        "--min-sentence", type=int, default=20,
        help="content characters a sentence needs before verbatim reuse of it "
             "counts as reuse rather than as a shared stock phrase (default 20)",
    )
    parser.add_argument("--json", dest="json_out")
    return parser.parse_args()


def two_proportion_p(hits_a, n_a, hits_b, n_b):
    """Two-sided p for two independent proportions, normal approximation.

    Used only for the gold-advocate versus other-advocate gap, where the two
    groups are different agents and a paired test does not apply. The counts
    there are in the hundreds, so the approximation is not the weak link.
    """
    if n_a == 0 or n_b == 0:
        return None
    pooled = (hits_a + hits_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return 1.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (hits_a / n_a - hits_b / n_b) / se
    return math.erfc(abs(z) / math.sqrt(2))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def tokens_of(text):
    """Content tokens, for overlap between two cases."""
    return {
        token for token in NON_CONTENT.split((text or "").lower())
        if len(token) >= 2
    }


def sentences_of(text, minimum):
    """Squeezed sentences long enough that repeating one is a real reuse."""
    parts, current = [], []
    for char in str(text or ""):
        current.append(char)
        if char in ".!?\n。":
            parts.append("".join(current))
            current = []
    parts.append("".join(current))
    return {tight for tight in (squeeze(part) for part in parts) if len(tight) >= minimum}


def jaccard(left, right):
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union) if union else None


def reuse(left, right):
    """Share of either case's sentences that appear verbatim in the other."""
    total = len(left) + len(right)
    if total == 0:
        return None
    shared = len(left & right)
    return (2 * shared) / total


def shingles(text):
    tight = squeeze(text)
    return [tight[i:i + SHINGLE] for i in range(0, max(0, len(tight) - SHINGLE + 1), SHINGLE_STEP)]


def looks_unfinished(text):
    """A case that stops without closing a sentence.

    A proxy for hitting `max_new_tokens`, not a measurement of it. Cheap enough
    to report on every round, and it is the difference between "the rebuttal
    made the agents concise" and "the rebuttal ran out of budget".

    The closing "Final stance: <label>" line carries no punctuation, so testing
    the raw last character calls every compliant case truncated -- it flagged
    100% of a fixture where nothing was cut. Declaration lines are dropped
    first, and a case that is nothing but a declaration is complete rather than
    truncated: a bare verdict is a different failure, counted by the judge
    diagnostics.
    """
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    while lines and DECLARATION.match(lines[-1]):
        lines.pop()
    if not lines:
        return False
    stripped = lines[-1].rstrip()
    return bool(stripped) and stripped[-1] not in SENTENCE_END


def held_flags(analyses, stances):
    """Per agent: does its text still argue the side it was assigned?

    Returns `(held, argued_label)`. `argued_label` is the assigned stance
    unless the agent explicitly declared another one, in which case the case
    has moved to that label and the ballot has to be read accordingly.
    """
    result = []
    for text, stance in zip(analyses, stances):
        label, source = stated_label(text)
        conceded = source == "marker" and label != stance
        result.append((not conceded, label if conceded else stance))
    return result


def usable_rows(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("items", [])
    kept = [
        row for row in rows
        if row.get("gold") in LABELS
        and row.get("analyses_by_round")
        and row.get("assigned_stances")
    ]
    if not kept:
        raise SystemExit(f"{path} has no rows with a gold label and saved rounds")
    widths = {len(row["analyses_by_round"]) for row in kept}
    if len(widths) > 1:
        raise SystemExit(
            f"{path} mixes runs of different round counts {sorted(widths)}; "
            "the per-round tables would compare different experiments"
        )
    return kept


def judged_round_of(row):
    saved = row.get("judged_round")
    if isinstance(saved, int):
        return saved
    return len(row["analyses_by_round"]) - 1


def round_report(rows, round_index, articles, min_sentence):
    """Everything measurable about one round's three cases."""
    chars, quotes, hedges, peers, unfinished = [], [], [], [], []
    held_gold = held_gold_n = held_other = held_other_n = 0
    ballot_gold = 0
    ballot_size = Counter()
    token_overlap, quote_overlap, sentence_reuse = [], [], []
    verified = Counter()
    shingle_hits = []

    for row in rows:
        analyses = row["analyses_by_round"][round_index]
        stances = row["assigned_stances"]
        gold = row["gold"]
        flags = held_flags(analyses, stances)

        for text in analyses:
            lowered = (text or "").lower()
            chars.append(float(len(text or "")))
            quotes.append(float(len(QUOTE_PATTERN.findall(text or ""))))
            hedges.append(float(sum(lowered.count(cue) for cue in HEDGES)))
            peers.append(float(len(PEER_PATTERN.findall(text or ""))))
            unfinished.append(looks_unfinished(text))

        for (held, _), stance in zip(flags, stances):
            if stance == gold:
                held_gold_n += 1
                held_gold += held
            else:
                held_other_n += 1
                held_other += held

        argued = {label for _, label in flags}
        ballot_gold += gold in argued
        ballot_size[len(argued)] += 1

        token_sets = [tokens_of(text) for text in analyses]
        quote_sets = [
            {squeeze(q) for q in QUOTE_PATTERN.findall(text or "")} for text in analyses
        ]
        sentence_sets = [sentences_of(text, min_sentence) for text in analyses]
        for i, j in itertools.combinations(range(len(analyses)), 2):
            token_overlap.append(jaccard(token_sets[i], token_sets[j]))
            quote_overlap.append(jaccard(quote_sets[i], quote_sets[j]))
            sentence_reuse.append(reuse(sentence_sets[i], sentence_sets[j]))

        article = articles.get(str(row["item_id"])) if articles else None
        if article is not None:
            tight_article = squeeze(article)
            for text in analyses:
                for quote in QUOTE_PATTERN.findall(text or ""):
                    tight = squeeze(quote)
                    verified["quotes"] += 1
                    verified["exact"] += quote.strip() in article
                    verified["loose"] += tight in tight_article
                    verified["prefix"] += bool(tight[:12]) and tight[:12] in tight_article
                pieces = shingles(text)
                if pieces:
                    shingle_hits.append(
                        sum(1 for piece in pieces if piece in tight_article) / len(pieces)
                    )

    peer_total = sum(peers)
    quote_total = sum(quotes)
    return {
        "round": round_index,
        "cases": len(chars),
        "chars": mean(chars),
        "quotes": mean(quotes),
        "hedges": mean(hedges),
        "peer_mentions": mean(peers),
        "peer_to_quote": (peer_total / quote_total) if quote_total else None,
        "unfinished_rate": sum(unfinished) / len(unfinished) if unfinished else None,
        "gold_advocate_held": held_gold,
        "gold_advocate_n": held_gold_n,
        "other_advocate_held": held_other,
        "other_advocate_n": held_other_n,
        "hold_gap_p": two_proportion_p(held_gold, held_gold_n, held_other, held_other_n),
        "ballot_has_gold": ballot_gold,
        "ballot_sizes": dict(ballot_size),
        "token_overlap": mean(token_overlap),
        "quote_overlap": mean(quote_overlap),
        "sentence_reuse": mean(sentence_reuse),
        "quote_verification": (
            None if not verified["quotes"] else {
                "quotes": verified["quotes"],
                "exact": verified["exact"] / verified["quotes"],
                "loose": verified["loose"] / verified["quotes"],
                "prefix12": verified["prefix"] / verified["quotes"],
            }
        ),
        "article_shingle_overlap": mean(shingle_hits) if shingle_hits else None,
    }


def ballot_paired(rows, first, last):
    """Did the same gold advocates hold in both rounds?

    Unpaired rates hide the direction: a rate that falls two points could be
    twenty advocates conceding and eighteen recanting. `mcnemar_exact` is
    reused with a constant gold of True, so `baseline == gold` reads as "held
    at round `first`".
    """
    before, after = [], []
    for row in rows:
        stances = row["assigned_stances"]
        gold = row["gold"]
        if gold not in stances:
            continue
        agent = stances.index(gold)
        before.append(held_flags(row["analyses_by_round"][first], stances)[agent][0])
        after.append(held_flags(row["analyses_by_round"][last], stances)[agent][0])
    golds = [True] * len(before)
    test = mcnemar_exact(before, after, golds)
    return {
        "items": len(before),
        "held_first": sum(before),
        "held_last": sum(after),
        "conceded_during_exchange": sum(1 for b, a in zip(before, after) if b and not a),
        "recanted_during_exchange": sum(1 for b, a in zip(before, after) if a and not b),
        "p_value_two_sided": test["p_value_two_sided"],
    }


def judge_report(rows):
    """What the judge did, split by the state of the case it read."""
    judged = [row for row in rows if row.get("pred_source") == "judge" and row.get("pred")]
    if not judged:
        return None
    buckets = {True: [0, 0], False: [0, 0]}   # gold advocate held -> [correct, n]
    picked_conceded = [0, 0]                  # judge chose a case that had conceded
    ballot_missing = [0, 0]
    positions = Counter()
    for row in judged:
        stances = row["assigned_stances"]
        gold = row["gold"]
        analyses = row["analyses_by_round"][judged_round_of(row)]
        flags = held_flags(analyses, stances)
        correct = row["pred"] == gold

        if gold in stances:
            held = flags[stances.index(gold)][0]
            buckets[held][0] += correct
            buckets[held][1] += 1

        if row["pred"] in stances:
            picked = flags[stances.index(row["pred"])][0]
            if not picked:
                picked_conceded[0] += correct
                picked_conceded[1] += 1

        if gold not in {label for _, label in flags}:
            ballot_missing[0] += correct
            ballot_missing[1] += 1

        for entry in row.get("candidate_order") or []:
            if entry.get("stance_argued") == row["pred"]:
                positions[entry.get("candidate_id")] += 1
                break

    return {
        "judged_items": len(judged),
        "accuracy": sum(row["pred"] == row["gold"] for row in judged) / len(judged),
        "gold_advocate_held": {
            "held": {"correct": buckets[True][0], "items": buckets[True][1]},
            "conceded": {"correct": buckets[False][0], "items": buckets[False][1]},
        },
        "judge_picked_a_conceded_case": {
            "correct": picked_conceded[0], "items": picked_conceded[1],
        },
        "gold_off_the_ballot": {"correct": ballot_missing[0], "items": ballot_missing[1]},
        "winning_position": dict(positions),
        "winning_position_test": position_bias_chi_square(positions),
    }


def rate(hits, total):
    return f"{hits / total:.4f}" if total else "     -"


def show(value, spec="7.4f"):
    return format(value, spec) if isinstance(value, float) else f"{'-':>{int(spec.split('.')[0])}}"


def analyse(path, articles, min_sentence):
    rows = usable_rows(path)
    n_rounds = len(rows[0]["analyses_by_round"])
    per_round = [
        round_report(rows, index, articles, min_sentence) for index in range(n_rounds)
    ]
    judged = Counter(judged_round_of(row) for row in rows)
    return {
        "path": str(path),
        "items": len(rows),
        "rounds": n_rounds,
        "judged_round": dict(judged),
        "per_round": per_round,
        "ballot_paired": ballot_paired(rows, 0, n_rounds - 1) if n_rounds > 1 else None,
        "judge": judge_report(rows),
    }


def print_report(report, articles):
    print(f"\n=== {report['path']}")
    print(f"[run] {report['items']} items   {report['rounds']} rounds   "
          f"judged round {report['judged_round']}")

    print("\n[per round] surface")
    print(f"  {'round':>5} {'cases':>6} {'chars':>8} {'quotes':>7} {'hedges':>7} "
          f"{'peers':>7} {'peer/quote':>11} {'unfinished':>11}")
    for row in report["per_round"]:
        print(f"  {row['round']:>5} {row['cases']:6d} {show(row['chars'], '8.1f')} "
              f"{show(row['quotes'])} {show(row['hedges'])} {show(row['peer_mentions'])} "
              f"{show(row['peer_to_quote'], '11.4f')} {show(row['unfinished_rate'], '11.4f')}")

    print("\n[the ballot] is the gold label still being argued?  "
          "(coverage, in analyze_information_flow's words)")
    print(f"  {'round':>5} {'gold adv held':>14} {'others held':>13} {'gap p':>8} "
          f"{'coverage':>15} {'3 lbl':>6} {'2 lbl':>6} {'1 lbl':>6}")
    for row in report["per_round"]:
        sizes = row["ballot_sizes"]
        gap = row["hold_gap_p"]
        print(f"  {row['round']:>5} "
              f"{rate(row['gold_advocate_held'], row['gold_advocate_n']):>14} "
              f"{rate(row['other_advocate_held'], row['other_advocate_n']):>13} "
              f"{show(gap, '8.4f')} "
              f"{rate(row['ballot_has_gold'], report['items']):>15} "
              f"{sizes.get(3, 0):6d} {sizes.get(2, 0):6d} {sizes.get(1, 0):6d}")
    paired = report["ballot_paired"]
    if paired:
        print(f"  paired, same gold advocates: held {paired['held_first']} -> "
              f"{paired['held_last']} of {paired['items']}   "
              f"conceded {paired['conceded_during_exchange']}, "
              f"recanted {paired['recanted_during_exchange']}   "
              f"p = {paired['p_value_two_sided']:.4f}")
        print("  a fall here means gold left the candidate set, so the loss is in "
              "the ballot rather than in the judge.")

    print("\n[homogenisation] mean over the three pairs of cases")
    print(f"  {'round':>5} {'tokens':>8} {'quotes':>8} {'sentences':>10}")
    for row in report["per_round"]:
        print(f"  {row['round']:>5} {show(row['token_overlap'], '8.4f')} "
              f"{show(row['quote_overlap'], '8.4f')} {show(row['sentence_reuse'], '10.4f')}")
    print("  rising overlap means the judge is separating three copies, whatever "
          "it reads.")

    if articles:
        print("\n[grounding] against the article")
        print(f"  {'round':>5} {'quotes':>8} {'exact':>8} {'loose':>8} {'prefix12':>9} "
              f"{'shingles':>9}")
        for row in report["per_round"]:
            check = row["quote_verification"]
            if not check:
                print(f"  {row['round']:>5} {'-':>8}")
                continue
            print(f"  {row['round']:>5} {check['quotes']:8d} {check['exact']:8.4f} "
                  f"{check['loose']:8.4f} {check['prefix12']:9.4f} "
                  f"{show(row['article_shingle_overlap'], '9.4f')}")
        print("  `shingles` is the share of the case's own text traceable to the "
              "article, so it falls when the case is about the other analysts.")
    else:
        print("\n[grounding] skipped -- pass --data-path to check the cases "
              "against the article")

    judge = report["judge"]
    if judge:
        held = judge["gold_advocate_held"]["held"]
        conceded = judge["gold_advocate_held"]["conceded"]
        picked = judge["judge_picked_a_conceded_case"]
        missing = judge["gold_off_the_ballot"]
        print(f"\n[judge] on the round it read, {judge['judged_items']} judged items, "
              f"accuracy {judge['accuracy']:.4f}")
        print(f"  gold advocate held      {rate(held['correct'], held['items'])}  "
              f"({held['items']} items)")
        print(f"  gold advocate conceded  {rate(conceded['correct'], conceded['items'])}  "
              f"({conceded['items']} items)")
        print(f"  judge picked a conceded case  {picked['items']} items, "
              f"{rate(picked['correct'], picked['items'])} correct")
        print(f"  gold off the ballot entirely  {missing['items']} items, "
              f"{rate(missing['correct'], missing['items'])} correct")
        test = judge["winning_position_test"]
        if test.get("p_value") is not None:
            print(f"  winning position {judge['winning_position']}  "
                  f"chi2 {test['chi_square']:.2f}, p = {test['p_value']:.4f}")

    if report["rounds"] > 1:
        first, last = report["per_round"][0], report["per_round"][-1]
        print(f"\n[round 0 -> {last['round']}]")
        for key, label in (
            ("chars", "case length"),
            ("quotes", "quoted passages"),
            ("peer_mentions", "mentions of a peer"),
            ("token_overlap", "token overlap between cases"),
            ("sentence_reuse", "verbatim sentence reuse"),
            ("article_shingle_overlap", "text traceable to the article"),
        ):
            a, b = first.get(key), last.get(key)
            if isinstance(a, float) and isinstance(b, float):
                print(f"  {label:32} {a:9.4f} -> {b:9.4f}  ({b - a:+.4f})")


def main():
    args = parse_args()
    articles = articles_by_id(args.data_path)
    reports = []
    for path in args.items:
        report = analyse(path, articles, args.min_sentence)
        print_report(report, articles)
        reports.append(report)

    if args.json_out:
        payload = reports[0] if len(reports) == 1 else {"runs": reports}
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()

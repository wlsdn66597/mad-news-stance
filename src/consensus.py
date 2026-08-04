"""Offline aggregation and evaluation for saved multi-agent trajectories.

This module never calls a model and never mutates the source result file.  It
accepts the Phase 2 ``debate_trace.answers_by_round`` schema and also the
transition-case export used by the stance experiments.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import LABELS_DEFAULT, accuracy, confusion, macro_f1, per_class_prf

LABELS = tuple(LABELS_DEFAULT)
POLAR_LABELS = frozenset({"supportive", "oppositional"})
FREE_MAD_DEFAULT_WEIGHTS = (20.0, 25.0, 30.0, 20.0)
FREE_MAD_CITATION = "https://aclanthology.org/2026.findings-acl.1600/"


@dataclass(frozen=True)
class VoteResult:
    label: str | None
    tied: bool
    counts: dict[str, int]


def stable_seed(seed: int, item_id: Any, namespace: str = "") -> int:
    token = f"{seed}\0{namespace}\0{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for label in LABELS:
        if text == label or text.startswith(label[:4]):
            return label
    return None


def parse_stance(text: Any) -> str | None:
    """Parse saved responses without instantiating the dataset task."""
    if not isinstance(text, str):
        return normalize_label(text)
    lowered = text.lower()
    import re

    matches = re.findall(
        r"final\s*stance\s*[:=\-]>?\s*[\"'(]?\s*([a-z]+)", lowered
    )
    if matches:
        parsed = normalize_label(matches[-1])
        if parsed:
            return parsed
    found = [label for label in LABELS if label in lowered]
    return max(found, key=lambda label: lowered.rfind(label)) if found else None


def unique_majority(values: Iterable[Any]) -> VoteResult:
    valid = [normalize_label(value) for value in values]
    counts = Counter(value for value in valid if value in LABELS)
    if not counts:
        return VoteResult(None, True, {})
    top = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top)
    return VoteResult(winners[0] if len(winners) == 1 else None, len(winners) != 1, dict(counts))


def deterministic_fallback(item_id: Any, seed: int = 0) -> str:
    return LABELS[stable_seed(seed, item_id, "fallback") % len(LABELS)]


def extract_round_predictions(record: Mapping[str, Any]) -> tuple[list[list[str | None]], list[list[str]]]:
    """Return parsed labels and raw answers for every available round."""
    trace = record.get("debate_trace", record)
    raw_rounds = trace.get("answers_by_round") or record.get("answers_by_round")
    if raw_rounds:
        raw = [[str(answer) for answer in round_answers] for round_answers in raw_rounds]
        return [[parse_stance(answer) for answer in round_answers] for round_answers in raw], raw

    round0 = record.get("round0_agent_preds")
    final = record.get("final_agent_preds")
    if round0 is not None and final is not None:
        labels = [
            [normalize_label(value) for value in round0],
            [normalize_label(value) for value in final],
        ]
        raw = [
            [str(value) for value in record.get("round0_agent_responses", round0)],
            [str(value) for value in record.get("final_agent_responses", final)],
        ]
        return labels, raw
    raise ValueError("record has no answers_by_round or round prediction fields")


def current_final_majority(rounds: Sequence[Sequence[Any]], item_id: Any, fallback_seed: int = 0):
    vote = unique_majority(rounds[-1])
    if vote.label:
        return vote.label, "final_unique_majority"
    # Preserve the repository's historical Counter.most_common(1) behavior:
    # when the final round is tied, the first valid agent label wins. This is
    # deliberately not used by the new fallback rules, which specify their
    # own deterministic tie handling.
    for value in rounds[-1]:
        label = normalize_label(value)
        if label in LABELS:
            return label, "final_tie_legacy_first_valid"
    return deterministic_fallback(item_id, fallback_seed), "final_no_valid_label_fallback"


def round0_fallback(rounds: Sequence[Sequence[Any]], item_id: Any, fallback_seed: int = 0):
    final = unique_majority(rounds[-1])
    if len(rounds[-1]) > 0 and max(final.counts.values(), default=0) == len(rounds[-1]):
        return final.label, "final_unanimous"
    initial = unique_majority(rounds[0])
    if initial.label:
        return initial.label, "round0_unique_majority"
    return deterministic_fallback(item_id, fallback_seed), "round0_tie_deterministic_fallback"


def cross_round_pooled_vote(rounds: Sequence[Sequence[Any]], item_id: Any, fallback_seed: int = 0):
    pooled = unique_majority(value for round_values in rounds for value in round_values)
    if pooled.label:
        return pooled.label, "cross_round_unique_majority"
    initial = unique_majority(rounds[0])
    if initial.label:
        return initial.label, "pooled_tie_round0_majority"
    return deterministic_fallback(item_id, fallback_seed), "pooled_and_round0_tie_deterministic_fallback"


def free_mad_scores(
    rounds: Sequence[Sequence[Any]],
    weights: Sequence[float] = FREE_MAD_DEFAULT_WEIGHTS,
) -> dict[str, float]:
    """Algorithm 1 trajectory score from Free-MAD (Findings ACL 2026).

    For round k (zero-indexed), f=1/(k+1).  The initial answer receives
    w1*f.  A changed answer subtracts w2*f from the previous answer and adds
    w3*f to the new answer.  An unchanged answer receives w4*f.
    """
    if len(weights) != 4:
        raise ValueError("Free-MAD requires exactly four weights: w1,w2,w3,w4")
    w1, w2, w3, w4 = map(float, weights)
    if not rounds:
        return {}
    n_agents = max(len(values) for values in rounds)
    scores: defaultdict[str, float] = defaultdict(float)
    for agent_index in range(n_agents):
        previous = None
        for round_index, values in enumerate(rounds):
            current = normalize_label(values[agent_index]) if agent_index < len(values) else None
            if current not in LABELS:
                previous = current
                continue
            factor = 1.0 / (round_index + 1)
            if round_index == 0:
                scores[current] += w1 * factor
            elif current != previous:
                if previous in scores:
                    scores[previous] -= w2 * factor
                scores[current] += w3 * factor
            else:
                scores[current] += w4 * factor
            previous = current
    return dict(scores)


def free_mad_exact(
    rounds: Sequence[Sequence[Any]], item_id: Any, weights=FREE_MAD_DEFAULT_WEIGHTS, tie_seed: int = 0
):
    scores = free_mad_scores(rounds, weights)
    if not scores:
        return deterministic_fallback(item_id, tie_seed), "free_mad_no_valid_label_fallback", scores
    top = max(scores.values())
    winners = sorted(label for label, score in scores.items() if math.isclose(score, top))
    # The paper samples a winner uniformly. A per-item seeded RNG preserves that
    # behavior while making repeated offline evaluation reproducible.
    winner = random.Random(stable_seed(tie_seed, item_id, "free_mad_tie")).choice(winners)
    return winner, ("free_mad_unique" if len(winners) == 1 else "free_mad_seeded_random_tie"), scores


def exploratory_neutral_boundary_fallback(rounds, item_id, fallback_seed=0):
    initial = unique_majority(rounds[0])
    final = unique_majority(rounds[-1])
    if initial.label and final.label and initial.label != final.label:
        if "neutral" in {initial.label, final.label}:
            return initial.label, "exploratory_neutral_boundary_round0"
        if {initial.label, final.label} == POLAR_LABELS:
            return final.label, "exploratory_polar_switch_final"
    return current_final_majority(rounds, item_id, fallback_seed)


AGGREGATORS = {
    "current_final_majority": current_final_majority,
    "round0_fallback": round0_fallback,
    "cross_round_pooled_vote": cross_round_pooled_vote,
    "exploratory_neutral_boundary_fallback": exploratory_neutral_boundary_fallback,
}


def aggregate(method: str, rounds, item_id, fallback_seed=0, free_mad_weights=FREE_MAD_DEFAULT_WEIGHTS):
    if method == "free_mad_exact":
        label, reason, scores = free_mad_exact(rounds, item_id, free_mad_weights, fallback_seed)
        return {"pred": label, "reason": reason, "scores": scores}
    if method not in AGGREGATORS:
        raise ValueError(f"unknown aggregation method: {method}")
    label, reason = AGGREGATORS[method](rounds, item_id, fallback_seed)
    return {"pred": label, "reason": reason}


def judge_trigger(rounds: Sequence[Sequence[Any]], mode: str = "instability") -> tuple[bool, str]:
    initial = unique_majority(rounds[0])
    final = unique_majority(rounds[-1])
    final_unanimous = bool(final.counts) and max(final.counts.values()) == len(rounds[-1])
    if mode == "all":
        return True, "all"
    if mode == "tie_only":
        return final.tied, "final_tie" if final.tied else "stable"
    if mode == "non_unanimous":
        return not final_unanimous, "final_non_unanimous" if not final_unanimous else "stable"
    if mode == "instability":
        if final.tied:
            return True, "final_tie"
        if initial.label and final.label and initial.label != final.label:
            return True, "round0_final_disagree"
        return False, "stable"
    raise ValueError(f"unknown judge trigger: {mode}")


def classification_metrics(preds: Sequence[Any], golds: Sequence[Any]) -> dict[str, Any]:
    return {
        "items": len(golds),
        "accuracy": accuracy(preds, golds),
        "macro_f1": macro_f1(preds, golds, LABELS),
        "per_class": per_class_prf(preds, golds, LABELS),
        "confusion": confusion(preds, golds, LABELS),
    }


def transition_metrics(baseline: Sequence[Any], candidate: Sequence[Any], golds: Sequence[Any]):
    counts = Counter()
    for before, after, gold in zip(baseline, candidate, golds):
        before_ok, after_ok = before == gold, after == gold
        counts[(before_ok, after_ok)] += 1
    w2c = counts[(False, True)]
    c2w = counts[(True, False)]
    return {
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "wrong_to_wrong": counts[(False, False)],
        "correct_to_correct": counts[(True, True)],
        "net_improvement": w2c - c2w,
        "error_correction_rate": w2c / max(1, sum(b != g for b, g in zip(baseline, golds))),
        "correct_preservation_rate": counts[(True, True)] / max(1, sum(b == g for b, g in zip(baseline, golds))),
    }


def mcnemar_exact(baseline: Sequence[Any], candidate: Sequence[Any], golds: Sequence[Any]) -> dict[str, Any]:
    transitions = transition_metrics(baseline, candidate, golds)
    b, c = transitions["correct_to_wrong"], transitions["wrong_to_correct"]
    n = b + c
    if n == 0:
        return {"discordant": 0, "p_value_two_sided": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return {"discordant": n, "p_value_two_sided": min(1.0, 2 * tail)}


def bootstrap_accuracy_delta(
    baseline: Sequence[Any], candidate: Sequence[Any], golds: Sequence[Any], seed=0, samples=2000
):
    if not golds:
        return {"samples": samples, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    n = len(golds)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        delta = sum(candidate[i] == golds[i] for i in indices) / n
        delta -= sum(baseline[i] == golds[i] for i in indices) / n
        deltas.append(delta)
    deltas.sort()
    return {"samples": samples, "ci95": [deltas[int(.025 * samples)], deltas[min(samples - 1, int(.975 * samples))]]}


def boundary_transition_counts(baseline, candidate):
    neutral = polar = 0
    for before, after in zip(baseline, candidate):
        if before == after:
            continue
        labels = {before, after}
        if "neutral" in labels and labels & POLAR_LABELS:
            neutral += 1
        elif labels == POLAR_LABELS:
            polar += 1
    return {"neutral_polar_switches": neutral, "supportive_oppositional_switches": polar}

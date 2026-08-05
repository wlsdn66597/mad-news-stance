"""Pipeline glue for selective consensus + selective article-only judge.

Decision order for one saved item::

    instability trigger
      -> extra reconsideration round by the original agents (optional)
      -> unanimous revised label accepted, judge skipped
      -> otherwise the existing article-only judge
      -> otherwise the existing round0 fallback

Nothing here imports torch: the model calls arrive as injected callables so the
routing, the judge cache and the metrics stay unit-testable on CPU.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .consensus import (
    aggregate,
    bootstrap_accuracy_delta,
    boundary_transition_counts,
    classification_metrics,
    judge_trigger,
    mcnemar_exact,
    transition_metrics,
    unique_majority,
)
from .consensus_round import NO_CONSENSUS_MODE
from .selective_judge import judge_user_payload

FALLBACK_METHOD = "round0_fallback"

#: Judge settings that must match before a cached prediction may be reused.
JUDGE_CACHE_CONFIG_FIELDS = (
    "judge_model",
    "judge_input_mode",
    "judge_temperature",
    "judge_max_new_tokens",
    "judge_max_retries",
)


def judge_payload_fingerprint(record: Mapping[str, Any]) -> str | None:
    """Hash of the exact article-only judge input for this item."""
    if not record.get("article"):
        # The article is joined from --data-path; without it a hash would only
        # fingerprint empty strings and could wrongly validate a cache.
        return None
    payload = judge_user_payload(record, [])
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _row_judge_config(row: Mapping[str, Any]) -> dict[str, Any]:
    decoding = row.get("judge_decoding") or {}
    return {
        "judge_model": row.get("judge_model"),
        "judge_input_mode": row.get("judge_input_mode"),
        "judge_temperature": decoding.get("temperature"),
        "judge_max_new_tokens": decoding.get("max_new_tokens"),
        "judge_max_retries": decoding.get("max_retries"),
    }


def _sibling_config(items_path: Path) -> dict[str, Any]:
    config_path = items_path.with_name(items_path.name.replace(".items.json", ".config.json"))
    if config_path == items_path or not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_judge_cache(
    path: str,
    expected: Mapping[str, Any],
    input_result: str | None = None,
    require_config: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a previous selective-judge ``.items.json`` for prediction reuse.

    Only rows produced with the same judge model, input mode and decoding config
    are kept.  When the companion ``.config.json`` is present it must also agree
    on the judge settings and on the source result file.
    """
    items_path = Path(path)
    rows = json.loads(items_path.read_text(encoding="utf-8"))
    config = _sibling_config(items_path)
    validation: dict[str, Any] = {
        "path": str(items_path),
        "rows": len(rows),
        "config_found": bool(config),
        "expected": dict(expected),
        "mismatched_rows": 0,
        "usable_rows": 0,
        "config_mismatch": {},
        "source_input_result": config.get("input_result"),
    }
    if config:
        observed = {
            "judge_model": config.get("judge_model"),
            "judge_input_mode": config.get("judge_input_mode"),
            "judge_temperature": config.get("judge_temperature"),
            "judge_max_new_tokens": config.get("judge_max_new_tokens"),
            "judge_max_retries": config.get("judge_max_retries"),
        }
        for field in JUDGE_CACHE_CONFIG_FIELDS:
            # A cached run may not pin the model explicitly; the rows carry the
            # resolved id and are checked below.
            if observed[field] is None:
                continue
            if observed[field] != expected[field]:
                validation["config_mismatch"][field] = {
                    "cache": observed[field],
                    "expected": expected[field],
                }
        if input_result and config.get("input_result"):
            if Path(config["input_result"]).name != Path(input_result).name:
                validation["config_mismatch"]["input_result"] = {
                    "cache": config.get("input_result"),
                    "expected": input_result,
                }
    elif require_config:
        raise ValueError(
            f"{items_path} has no companion .config.json; the judge cache cannot be "
            "verified. Re-run without --judge-result or pass --no-judge-cache-require-config."
        )
    if validation["config_mismatch"]:
        raise ValueError(
            "judge cache was produced with a different configuration: "
            f"{json.dumps(validation['config_mismatch'], ensure_ascii=False)}"
        )

    usable: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("triggered"):
            continue
        if _row_judge_config(row) != {field: expected[field] for field in JUDGE_CACHE_CONFIG_FIELDS}:
            validation["mismatched_rows"] += 1
            continue
        usable[str(row["item_id"])] = row
    validation["usable_rows"] = len(usable)
    return usable, validation


def direct_judge_predictions(path: str) -> dict[str, Any]:
    """Final predictions of a previous direct article-only judge run."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["item_id"]): row.get("final_prediction") for row in rows}


def process_record(
    record: Mapping[str, Any],
    aggregation_method: str = "current_final_majority",
    trigger_mode: str = "instability",
    fallback_seed: int = 0,
    consensus_mode: str = NO_CONSENSUS_MODE,
    consensus_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    judge_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    judge_cache: Mapping[str, Mapping[str, Any]] | None = None,
    judge_enabled: bool = True,
) -> dict[str, Any]:
    """Route one saved item through consensus, judge and fallback."""
    item_id = record["id"]
    rounds = record["round_predictions"]
    gold = record.get("gold")
    baseline = aggregate(aggregation_method, rounds, item_id, fallback_seed)
    triggered, trigger_reason = judge_trigger(rounds, trigger_mode)
    payload_fingerprint = judge_payload_fingerprint(record) if triggered else None

    row: dict[str, Any] = {
        "item_id": item_id,
        "gold": gold,
        "triggered": triggered,
        "trigger_mode": trigger_mode,
        "trigger_reason": trigger_reason,
        "round0_agent_preds": rounds[0],
        "final_agent_preds": rounds[-1],
        "round0_majority": unique_majority(rounds[0]).label,
        "final_majority": unique_majority(rounds[-1]).label,
        "baseline_prediction": baseline["pred"],
        "baseline_reason": baseline["reason"],
        "baseline_correct": baseline["pred"] == gold,
        "consensus_prompt_mode": consensus_mode if triggered else None,
        "previous_agent_labels": [],
        "revised_agent_labels": [],
        "revised_agent_outputs": [],
        "peer_orders": [],
        "consensus_reached": False,
        "consensus_label": None,
        "consensus_reason": None,
        "consensus_parse_failure": False,
        "judge_required": False,
        "judge_source": "none",
        "judge_prediction": None,
        "judge_payload_sha256": payload_fingerprint,
        "fallback_used": False,
        "fallback_reason": None,
        "consensus_cost": {"calls": 0, "latency_seconds": 0.0,
                           "token_usage": {"input_tokens": 0, "output_tokens": 0}},
        "judge_cost": {"calls": 0, "latency_seconds": 0.0,
                       "token_usage": {"input_tokens": 0, "output_tokens": 0}},
    }

    if not triggered:
        row["final_prediction"] = baseline["pred"]
        row["final_source"] = "baseline"
        row["final_correct"] = baseline["pred"] == gold
        return row

    if consensus_mode != NO_CONSENSUS_MODE:
        if consensus_fn is None:
            raise ValueError("consensus_fn is required when a consensus prompt mode is set")
        consensus = consensus_fn(record)
        row.update(
            {
                "consensus_prompt_mode": consensus["consensus_prompt_mode"],
                "previous_agent_labels": consensus["previous_agent_labels"],
                "revised_agent_labels": consensus["revised_agent_labels"],
                "revised_agent_outputs": consensus["revised_agent_outputs"],
                "peer_orders": consensus["peer_orders"],
                "consensus_reached": consensus["consensus_reached"],
                "consensus_label": consensus["consensus_label"],
                "consensus_reason": consensus["consensus_reason"],
                "consensus_parse_failure": consensus["parse_failure"],
                "consensus_label_counts": consensus["label_counts"],
                "consensus_agent_context_source": consensus.get("agent_context_source"),
                "consensus_debate_template_source": consensus.get("debate_template_source"),
                "consensus_agent_seeds": consensus.get("agent_seeds"),
                "consensus_cost": {
                    "calls": consensus.get("calls", 0),
                    "latency_seconds": consensus.get("latency_seconds", 0.0),
                    "token_usage": consensus.get(
                        "token_usage", {"input_tokens": 0, "output_tokens": 0}
                    ),
                },
            }
        )
        if consensus["consensus_reached"]:
            row["final_prediction"] = consensus["consensus_label"]
            row["final_source"] = "consensus"
            row["final_correct"] = consensus["consensus_label"] == gold
            return row

    row["judge_required"] = True
    cached = (judge_cache or {}).get(str(item_id))
    if cached is not None:
        cached_fingerprint = cached.get("judge_payload_sha256")
        if cached_fingerprint and cached_fingerprint != payload_fingerprint:
            row["judge_cache_rejected"] = "payload_mismatch"
            cached = None
    if cached is not None:
        row.update(
            {
                "judge_source": "cache",
                "judge_prediction": cached.get("judge_prediction"),
                "judge_raw_output": cached.get("judge_raw_output"),
                "judge_parsed_output": cached.get("judge_parsed_output"),
                "judge_cache_item": str(item_id),
            }
        )
    elif judge_enabled:
        if judge_fn is None:
            raise ValueError("judge_fn is required when the judge is enabled")
        result = judge_fn(record)
        row.update(
            {
                "judge_source": "model_call",
                "judge_prediction": result["prediction"],
                "judge_raw_output": result.get("raw_output"),
                "judge_parsed_output": result.get("parsed_output"),
                "judge_attempts": result.get("attempts"),
                "judge_cost": {
                    "calls": len(result.get("attempts") or []),
                    "latency_seconds": result.get("latency_seconds", 0.0),
                    "token_usage": result.get(
                        "token_usage", {"input_tokens": 0, "output_tokens": 0}
                    ),
                },
            }
        )

    if row["judge_prediction"]:
        row["final_prediction"] = row["judge_prediction"]
        row["final_source"] = "judge"
    else:
        fallback = aggregate(FALLBACK_METHOD, rounds, item_id, fallback_seed)
        row["final_prediction"] = fallback["pred"]
        row["final_source"] = "fallback"
        row["fallback_used"] = True
        row["fallback_reason"] = (
            "judge_disabled" if not judge_enabled and row["judge_source"] == "none"
            else fallback["reason"]
        )
    row["final_correct"] = row["final_prediction"] == gold
    return row


MODE_SHORT = {
    "none": "direct",
    "plain_extra_round": "plain",
    "independent_reconsideration": "reconsider",
}
AGG_SHORT = {
    "current_final_majority": "finalmaj",
    "round0_fallback": "r0fb",
    "cross_round_pooled_vote": "pooled",
    "free_mad_exact": "freemad",
    "exploratory_neutral_boundary_fallback": "explneut",
}


def safe_name(value: Any) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def output_prefix(
    input_result: str,
    consensus_mode: str,
    trigger: str,
    consensus_model: Any,
    judge_model: Any,
    consensus_temperature: Any,
    consensus_seed: Any,
    consensus_order_seed: Any,
    consensus_max_new_tokens: Any,
    judge_input_mode: str,
    judge_temperature: Any,
    judge_order_seed: Any,
    aggregation_method: str,
    limit: int = 200,
) -> str:
    """Compact, config-identifying file stem.

    The source stems are already ~90 characters, so the suffix is abbreviated and
    the whole name is capped: ext4 rejects a path component over 255 bytes and
    the run writes `<prefix>.summary.json` on top of this stem.
    """
    parts = [
        Path(input_result).stem,
        f"cons-{MODE_SHORT.get(consensus_mode, safe_name(consensus_mode))}",
        f"trig-{trigger}",
        f"m-{safe_name(str(consensus_model).split('/')[-1])}",
    ]
    if judge_model != consensus_model:
        parts.append(f"jm-{safe_name(str(judge_model).split('/')[-1])}")
    if consensus_mode != NO_CONSENSUS_MODE:
        parts += [
            f"ct{float(consensus_temperature):g}",
            f"cs{consensus_seed}",
            f"co{consensus_order_seed}",
            f"ctok{consensus_max_new_tokens}",
        ]
    parts += [
        f"j-{'art' if judge_input_mode == 'article_only' else 'trace'}",
        f"jt{float(judge_temperature):g}",
        f"js{judge_order_seed}",
        f"agg-{AGG_SHORT.get(aggregation_method, safe_name(aggregation_method))}",
    ]
    name = "_".join(parts)
    if len(name) <= limit:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: limit - 9]}-{digest}"


def load_existing_rows(items_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Rows already written by an interrupted run, keyed by item id."""
    if not items_path or not Path(items_path).exists():
        return {}
    rows = json.loads(Path(items_path).read_text(encoding="utf-8"))
    return {str(row["item_id"]): row for row in rows}


def process_records(
    records: Sequence[Mapping[str, Any]],
    items_path: str | Path | None = None,
    save: Callable[[Path, list], Any] | None = None,
    resume: bool = True,
    on_row: Callable[[int, int, Mapping[str, Any], bool], Any] | None = None,
    **process_kwargs: Any,
) -> list[dict[str, Any]]:
    """Process every record, skipping items already present in ``items_path``."""
    existing = load_existing_rows(items_path) if resume else {}
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        key = str(record["id"])
        reused = key in existing
        row = existing[key] if reused else process_record(record, **process_kwargs)
        rows.append(row)
        if not reused and save is not None and items_path is not None:
            save(Path(items_path), rows)
        if on_row is not None:
            on_row(index, len(records), row, reused)
    return rows


def _cost_totals(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return {
        "calls": sum(row.get(key, {}).get("calls", 0) for row in rows),
        "latency_seconds": sum(row.get(key, {}).get("latency_seconds", 0.0) for row in rows),
        "input_tokens": sum(
            row.get(key, {}).get("token_usage", {}).get("input_tokens", 0) for row in rows
        ),
        "output_tokens": sum(
            row.get(key, {}).get("token_usage", {}).get("output_tokens", 0) for row in rows
        ),
    }


def consensus_stage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if row["triggered"]]
    attempted = [row for row in triggered if row.get("revised_agent_labels")]
    reached = [row for row in attempted if row["consensus_reached"]]
    correct = [row for row in reached if row["final_correct"]]
    judged = [row for row in triggered if row["judge_required"]]
    return {
        "triggered_total": len(triggered),
        "consensus_attempted": len(attempted),
        "consensus_reached": len(reached),
        "consensus_reached_ratio": len(reached) / max(1, len(attempted)),
        "consensus_correct": len(correct),
        "consensus_accuracy": len(correct) / max(1, len(reached)),
        "wrong_unanimous_consensus": len(reached) - len(correct),
        "consensus_preserved_correct": sum(
            1 for row in reached if row["baseline_correct"] and row["final_correct"]
        ),
        "consensus_fixed_wrong": sum(
            1 for row in reached if not row["baseline_correct"] and row["final_correct"]
        ),
        "consensus_broke_correct": sum(
            1 for row in reached if row["baseline_correct"] and not row["final_correct"]
        ),
        "consensus_parse_failures": sum(1 for row in attempted if row["consensus_parse_failure"]),
        "consensus_outcome_counts": dict(
            Counter(row["consensus_reason"] for row in attempted)
        ),
        "judge_required": len(judged),
        "judge_required_ratio": len(judged) / max(1, len(triggered)),
        "judge_source_counts": dict(Counter(row["judge_source"] for row in judged)),
        "fallback_used": sum(1 for row in triggered if row["fallback_used"]),
        "final_source_counts": dict(Counter(row["final_source"] for row in rows)),
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_meta: Mapping[str, Any] | None = None,
    judge_cache_validation: Mapping[str, Any] | None = None,
    direct_predictions: Mapping[str, Any] | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    golds = [row["gold"] for row in rows]
    baseline_preds = [row["baseline_prediction"] for row in rows]
    final_preds = [row["final_prediction"] for row in rows]
    triggered = [row for row in rows if row["triggered"]]

    summary: dict[str, Any] = {
        "items": len(rows),
        "config": dict(config),
        "source_meta": dict(source_meta or {}),
        "judge_cache": dict(judge_cache_validation or {}),
        "baseline_metrics": classification_metrics(baseline_preds, golds),
        "final_metrics": classification_metrics(final_preds, golds),
        "overall_transitions": transition_metrics(baseline_preds, final_preds, golds),
        "boundary_transitions": boundary_transition_counts(baseline_preds, final_preds),
        "mcnemar_exact": mcnemar_exact(baseline_preds, final_preds, golds),
        "bootstrap_accuracy_delta": bootstrap_accuracy_delta(
            baseline_preds, final_preds, golds, bootstrap_seed
        ),
        "trigger_subset": {
            "triggered_total": len(triggered),
            "trigger_ratio": len(triggered) / max(1, len(rows)),
            "baseline_metrics": classification_metrics(
                [row["baseline_prediction"] for row in triggered],
                [row["gold"] for row in triggered],
            ),
            "final_metrics": classification_metrics(
                [row["final_prediction"] for row in triggered],
                [row["gold"] for row in triggered],
            ),
            "transitions": transition_metrics(
                [row["baseline_prediction"] for row in triggered],
                [row["final_prediction"] for row in triggered],
                [row["gold"] for row in triggered],
            ),
        },
        "consensus_stage": consensus_stage_summary(rows),
        "cost": {
            "consensus": _cost_totals(rows, "consensus_cost"),
            "judge": _cost_totals(rows, "judge_cost"),
        },
    }

    if direct_predictions:
        covered = [row for row in rows if str(row["item_id"]) in direct_predictions]
        direct_preds = [direct_predictions[str(row["item_id"])] for row in covered]
        covered_final = [row["final_prediction"] for row in covered]
        covered_golds = [row["gold"] for row in covered]
        direct_metrics = classification_metrics(direct_preds, covered_golds)
        final_metrics = classification_metrics(covered_final, covered_golds)
        summary["direct_article_judge_comparison"] = {
            "covered_items": len(covered),
            "complete_coverage": len(covered) == len(rows),
            "direct_metrics": direct_metrics,
            "this_run_metrics": final_metrics,
            "accuracy_delta": final_metrics["accuracy"] - direct_metrics["accuracy"],
            "macro_f1_delta": final_metrics["macro_f1"] - direct_metrics["macro_f1"],
            "transitions_vs_direct": transition_metrics(
                direct_preds, covered_final, covered_golds
            ),
            "mcnemar_exact_vs_direct": mcnemar_exact(
                direct_preds, covered_final, covered_golds
            ),
        }
    return summary

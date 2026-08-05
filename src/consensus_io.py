"""Schema adapters for Phase 2 result and transition-case JSON files."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .consensus import extract_round_predictions


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def flatten_transition_file(path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows = []
    for group, values in data.get("groups", {}).items():
        for value in values:
            row = dict(value)
            row["transition_group"] = group
            rows.append(row)
    return rows


def parse_prompt_fields(prompt: str | None) -> dict[str, str]:
    if not prompt:
        return {}
    match = re.search(
        r"Issue:\s*(.*?)\r?\nHeadline:\s*(.*?)\r?\n(?:Article(?: \(Korean\))?):\s*\r?\n(.*)",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}
    return {"issue": match.group(1).strip(), "headline": match.group(2).strip(), "article": match.group(3).strip()}


def dataset_by_id(data_path: str | None) -> dict[str, dict[str, Any]]:
    if not data_path or not Path(data_path).exists():
        return {}
    data = load_json(data_path)
    return {
        str(row["id"]): {
            "issue": row.get("issue", ""),
            "headline": row.get("haedline") or row.get("headline", ""),
            "article": row.get("article", ""),
            "gold": row.get("stance"),
        }
        for row in data
    }


def result_records(data: Mapping[str, Any], data_path: str | None = None):
    source = data.get("debate")
    if not isinstance(source, Mapping):
        raise ValueError("input result has no top-level debate mapping")
    dataset = dataset_by_id(data_path)
    rows = []
    for item_id, original in source.items():
        row = dict(original)
        row["item_id"] = item_id
        row["id"] = item_id
        question = row.get("prompt") or row.get("question_or_prompt")
        if not question:
            question = (row.get("debate_trace") or {}).get("question")
        row.update({key: value for key, value in parse_prompt_fields(question).items() if not row.get(key)})
        if str(item_id) in dataset:
            row.update({key: value for key, value in dataset[str(item_id)].items() if value is not None})
        parsed_rounds, raw_rounds = extract_round_predictions(row)
        row["round_predictions"] = parsed_rounds
        row["raw_rounds"] = raw_rounds
        rows.append(row)
    return rows


def transition_records(data: Mapping[str, Any]):
    rows = []
    for group, values in data.get("groups", {}).items():
        for original in values:
            row = dict(original)
            row["transition_group"] = group
            row["id"] = row.get("item_id")
            question = row.get("question_or_prompt")
            row.update({key: value for key, value in parse_prompt_fields(question).items() if not row.get(key)})
            parsed_rounds, raw_rounds = extract_round_predictions(row)
            row["round_predictions"] = parsed_rounds
            row["raw_rounds"] = raw_rounds
            rows.append(row)
    return rows


def load_consensus_records(input_result=None, transition_file=None, data_path=None):
    if not input_result and not transition_file:
        raise ValueError("provide --input-result or --transition-file")
    if input_result:
        data = load_json(input_result)
        rows = result_records(data, data_path=data_path)
        meta = dict(data.get("_meta", {}))
    else:
        data = load_json(transition_file)
        rows = transition_records(data)
        meta = {"source": data.get("source"), "transition_summary": data.get("summary")}

    transition_summary = None
    if transition_file:
        transition = load_json(transition_file)
        transition_summary = transition.get("summary")
        expected_ids = {
            str(row.get("item_id"))
            for values in transition.get("groups", {}).values()
            for row in values
        }
        if input_result:
            available = {str(row["id"]) for row in rows}
            missing = sorted(expected_ids - available)
            if missing:
                raise ValueError(f"transition file contains {len(missing)} IDs absent from input result")
    return rows, meta, transition_summary

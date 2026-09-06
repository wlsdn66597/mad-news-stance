"""Utilities for a zero-shot JoA-style segment-to-article pipeline.

The source segmentation file contains labels predicted by a RoBERTa segment
classifier.  Those labels must not leak into the replacement EXAONE segment
agent.  This module therefore keeps only the XML-like boundaries, extracts one
input per marked span, and later inserts labels predicted by EXAONE.
"""

from __future__ import annotations

import re
from collections import Counter


LABEL_TO_KOREAN = {
    "supportive": "지지적",
    "neutral": "중립적",
    "oppositional": "비판적",
}

TAG_TO_TYPE = {
    "제목": "headline",
    "도입부": "lead",
    "직접 인용구": "quotation",
    "결론부": "conclusion",
}

_TAG_NAME_PATTERN = r"제목|도입부|결론부|직접\s*인용구"
_OPEN_TAG_RE = re.compile(
    rf"<\s*(?!/)\s*(?P<tag>{_TAG_NAME_PATTERN})(?P<attrs>[^>]*)>",
    flags=re.IGNORECASE,
)
_TAG_RE = re.compile(
    rf"<\s*(?P<close>/)?\s*(?P<tag>{_TAG_NAME_PATTERN})(?P<attrs>[^>]*)>",
    flags=re.IGNORECASE,
)
_STANCE_ATTR_RE = re.compile(
    r"\s+입장\s*=\s*(?:\"[^\"]*\"|'[^']*')",
    flags=re.IGNORECASE,
)


def normalize_tag(tag: str) -> str:
    """Normalize whitespace in the only multi-word segment tag."""
    compact = re.sub(r"\s+", "", tag)
    return "직접 인용구" if compact == "직접인용구" else compact


def remove_segment_labels(markup: str) -> str:
    """Remove stance attributes while retaining every segment boundary."""

    def replace(match: re.Match) -> str:
        attrs = _STANCE_ATTR_RE.sub("", match.group("attrs"))
        return f"<{normalize_tag(match.group('tag'))}{attrs}>"

    cleaned = _OPEN_TAG_RE.sub(replace, markup or "")
    if re.search(r"입장\s*=", cleaned):
        raise ValueError("a stance attribute remains after label removal")
    return cleaned


def strip_segment_tags(text: str) -> str:
    """Return span text without the known structural tags."""
    return _TAG_RE.sub("", text).strip()


def extract_segments(markup: str, location: str) -> list[dict]:
    """Extract nested marked spans in their opening-tag order.

    Lead and conclusion spans can contain quotation spans.  Each outer span is
    sent once with the inner markup removed, and every quotation is also sent
    once as its own segment, matching the marked segment inventory.
    """
    stack: list[tuple[str, int, int]] = []
    records = []
    for match in _TAG_RE.finditer(markup or ""):
        tag = normalize_tag(match.group("tag"))
        if not match.group("close"):
            stack.append((tag, match.start(), match.end()))
            continue

        open_index = next(
            (index for index in range(len(stack) - 1, -1, -1)
             if stack[index][0] == tag),
            None,
        )
        if open_index is None:
            raise ValueError(f"closing tag without opening tag: {tag}")
        open_tag, open_start, content_start = stack.pop(open_index)
        text = strip_segment_tags(markup[content_start:match.start()])
        records.append(
            {
                "location": location,
                "type": TAG_TO_TYPE[open_tag],
                "tag": open_tag,
                "open_start": open_start,
                "text": text,
            }
        )

    if stack:
        raise ValueError(f"unclosed segment tag(s): {[entry[0] for entry in stack]}")

    records.sort(key=lambda record: record.pop("open_start"))
    for index, record in enumerate(records):
        record["segment_id"] = f"{location}:{index}"
    return records


def prepare_segment_row(row: dict) -> dict:
    """Convert one RoBERTa-labelled row to a label-free segment record."""
    title = remove_segment_labels(row.get("title_joa_icl", ""))
    body = remove_segment_labels(row.get("main_body_joa_icl", ""))
    segments = extract_segments(title, "title") + extract_segments(body, "body")
    if not segments:
        raise ValueError(f"item {row.get('id')} has no marked segments")
    return {
        "id": row["id"],
        "event_name": row.get("event_name", ""),
        "title_joa_unlabeled": title,
        "main_body_joa_unlabeled": body,
        "segments": segments,
    }


def annotate_markup(markup: str, location: str, predictions: list[dict]) -> str:
    """Insert zero-shot predictions into the label-free opening tags."""
    by_id = {entry["segment_id"]: entry["pred"] for entry in predictions}
    opening_index = 0

    def replace(match: re.Match) -> str:
        nonlocal opening_index
        segment_id = f"{location}:{opening_index}"
        opening_index += 1
        if segment_id not in by_id:
            raise ValueError(f"missing prediction for {segment_id}")
        label = by_id[segment_id]
        if label not in LABEL_TO_KOREAN:
            raise ValueError(f"invalid prediction for {segment_id}: {label}")
        tag = normalize_tag(match.group("tag"))
        attrs = _STANCE_ATTR_RE.sub("", match.group("attrs"))
        return f'<{tag}{attrs} 입장="{LABEL_TO_KOREAN[label]}">'

    annotated = _OPEN_TAG_RE.sub(replace, markup or "")
    expected = sum(1 for entry in predictions if entry["location"] == location)
    if opening_index != expected:
        raise ValueError(
            f"{location} has {opening_index} opening tags but {expected} predictions"
        )
    return annotated


def segment_prompt(issue: str, segment: str) -> str:
    """The fixed English zero-shot prompt used for every marked span."""
    return (
        "Classify the stance expressed by the following news segment toward "
        "the specified issue as one of supportive, neutral, or oppositional.\n\n"
        "- supportive: The segment supports, favors, or positively evaluates "
        "the issue.\n"
        "- neutral: The segment reports information or presents an attributed "
        "statement without expressing a clear supportive or oppositional "
        "position.\n"
        "- oppositional: The segment criticizes, rejects, or negatively "
        "evaluates the issue.\n\n"
        f"Issue: {issue}\n\n"
        f"News segment:\n{segment}\n\n"
        "Answer on the final line in exactly this format:\n\n"
        "Final stance: <supportive|neutral|oppositional>"
    )


def segment_type_counts(rows: list[dict]) -> Counter:
    return Counter(segment["type"] for row in rows for segment in row["segments"])

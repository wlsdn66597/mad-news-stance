"""Journalism-segment agents followed by majority vote or an aggregator.

Four agents independently read the headline, lead, conclusion, or quotations.
They keep that information boundary throughout debate: later rounds exchange
analyses, not the raw segments. By default, the final agent labels are combined
with the repository's existing majority rule. The earlier article-level
aggregator remains available as an explicit reproduction option.
"""
import re

from .consensus import current_final_majority, unique_majority
from .debate import run_debate as run_debate_engine
from .llm import build_messages, chat, strip_think


SEGMENT_ORDER = ("headline", "lead", "conclusion", "quotations")

SEGMENT_SYSTEM_PROMPTS = {
    "headline": (
        "You are the Headline agent. Analyze only how the headline frames the "
        "specified issue: what it asserts, emphasizes, presupposes, or leaves "
        "open. Do not invent information from the unseen article. In later "
        "rounds, keep this headline perspective while using peers' analyses as "
        "secondary evidence about the whole article."
    ),
    "lead": (
        "You are the Lead agent. Analyze only the opening paragraph's framing "
        "of the specified issue: the main claim, selected facts, and initial "
        "direction given to the reader. Do not invent information from unseen "
        "parts of the article. In later rounds, keep this lead perspective "
        "while using peers' analyses as secondary evidence."
    ),
    "conclusion": (
        "You are the Conclusion agent. Analyze only how the final paragraph "
        "closes the issue: which claim or consequence receives the last word "
        "and whether the ending reinforces, qualifies, or leaves the issue "
        "unresolved. In later rounds, keep this perspective while using peers' "
        "analyses as secondary evidence."
    ),
    "quotations": (
        "You are the Quotation agent. Analyze the quoted speakers, their "
        "positions, ordering, and attribution. A source's opinion is not the "
        "journalist's stance unless the article adopts or privileges it. Do "
        "not invent unshown context. In later rounds, keep this sourcing "
        "perspective while using peers' analyses as secondary evidence."
    ),
}

SEGMENT_EXCHANGE_TEMPLATE = (
    "The other segment agents produced the following analyses of the same "
    "article:\n\n{others}\n\n"
    "Reassess your article-level prediction using concrete information in "
    "their analyses, but retain your assigned segment perspective and your "
    "own segment evidence. Do not copy a majority label. Distinguish a quoted "
    "speaker's opinion from the journalist's stance. State what peer evidence "
    "changed or preserved your judgment. End with exactly one label line:\n"
    "Final stance: <supportive|oppositional|neutral>"
)

AGGREGATOR_SYSTEM_PROMPT = (
    "You aggregate complementary journalism-segment analyses into one news "
    "stance label. This is an article-level synthesis, not majority voting: "
    "weigh evidence by its relevance to the specified issue and distinguish "
    "the journalist's framing from attributed opinions."
)


def _paragraphs(article):
    """Return non-empty article blocks without silently discarding text."""
    text = str(article or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) == 1:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines
    return blocks


def extract_quotations(article):
    """Extract direct quotations and short attribution context deterministically."""
    text = str(article or "")
    patterns = (
        re.compile(r'“([^”]{2,1000})”', re.DOTALL),
        re.compile(r'"([^"\n]{2,1000})"'),
        re.compile(r'‘([^’]{2,1000})’', re.DOTALL),
    )
    found = []
    occupied = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            quote = re.sub(r"\s+", " ", match.group(1)).strip()
            if not quote:
                continue
            left = max(0, text.rfind(".", 0, span[0]) + 1, text.rfind("\n", 0, span[0]) + 1)
            right_candidates = [
                position for position in (text.find(".", span[1]), text.find("\n", span[1]))
                if position >= 0
            ]
            right = min(right_candidates) + 1 if right_candidates else min(len(text), span[1] + 120)
            context = re.sub(r"\s+", " ", text[left:right]).strip()
            found.append({"quote": quote, "context": context, "_start": span[0]})
            occupied.append(span)
    found.sort(key=lambda entry: entry["_start"])
    for entry in found:
        entry.pop("_start")
    return found


def split_journalism_segments(item):
    paragraphs = _paragraphs(item.get("article", ""))
    quotations = extract_quotations(item.get("article", ""))
    quote_text = (
        "\n".join(
            f"{index}. Quote: {entry['quote']}\n   Attribution context: {entry['context']}"
            for index, entry in enumerate(quotations, 1)
        )
        if quotations
        else "(No direct quotation was found in the article.)"
    )
    return {
        "headline": str(item.get("headline", "")).strip() or "(No headline available.)",
        "lead": paragraphs[0] if paragraphs else "(No lead paragraph available.)",
        "conclusion": paragraphs[-1] if paragraphs else "(No conclusion paragraph available.)",
        "quotations": quote_text,
    }


def build_segment_questions(item, segments=None):
    segments = segments or split_journalism_segments(item)
    questions = []
    for name in SEGMENT_ORDER:
        questions.append(
            "Classify the Korean news article's stance toward the specified issue "
            "as supportive, oppositional, or neutral, using your assigned segment.\n\n"
            f"Issue: {item['issue']}\n"
            f"Assigned segment ({name}):\n{segments[name]}\n\n"
            "Report the segment's local signal separately from your best article-level "
            "prediction. Absence of a signal in one segment does not by itself prove "
            "that the article is neutral. Answer in this form:\n"
            "Local signal: <supportive|oppositional|neutral|insufficient>\n"
            "Evidence: <one concise, segment-grounded sentence>\n"
            "Article-level reasoning: <one concise sentence>\n"
            "Final stance: <supportive|oppositional|neutral>"
        )
    return questions


def build_aggregator_prompt(item, final_analyses):
    analyses = "\n\n".join(
        f"[{name.title()} agent]\n{analysis}"
        for name, analysis in zip(SEGMENT_ORDER, final_analyses)
    )
    return (
        "Determine the article's stance toward the issue. Use the complete article "
        "to verify the four segment analyses. Do not decide by counting their labels; "
        "resolve conflicts using issue relevance, journalistic emphasis, attribution, "
        "and wording. End with exactly one label line.\n\n"
        f"Issue: {item['issue']}\n"
        f"Headline: {item.get('headline', '')}\n"
        f"Article (Korean):\n{item.get('article', '')}\n\n"
        f"Final segment analyses:\n{analyses}\n\n"
        "Final stance: <supportive|oppositional|neutral>"
    )


def run_segment_debate(
    model,
    tokenizer,
    task,
    item,
    max_new_tokens=1024,
    n_rounds=4,
    temperature=1.0,
    aggregator_temperature=None,
    aggregator_max_new_tokens=512,
    enable_thinking=None,
    decision_rule="majority",
    tie_seed=0,
):
    """Run segment agents and combine their final labels.

    ``majority`` matches the repository's existing final-round vote, including
    its legacy first-valid-label handling for a tie. ``aggregator`` reproduces
    the original segment experiment in which another model call reads the full
    article and the four final analyses.
    """
    if decision_rule not in {"majority", "aggregator"}:
        raise ValueError(f"unknown segment decision rule: {decision_rule}")

    segments = split_journalism_segments(item)
    agent_questions = build_segment_questions(item, segments)
    trace = run_debate_engine(
        model,
        tokenizer,
        question=f"Segment debate for item {item['id']}",
        agent_questions=agent_questions,
        debate_template=SEGMENT_EXCHANGE_TEMPLATE,
        system_prompt=[SEGMENT_SYSTEM_PROMPTS[name] for name in SEGMENT_ORDER],
        n_agents=len(SEGMENT_ORDER),
        n_rounds=n_rounds,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        debate_protocol="segment_reasoned_exchange",
    )
    final_analyses = trace["answers_by_round"][-1]
    agent_preds = [task.parse(answer) for answer in final_analyses]
    vote = unique_majority(agent_preds)

    aggregator = None
    raw = None
    if decision_rule == "majority":
        pred, decision_reason = current_final_majority(
            [agent_preds], item["id"], fallback_seed=tie_seed
        )
    else:
        aggregator_prompt = build_aggregator_prompt(item, final_analyses)
        aggregator_messages = build_messages(
            aggregator_prompt, system_prompt=AGGREGATOR_SYSTEM_PROMPT
        )
        effective_aggregator_temperature = (
            temperature if aggregator_temperature is None else aggregator_temperature
        )
        raw = strip_think(
            chat(
                model,
                tokenizer,
                aggregator_messages,
                max_new_tokens=aggregator_max_new_tokens,
                temperature=effective_aggregator_temperature,
                enable_thinking=enable_thinking,
            )
        )
        pred = task.parse(raw)
        decision_reason = "article_level_aggregator"
        aggregator = {
            "prompt": aggregator_prompt,
            "messages": aggregator_messages,
            "raw": raw,
            "pred": pred,
            "temperature": effective_aggregator_temperature,
            "max_new_tokens": aggregator_max_new_tokens,
        }

    aggregator_calls = int(decision_rule == "aggregator")
    return {
        "pred": pred,
        "raw": raw,
        "preds": agent_preds,
        "segments": segments,
        "segment_order": list(SEGMENT_ORDER),
        "debate_trace": trace,
        "decision": {
            "rule": decision_rule,
            "reason": decision_reason,
            "final_agent_labels": agent_preds,
            "label_counts": vote.counts,
            "tied": vote.tied,
        },
        "aggregator": aggregator,
        "call_counts": {
            "segment_agent_generation_calls": len(SEGMENT_ORDER) * n_rounds,
            "aggregator_generation_calls": aggregator_calls,
            "total_generation_calls": len(SEGMENT_ORDER) * n_rounds + aggregator_calls,
        },
    }

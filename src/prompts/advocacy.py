"""Prompts for stance-advocacy debate.

Each agent is assigned one of the three labels in advance and builds the
strongest article-grounded case for it; a judge then contrasts the three cases.

The design follows Tree-of-Counterfactual prompting (Weinzierl & Harabagiu,
ACL 2024), which generates one rationale per stance value and picks a winner
with contrastive verification. Two departures are deliberate:

1. Each advocate must also report the strongest counter-evidence and how well
   the article supports its assigned stance. Every advocate is fluent by
   construction, so the judge needs a signal other than persuasiveness.
2. The judge is told explicitly that the analyses are assigned advocacy rather
   than independent opinions, and the candidate order is shuffled per item
   (ToC does neither).

The rebuttal round mirrors PREDICT's second debate round (Park et al., EMNLP
2024), where each side may refute or concede after seeing the opposing case.
"""

STANCE_LABELS = ("supportive", "oppositional", "neutral")

SUPPORT_LEVELS = ("weak", "moderate", "strong")

ADVOCATE_SYSTEM_PROMPT = """You are an expert analyst of Korean news articles.

You will be asked to explain why an article may take a specified stance toward a
given issue. Judge only the stance expressed by the article's own framing,
wording, emphasis, and narrative structure. Distinguish the journalist's framing
from opinions that are merely quoted from governments, organizations, or
individuals.

Discuss your reasoning step by step. If the article provides little support for
the specified stance, say so explicitly rather than inventing evidence."""

ADVOCATE_USER_TEMPLATE = """Issue: {issue}
Headline: {headline}
Article:
{article}

Assigned stance: {stance}

Respond with these fields:
Article evidence (at most three short quoted passages):
Why this supports the assigned stance:
Strongest counter-evidence against the assigned stance:
Support for the assigned stance: <weak|moderate|strong>"""

REBUTTAL_TEMPLATE = """The other analysts argued as follows:

{others}

Based on your own analysis, agree with or rebut their arguments in one
paragraph. If their article-grounded evidence is stronger than yours, say so
explicitly. Then restate the final line in exactly this format:

Support for the assigned stance: <weak|moderate|strong>"""

PEER_TEMPLATE = "Analyst {index} (arguing for {stance}):\n{answer}"

JUDGE_SYSTEM_PROMPT = """You are an independent adjudicator for news stance classification.

Determine the article's stance toward the specified issue as supportive,
oppositional, or neutral.

Three analyses are provided. Each analyst was instructed in advance to argue for
a specific assigned stance, so these are advocacy arguments, not evidence.
Verify every claimed passage against the original article before relying on it.

Judge the stance expressed by the article's framing, wording, emphasis, and
narrative structure. Distinguish the journalist's framing from opinions that are
merely quoted from governments, organizations, or individuals.

Discuss the strengths and weaknesses of each analysis, then decide. Do not infer
the answer from the order in which the analyses appear, and do not follow an
analysis merely because it is more forcefully written. Return one of the three
labels even when all analyses are flawed.

Return valid JSON only."""

JUDGE_SCHEMA = {
    "label": "supportive|oppositional|neutral",
    "evidence_sufficient": True,
    "evidence": ["short article-grounded evidence"],
    "rationale": "brief rationale without hidden chain-of-thought",
}


def advocate_question(item, stance: str) -> str:
    if stance not in STANCE_LABELS:
        raise ValueError(f"unknown stance: {stance}")
    return ADVOCATE_USER_TEMPLATE.format(
        issue=item.get("issue", ""),
        headline=item.get("headline", ""),
        article=item.get("article", ""),
        stance=stance,
    )


def format_peers(peers) -> str:
    """peers: sequence of (stance, answer) already ordered for delivery."""
    return "\n\n".join(
        PEER_TEMPLATE.format(index=index, stance=stance, answer=answer)
        for index, (stance, answer) in enumerate(peers, start=1)
    )

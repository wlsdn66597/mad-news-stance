"""Prompts for stance-advocacy debate.

Each agent is assigned one of the three labels in advance and builds the
strongest article-grounded case for it. The agents then exchange cases for one
or more rebuttal rounds, and a judge contrasts the final cases and decides.

Backbone: Tree-of-Counterfactual prompting (Weinzierl & Harabagiu, ACL 2024),
which generates one rationale per stance value and picks a winner with
contrastive verification. Round structure: PREDICT (Park et al., EMNLP 2024),
which runs a fixed two-round debate and then hands the history to a judge.

The published prompts are short — ToC's advocate system prompt is 34 words and
its judge 82, PREDICT's debater 32, MAD's 16 — and this repository's own stance
profile is 32. `toc` keeps that scale so a gain cannot be attributed to a longer
instruction; `structured` is the variant that spells out the evidence fields and
tells the judge the analyses are assigned advocacy.

Neither style asks an advocate to rate its own confidence. Self-reported
confidence is not used by any of the source papers and was not carrying its
weight in prompt length here.
"""

STANCE_LABELS = ("supportive", "oppositional", "neutral")

LABEL_LINE = "Final stance: <supportive|oppositional|neutral>"

# ---------------------------------------------------------------- toc style

TOC_ADVOCATE_SYSTEM_PROMPT = """You are an expert linguistic assistant.
You will be tasked with explaining why a news article may have a stance towards a provided issue.
You should discuss your reasoning in detail, thinking step-by-step."""

TOC_ADVOCATE_USER_TEMPLATE = """News Article: {headline}
{article}
Issue: {issue}
Stance: {stance}"""

TOC_REBUTTAL_TEMPLATE = """The other analysts argued as follows:

{others}

Based on your own analysis, agree with or rebut their arguments and explain your reason."""

TOC_JUDGE_SYSTEM_PROMPT = """You are an expert linguistic assistant.
You will be tasked with judging which stance value a news article has towards a provided issue.
Thorough rationales will be provided for each stance value.
You should discuss your reasoning in detail, thinking step-by-step.
Discuss the strengths and weaknesses for each rationale, providing a final judgement for the stance value of the article towards the provided issue.
Your final sentence should include only one possible stance value: supportive, oppositional, or neutral"""

TOC_JUDGE_USER_TEMPLATE = """News Article: {headline}
{article}
Issue: {issue}
{analyses}"""

TOC_JUDGE_ANALYSIS_TEMPLATE = "Stance: {stance} Rationale: {analysis}"

TOC_JUDGE_REPAIR_TEMPLATE = """Your previous answer did not name a stance value.

{invalid_output}

Answer again. Discuss the strengths and weaknesses of each rationale, and make
your final sentence include only one possible stance value: supportive,
oppositional, or neutral."""

# ------------------------------------------------------------- our variant

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
Strongest counter-evidence against the assigned stance:"""

REBUTTAL_TEMPLATE = """The other analysts argued as follows:

{others}

Based on your own analysis, agree with or rebut their arguments in one
paragraph. If their article-grounded evidence is stronger than yours, say so
explicitly."""

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

PEER_TEMPLATE = "Analyst {index} (arguing for {stance}):\n{answer}"


PROMPT_STYLES = {
    "toc": {
        "advocate_system": TOC_ADVOCATE_SYSTEM_PROMPT,
        "advocate_user": TOC_ADVOCATE_USER_TEMPLATE,
        "rebuttal": TOC_REBUTTAL_TEMPLATE,
        "judge_system": TOC_JUDGE_SYSTEM_PROMPT,
        "judge_input": "text",
        "judge_user": TOC_JUDGE_USER_TEMPLATE,
        "judge_analysis": TOC_JUDGE_ANALYSIS_TEMPLATE,
        "judge_output": "final_line",
        "judge_repair": TOC_JUDGE_REPAIR_TEMPLATE,
    },
    "structured": {
        "advocate_system": ADVOCATE_SYSTEM_PROMPT,
        "advocate_user": ADVOCATE_USER_TEMPLATE,
        "rebuttal": REBUTTAL_TEMPLATE,
        "judge_system": JUDGE_SYSTEM_PROMPT,
        "judge_input": "json",
        "judge_user": None,
        "judge_analysis": None,
        "judge_output": "json",
        "judge_repair": None,
    },
}


def get_prompt_style(name: str) -> dict:
    try:
        return PROMPT_STYLES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"unknown prompt style {name!r}; choose one of: {choices}") from exc


def advocate_question(item, stance: str, style: str = "toc") -> str:
    if stance not in STANCE_LABELS:
        raise ValueError(f"unknown stance: {stance}")
    return get_prompt_style(style)["advocate_user"].format(
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

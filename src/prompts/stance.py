"""Prompt profiles for Korean news stance detection.

The legacy prompt is preserved for comparison. The v2 profiles encode the same
decision protocol in English and Korean so prompt language can be ablated
without changing the underlying task definition.
"""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple


LABEL_LINE_EN = "Final stance: <supportive|oppositional|neutral>"
LABEL_LINE_KO = "최종 입장: <supportive|oppositional|neutral>"


@dataclass(frozen=True)
class StancePromptProfile:
    name: str
    language: str
    cot_instruction: str
    vanilla_instruction: str
    debate_template: str
    # the self-refine control: the same round structure with the agent's own
    # previous answer in place of its peers'
    self_refine_template: Optional[str] = None
    # named alternatives to debate_template, each a tuple of one template per
    # exchange round (a single entry repeats for every round)
    debate_protocols: Optional[Dict[str, Tuple[str, ...]]] = None
    memory_summary_template: Optional[str] = None
    memory_debate_template: Optional[str] = None
    instruction_first: bool = False
    article_heading_en: str = "Article (Korean)"
    other_agent_template: str = "[Agent {index}]\n{answer}"

    def article_block(self, item):
        if self.language == "ko":
            return (
                f"이슈: {item['issue']}\n"
                f"제목: {item['headline']}\n"
                f"기사:\n{item['article']}"
            )
        return (
            f"Issue: {item['issue']}\n"
            f"Headline: {item['headline']}\n"
            f"{self.article_heading_en}:\n{item['article']}"
        )

    def question(self, item, style="cot"):
        instruction = self.cot_instruction if style == "cot" else self.vanilla_instruction
        if self.instruction_first:
            return f"{instruction}\n\n{self.article_block(item)}"
        return f"{self.article_block(item)}\n\n{instruction}"

    def memory_context(self, item):
        return self.article_block(item)

    def format_other_answers(self, answers):
        return "\n\n".join(
            self.other_agent_template.format(index=index, answer=answer)
            for index, answer in enumerate(answers, start=1)
        )

LEGACY_KO = StancePromptProfile(
    name="legacy_ko",
    language="ko",
    cot_instruction=(
        "위 기사가 해당 이슈에 대해 취하는 입장을 판단하라. "
        "supportive(지지/찬성), oppositional(반대/비판), neutral(중립) 중 하나다. "
        "인용문 화자의 입장이 아니라 기사 자체의 논조를 기준으로 판단하라.\n\n"
        "먼저 핵심 근거 문장을 짚은 뒤 판단하고, 마지막 줄에 반드시 "
        "'최종 입장: <supportive|oppositional|neutral>' 형식으로 셋 중 하나만 영어로 써라."
    ),
    vanilla_instruction=(
        "위 기사가 해당 이슈에 대해 취하는 입장을 판단하라. "
        "supportive(지지/찬성), oppositional(반대/비판), neutral(중립) 중 하나다. "
        "인용문 화자의 입장이 아니라 기사 자체의 논조를 기준으로 판단하라.\n\n"
        "마지막 줄에 반드시 '최종 입장: <supportive|oppositional|neutral>' "
        "형식으로 셋 중 하나만 영어로 써라."
    ),
    debate_template=(
        "다음은 다른 에이전트들이 같은 기사에 대해 내놓은 판단입니다:\n\n{others}\n\n"
        "다른 에이전트들의 추론을 추가 조언으로 참고하여, 당신의 판단과 다른 "
        "에이전트들의 판단을 단계별로 검토한 뒤 갱신된 답을 제시하라. "
        "마지막 줄에 반드시 '최종 입장: <supportive|oppositional|neutral>' "
        "형식으로 답하라."
    ),
)


STANCE_V2_EN = StancePromptProfile(
    name="stance_v2_en",
    language="en",
    cot_instruction=(
        "Classify the stance taken by the news article itself toward the specified issue.\n\n"
        "Labels:\n"
        "- supportive: the article endorses, favors, or positively justifies the issue, "
        "policy, proposal, or claim.\n"
        "- oppositional: the article rejects, criticizes, or frames it negatively.\n"
        "- neutral: the article mainly reports facts or presents competing views without "
        "a clear authorial direction.\n\n"
        "Decision protocol:\n"
        "1. Restate the issue as the proposition being evaluated.\n"
        "2. Identify at most two short passages that reveal the journalist's or outlet's "
        "own framing.\n"
        "3. Separate authorial narration, headline framing, and editorial wording from "
        "statements merely attributed to quoted speakers.\n"
        "4. Note the strongest evidence for a competing label.\n"
        "5. Choose the label supported by the article's overall framing. Do not treat a "
        "speaker's stance as the article's stance.\n\n"
        "Respond with these fields:\n"
        "Issue proposition:\nAuthorial evidence:\nQuoted-source evidence:\n"
        "Counter-evidence:\nDecision rationale:\n"
        + LABEL_LINE_EN
    ),
    vanilla_instruction=(
        "Classify the stance taken by the news article itself toward the specified issue "
        "as supportive, oppositional, or neutral. Distinguish the journalist's framing "
        "from the views of quoted speakers. A speaker's stance is not automatically the "
        "article's stance. End with exactly:\n"
        + LABEL_LINE_EN
    ),
    debate_template=(
        "Below are other agents' analyses of the same Korean news article:\n\n"
        "{others}\n\n"
        "Audit the analyses against the original issue and article already in your "
        "context. Follow this protocol:\n"
        "1. Verify that every claimed passage or fact actually appears in the article.\n"
        "2. Distinguish authorial or headline framing from language attributed to a "
        "quoted speaker.\n"
        "3. Identify which competing label has the strongest article-grounded evidence.\n"
        "4. Agent vote counts and confidence are not evidence. Do not follow the majority "
        "merely because it is the majority.\n"
        "5. Change your previous label only when a concrete passage or framing cue "
        "warrants the change; otherwise retain it.\n\n"
        "Respond with:\nPrevious stance:\nValid new evidence:\nRejected evidence and why:\n"
        "Changed stance: <yes|no>\n"
        + LABEL_LINE_EN
    ),
    memory_summary_template=(
        "You are a shared evidence-memory writer for a multi-agent Korean news stance "
        "debate. Compress the agent responses into a neutral evidence ledger. Do not "
        "decide the final label, solve the classification yourself, count votes as "
        "evidence, or force consensus.\n\n"
        "Original task context:\n{task_context}\n\n"
        "Latest agent responses:\n{responses}\n\n"
        "Verify proposed evidence against the original article and return only these "
        "sections:\n"
        "[Candidate stances] Agent labels and their candidate labels.\n"
        "[Authorial evidence] Short exact passages showing headline, journalist, or outlet "
        "framing, with source agent labels.\n"
        "[Quoted-source evidence] Speaker-attributed passages that must not automatically "
        "be treated as the article's stance.\n"
        "[Counter-evidence] Evidence supporting competing labels.\n"
        "[Errors or unsupported claims] Claims not found in the article, attribution "
        "mistakes, or reasoning conflicts.\n"
        "[Unresolved points] Questions that agents should re-check.\n"
        "Preserve minority evidence when it is concrete and article-grounded. Be concise."
    ),
    memory_debate_template=(
        "A shared memory writer compressed the latest agent responses into this evidence "
        "ledger:\n\n{memory}\n\n"
        "Treat the ledger as an index of claims, not as an answer. Re-check each relevant "
        "claim against the original Korean article and your previous analysis. Vote counts "
        "and confidence are not evidence. Change your label only when a concrete article "
        "passage or framing cue warrants it; otherwise retain your previous label.\n\n"
        "Respond with:\nPrevious stance:\nDecisive verified evidence:\n"
        "Rejected or misattributed evidence:\nChanged stance: <yes|no>\n"
        + LABEL_LINE_EN
    ),
)


STANCE_V2_EN_GENERIC_MEMORY = replace(
    STANCE_V2_EN,
    name="stance_v2_en_generic_memory",
    memory_summary_template=None,
    memory_debate_template=None,
)


# ------------------------------------------------------------- debate protocols
#
# Two alternatives to the baseline exchange, both for `stance_minimal_en` and
# both keeping the persona system prompts, the round count, the call count and
# the aggregation untouched. Only the user prompt of rounds 1..n-1 changes, so a
# difference between the three runs is the protocol.
#
# `evidence_gated` repeats one prompt that permits a stance change only against
# stronger article-grounded evidence. Round 0 already agrees 72.0% of the time
# and the exchange drives that to 84.7% while the candidate pool falls from
# 0.6543 to 0.6014; the gate is the intervention on whether that convergence is
# consensus or conformity.
#
# `round_specific` gives each round one cognitive task instead of asking a 1.2B
# model to audit, rebut, compare and decide in a single pass: verify against the
# article, then test the strongest contrary reading, then settle. It needs
# exactly four rounds.
_MINIMAL_EN_OTHERS = "\n\nThe other agents' judgments are as follows:\n\n{others}"

STANCE_MINIMAL_EN_EVIDENCE_GATED = (
    "Review the other agents' judgments while keeping your assigned analysis "
    "perspective.\n"
    "Check their claims against the original article.\n"
    "Focus on evidence relevant to your perspective.\n"
    "Change your stance only if another agent provides stronger article-grounded "
    "evidence. Otherwise, keep your previous stance.\n"
    "Do not change your answer merely to agree with the other agents.\n"
    "Briefly state the decisive evidence for keeping or changing your judgment, "
    "and answer on the final line in exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

STANCE_MINIMAL_EN_ROUND1_EVIDENCE = (
    "Review the other agents' judgments.\n"
    "Your only task in this round is to check their claims against the original "
    "article from your assigned analysis perspective.\n"
    "Identify the single strongest article-grounded point that supports or "
    "contradicts your current stance. Do not follow the majority and do not "
    "evaluate arguments by persuasiveness.\n"
    "Briefly state the verified evidence, and answer on the final line in exactly "
    "this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

STANCE_MINIMAL_EN_ROUND2_COUNTER = (
    "Review the other agents' latest judgments and the evidence identified so "
    "far.\n"
    "Your only task in this round is to test your current stance against the "
    "strongest article-grounded evidence for a different stance.\n"
    "Identify one piece of contrary evidence, if any.\n"
    "Change your stance only if that evidence is stronger than the evidence "
    "supporting your current judgment.\n"
    "Briefly state the decisive comparison, and answer on the final line in "
    "exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

STANCE_MINIMAL_EN_ROUND3_FINAL = (
    "Make your final judgment using only the article-grounded evidence identified "
    "during the debate.\n"
    "Keep your assigned analysis perspective, but consider the strongest verified "
    "evidence raised by the other agents.\n"
    "Do not follow the majority simply because the agents agree.\n"
    "Select the stance best supported by the original article.\n"
    "Briefly state the decisive evidence, and answer on the final line in exactly "
    "this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

# --------------------------------------------------------------- and their v2
#
# Measured on test s6000, both v1 protocols failed in a specific way.
#
# The gate froze completely: 0 of 3003 agent labels moved at any of the three
# exchange rounds. "Otherwise, keep your previous stance" is a default that
# costs nothing to obey, so the model satisfied the instruction by restating
# its label. v2 removes the default and asks instead for something only reading
# can produce -- the deciding passage, quoted, checked against the article.
#
# The round-specific R2 collapsed to neutral: accuracy 0.4975 -> 0.3976, neutral
# recall 0.3121 -> 0.8061, net -101 (p<0.001), which the final round then mostly
# undid. Asked to look for contrary evidence, the model read "evidence on both
# sides" as "neutral" -- neutral used as abstention rather than as a class. v2
# says what neutral is, and R3 repeats it so the two rounds cannot disagree.
# R1 also barely moved anything (9 items), so v2 gives it the same quote
# requirement and it feeds verified passages into the later rounds.
#
# The gate keeps a single change so the de-freeze can be attributed on its own;
# if it moves again but neutral recall stays flat, the neutral clause is the
# next thing to add.
STANCE_MINIMAL_EN_EVIDENCE_GATED_V2 = (
    "Review the other agents' judgments while keeping your assigned analysis "
    "perspective.\n"
    "Check their claims against the original article: quote the one passage "
    "that decides the question, and say whether it appears in the article as "
    "written.\n"
    "Vote counts are not evidence. Do not change your answer merely to agree "
    "with the other agents; change it only when a passage another agent cites "
    "is stronger than the one supporting your current judgment.\n"
    "Briefly state the decisive passage, and answer on the final line in "
    "exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

STANCE_MINIMAL_EN_ROUND1_EVIDENCE_V2 = (
    "Review the other agents' judgments.\n"
    "Your only task in this round is to check their claims against the original "
    "article from your assigned analysis perspective.\n"
    "Quote the single strongest passage that supports or contradicts your "
    "current stance, and say whether it appears in the article as written. Do "
    "not follow the majority and do not evaluate arguments by "
    "persuasiveness.\n"
    "Briefly state the verified passage, and answer on the final line in "
    "exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

_NEUTRAL_IS_NOT_A_TIE = (
    "Neutral is not a tie-breaker: choose neutral only if the article itself "
    "takes no side, never because the evidence points both ways.\n"
)

STANCE_MINIMAL_EN_ROUND2_COUNTER_V2 = (
    "Review the other agents' latest judgments and the passages verified so "
    "far.\n"
    "Your only task in this round is to test your current stance against the "
    "strongest article-grounded evidence for a different stance.\n"
    "Name the one competing stance with the best support, and the single "
    "passage that supports it.\n"
    + _NEUTRAL_IS_NOT_A_TIE
    + "Change your stance only if that passage outweighs the evidence for your "
    "current judgment.\n"
    "Briefly state the decisive comparison, and answer on the final line in "
    "exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

STANCE_MINIMAL_EN_ROUND3_FINAL_V2 = (
    "Make your final judgment using only the article-grounded passages verified "
    "during the debate.\n"
    "Keep your assigned analysis perspective, but consider the strongest "
    "verified passage raised by the other agents.\n"
    "Do not follow the majority simply because the agents agree.\n"
    + _NEUTRAL_IS_NOT_A_TIE
    + "Select the stance best supported by the original article.\n"
    "Briefly state the decisive passage, and answer on the final line in "
    "exactly this format:\n\n"
    + LABEL_LINE_EN
    + _MINIMAL_EN_OTHERS
)

# ------------------------------------------------------- filling the channel
#
# Measured on test s6000: 90%+ of what an agent writes is the label alone --
# round-0 answers have a median of 24 characters, the length of "Final stance:
# supportive", and persona rounds 1-3 stay there. Since each round pastes the
# other agents' previous answers into the prompt verbatim, an agent's only new
# information from its peers is how they voted. Published MAD assumes it reads
# their reasoning; here there is none to read, so what looks like debate can
# only be majority pressure -- which is what unanimity 595 -> 784 against a net
# of +12 items looks like.
#
# This protocol changes what travels between agents and nothing else. It does
# not say which way to decide, unlike the two above: no default stance, no
# definition of neutral, no instruction about the majority. It demands one
# quoted sentence before the label, and it puts the format demand *after* the
# peers' answers rather than before them, because in the baseline template the
# demand sits far from the point of generation and is ignored.
#
# The test is the label-only rate at rounds 2 and 3, where peers first see an
# evidence line. Accuracy is secondary: if the channel cannot be filled at this
# model size, that is the finding, and it separates "MAD does not help here"
# from "MAD never ran here".
STANCE_MINIMAL_EN_REASONED_EXCHANGE = (
    "Use the other agents' responses as additional information and reconsider "
    "your previous judgment.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    "Answer in exactly two lines, in this order:\n"
    "Evidence: <one sentence, quoting the wording in the article that decides "
    "your judgment>\n"
    + LABEL_LINE_EN
)

# The channel fills -- label-only drops from 95% to 0.5% and the evidence lines
# are real Korean quotations -- but 341 of 3003 round-1 answers carry no label
# at all, and 210 at round 3. Those answers have a median of 87 characters
# against a 1024-token budget, so nothing was truncated: the agent writes the
# Evidence line and stops. Each one drops an agent out of a three-way vote, so
# the +23 at the first exchange round was measured while losing 7-11% of the
# ballots. v2 keeps the ordering, since evidence before the label is the point,
# and anchors the closing line instead.
STANCE_MINIMAL_EN_REASONED_EXCHANGE_V2 = (
    "Use the other agents' responses as additional information and reconsider "
    "your previous judgment.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    "Answer with both of these lines, in this order and nothing else:\n"
    "Evidence: <at most 25 words, quoting the wording in the article that "
    "decides your judgment>\n"
    + LABEL_LINE_EN
    + "\nThe last line of your answer must begin with \"Final stance:\". Do not "
    "stop before it."
)

# --------------------------------------------- what the channel should carry
#
# Only reasoned_exchange ever filled the channel. The four protocols above it
# all asked for reasoning and all got the label alone: mean answer length by
# round was 24/24/24/24 for the gate, 24/25/23/24 for round_specific, against
# 24/96/88/87 once the format demand moved after the peers' answers. So neither
# the gate nor the stage decomposition was ever exercised -- there was nothing
# in the channel for a gate to weigh or a verification stage to check, and
# their failures say only that those prompts did not change what agents emit.
#
# These three re-ask both questions on top of the format that works, and add
# the length question the short form raises. reasoned_exchange is the control
# for both axes: _long differs from it only in how much travels, the two _v3
# protocols only in what the instruction asks for.
_REASONED_CLOSE = (
    "Answer with both of these lines, in this order and nothing else:\n"
    "Evidence: <one sentence, quoting the wording in the article that decides "
    "your judgment>\n"
    + LABEL_LINE_EN
    + "\nThe last line of your answer must begin with \"Final stance:\". Do not "
    "stop before it."
)

# a quotation is a citation, not an argument: it does not say why the wording
# was read that way, and confusing the article's own framing with a quoted
# speaker's is the failure this task turns on
STANCE_MINIMAL_EN_REASONED_LONG = (
    "Use the other agents' responses as additional information and reconsider "
    "your previous judgment.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    "Answer in this format, and nothing else:\n"
    "Evidence: <quote the wording in the article that decides your judgment>\n"
    "Reasoning: <2 to 3 sentences: whether that wording is the article's own "
    "framing or a quoted speaker's, and where the other agents' readings "
    "differ from yours>\n"
    + LABEL_LINE_EN
    + "\nThe last line of your answer must begin with \"Final stance:\". Do not "
    "stop before it."
)

# the gate, now that there is something to gate on. "Do not change merely to
# agree" is dropped: v1 and v2 both froze completely, and a bare prohibition is
# the part a 1.2B can satisfy by never moving. What remains is the comparison
# it was supposed to make.
STANCE_MINIMAL_EN_EVIDENCE_GATED_V3 = (
    "Use the other agents' responses as additional information and reconsider "
    "your previous judgment.\n"
    "Change your stance only when another agent's quoted wording is stronger "
    "than the wording supporting your current judgment. Vote counts are not "
    "evidence.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    + _REASONED_CLOSE
)

# the stages, now that each has something to work on. The v2 neutral clause is
# deliberately left out: it moved neutral recall from 0.5507 to 0.0290, so its
# effect is already known and including it would confound the stage question.
STANCE_MINIMAL_EN_ROUND1_EVIDENCE_V3 = (
    "Review the other agents' judgments.\n"
    "Your only task in this round is to check their quoted wording against the "
    "original article, and to quote the wording that decides your own "
    "judgment.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    + _REASONED_CLOSE
)

STANCE_MINIMAL_EN_ROUND2_COUNTER_V3 = (
    "Review the other agents' latest judgments.\n"
    "Your only task in this round is to test your current stance against the "
    "strongest competing wording quoted so far, and to say which of the two is "
    "stronger.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    + _REASONED_CLOSE
)

STANCE_MINIMAL_EN_ROUND3_FINAL_V3 = (
    "Make your final judgment using the wording quoted during the debate.\n"
    "Do not look for new evidence in this round. Decide between the passages "
    "already on the table.\n\n"
    "The other agents' judgments are as follows:\n\n"
    "{others}\n\n"
    + _REASONED_CLOSE
)

STANCE_MINIMAL_EN_PROTOCOLS = {
    "evidence_gated": (STANCE_MINIMAL_EN_EVIDENCE_GATED,),
    "round_specific": (
        STANCE_MINIMAL_EN_ROUND1_EVIDENCE,
        STANCE_MINIMAL_EN_ROUND2_COUNTER,
        STANCE_MINIMAL_EN_ROUND3_FINAL,
    ),
    "reasoned_exchange": (STANCE_MINIMAL_EN_REASONED_EXCHANGE,),
    "reasoned_exchange_v2": (STANCE_MINIMAL_EN_REASONED_EXCHANGE_V2,),
    "reasoned_exchange_long": (STANCE_MINIMAL_EN_REASONED_LONG,),
    "evidence_gated_v3": (STANCE_MINIMAL_EN_EVIDENCE_GATED_V3,),
    "round_specific_v3": (
        STANCE_MINIMAL_EN_ROUND1_EVIDENCE_V3,
        STANCE_MINIMAL_EN_ROUND2_COUNTER_V3,
        STANCE_MINIMAL_EN_ROUND3_FINAL_V3,
    ),
    "evidence_gated_v2": (STANCE_MINIMAL_EN_EVIDENCE_GATED_V2,),
    "round_specific_v2": (
        STANCE_MINIMAL_EN_ROUND1_EVIDENCE_V2,
        STANCE_MINIMAL_EN_ROUND2_COUNTER_V2,
        STANCE_MINIMAL_EN_ROUND3_FINAL_V2,
    ),
}


STANCE_MINIMAL_EN = StancePromptProfile(
    name="stance_minimal_en",
    language="en",
    cot_instruction=(
        "Classify this article's stance toward the specified issue as one of "
        "supportive, oppositional, or neutral.\n"
        "Briefly explain your reasoning, and answer on the final line in exactly "
        "this format:\n\n"
        + LABEL_LINE_EN
    ),
    vanilla_instruction=(
        "Classify this article's stance toward the specified issue as one of "
        "supportive, oppositional, or neutral. Answer on the final line in exactly "
        "this format:\n\n"
        + LABEL_LINE_EN
    ),
    debate_template=(
        "Use the other agents' responses as additional information and reconsider "
        "your previous judgment.\n"
        "Briefly explain your reasoning, and answer on the final line in exactly "
        "this format:\n\n"
        + LABEL_LINE_EN
        + "\n\nThe other agents' judgments are as follows:\n\n{others}"
    ),
    # two phrases away from debate_template, so a difference between the two runs
    # is the peer signal and not the wording
    self_refine_template=(
        "Use your own earlier response as additional information and reconsider "
        "your previous judgment.\n"
        "Briefly explain your reasoning, and answer on the final line in exactly "
        "this format:\n\n"
        + LABEL_LINE_EN
        + "\n\nYour previous judgment is as follows:\n\n{others}"
    ),
    debate_protocols=STANCE_MINIMAL_EN_PROTOCOLS,
    instruction_first=True,
    article_heading_en="Article",
    other_agent_template="Agent {index}:\n{answer}",
)

# ------------------------------------------------ neutral as a gate, not a class
#
# `stance_minimal_en` asks for one choice among three. Measured on the test
# split, EXAONE-4.0-1.2B then recovers 90% of supportive articles and 25% of
# neutral ones, while picking the right polarity on 74% of the articles that do
# take a side. It can tell the sides apart; it cannot tell that an article takes
# no side, so it falls back on the majority polarity. Perfect neutral detection
# with its current polarity calls would move it from 0.5035 to 0.7522, more than
# any aggregation rule or debate protocol measured here can reach.
#
# These two profiles ask the same question as two decisions instead of three
# options. They are not length-matched to the 32-word baseline: `twostep` is 53
# words and `gate` is 60, because the gate variant exists precisely to add the
# definition of "takes no side". Length is a live confound in principle; the
# judge-prompt sweep on this repository found 19, 32, 34 and 64 word
# instructions within five items of each other on 1001, every pairwise p at or
# above 0.405, which is the reason for not paying to control it here.

STANCE_TWOSTEP_EN = StancePromptProfile(
    name="stance_twostep_en",
    language="en",
    cot_instruction=(
        "Answer two questions about this article, in order.\n"
        "1. Does the article itself take a side on the issue, or does it not?\n"
        "2. If it does, which side: supportive or oppositional?\n"
        "Briefly explain, then answer on the final line in exactly this format "
        "(neutral if it takes no side):\n\n"
        + LABEL_LINE_EN
    ),
    vanilla_instruction=(
        "Answer two questions about this article, in order. Does the article "
        "itself take a side on the issue? If it does, which side: supportive or "
        "oppositional? Answer on the final line in exactly this format (neutral "
        "if it takes no side):\n\n"
        + LABEL_LINE_EN
    ),
    debate_template=(
        "Use the other agents' responses as additional information and reconsider, "
        "in order: does the article take a side, and if so which one?\n"
        "Briefly explain, then answer on the final line in exactly this format "
        "(neutral if it takes no side):\n\n"
        + LABEL_LINE_EN
        + "\n\nThe other agents' judgments are as follows:\n\n{others}"
    ),
    instruction_first=True,
    article_heading_en="Article",
    other_agent_template="Agent {index}:\n{answer}",
)

# The same gate, plus the two ways an article takes no side. Without them the
# model has to infer what "no side" means, and it resolves that by not using the
# option.
STANCE_GATE_EN = StancePromptProfile(
    name="stance_gate_en",
    language="en",
    cot_instruction=(
        "First: does the article's own framing argue for or against the issue? "
        "An article that only reports what others say, or that gives both sides "
        "comparable weight, takes no side.\n"
        "Only if it takes a side, decide which one.\n"
        "Briefly explain, then answer on the final line in exactly this format "
        "(neutral if it takes no side):\n\n"
        + LABEL_LINE_EN
    ),
    vanilla_instruction=(
        "First: does the article's own framing argue for or against the issue? "
        "An article that only reports what others say, or that gives both sides "
        "comparable weight, takes no side. Only if it takes a side, decide which "
        "one. Answer on the final line in exactly this format (neutral if it "
        "takes no side):\n\n"
        + LABEL_LINE_EN
    ),
    debate_template=(
        "Use the other agents' responses as additional information and reconsider. "
        "First, does the article's own framing argue for or against the issue, or "
        "does it only report what others say? Only if it takes a side, decide "
        "which one.\n"
        "Briefly explain, then answer on the final line in exactly this format "
        "(neutral if it takes no side):\n\n"
        + LABEL_LINE_EN
        + "\n\nThe other agents' judgments are as follows:\n\n{others}"
    ),
    instruction_first=True,
    article_heading_en="Article",
    other_agent_template="Agent {index}:\n{answer}",
)

STANCE_V2_KO = StancePromptProfile(
    name="stance_v2_ko",
    language="ko",
    cot_instruction=(
        "기사 자체가 지정된 이슈에 대해 취하는 입장을 분류하라.\n\n"
        "라벨 정의:\n"
        "- supportive: 기사 자체가 이슈·정책·주장을 지지하거나 긍정적으로 정당화함\n"
        "- oppositional: 기사 자체가 이슈를 반대·비판하거나 부정적으로 프레이밍함\n"
        "- neutral: 사실 전달이나 상반된 견해 소개가 중심이고 명확한 기사 논조가 없음\n\n"
        "판단 절차:\n"
        "1. 평가 대상 이슈를 하나의 명제로 다시 적는다.\n"
        "2. 기자 또는 매체의 논조를 보여주는 짧은 문구를 최대 두 개 찾는다.\n"
        "3. 기자 서술·제목 프레이밍·논평과 인용된 화자의 발언을 구분한다.\n"
        "4. 다른 라벨을 지지하는 가장 강한 반대 근거도 확인한다.\n"
        "5. 기사 전체 프레이밍에 근거해 판단하며, 인용 화자의 입장을 기사 자체의 "
        "입장으로 간주하지 않는다.\n\n"
        "다음 형식으로 답하라:\n이슈 명제:\n기사 자체의 근거:\n인용 화자의 근거:\n"
        "반대 근거:\n판단 이유:\n"
        + LABEL_LINE_KO
    ),
    vanilla_instruction=(
        "기사 자체가 해당 이슈에 대해 취하는 입장을 supportive, oppositional, neutral "
        "중 하나로 분류하라. 기자의 프레이밍과 인용 화자의 견해를 구분하고, 인용 "
        "화자의 입장을 기사 자체의 입장으로 간주하지 마라. 마지막 줄은 정확히 "
        "다음 형식으로 작성하라:\n"
        + LABEL_LINE_KO
    ),
    debate_template=(
        "다음은 같은 한국어 뉴스 기사에 대한 다른 에이전트들의 분석이다:\n\n"
        "{others}\n\n"
        "기존 대화에 있는 원래 이슈와 기사에 비추어 다음 절차로 검증하라:\n"
        "1. 제시된 문구와 사실이 실제 기사에 존재하는지 확인한다.\n"
        "2. 기자·제목의 프레이밍과 인용 화자의 발언을 구분한다.\n"
        "3. 서로 다른 라벨 중 기사 근거가 가장 강한 것을 찾는다.\n"
        "4. 에이전트 수와 자신감은 근거가 아니며 단순히 다수 의견을 따르지 않는다.\n"
        "5. 구체적인 기사 문구나 프레이밍 근거가 있을 때만 기존 라벨을 변경하고, "
        "그렇지 않으면 유지한다.\n\n"
        "다음 형식으로 답하라:\n이전 입장:\n새롭게 확인한 유효 근거:\n"
        "배제한 근거와 이유:\n입장 변경: <예|아니오>\n"
        + LABEL_LINE_KO
    ),
    memory_summary_template=(
        "당신은 한국어 뉴스 입장 토론을 위한 공유 근거 메모리 작성자다. 에이전트 "
        "응답을 중립적인 근거 원장으로 압축하라. 최종 라벨을 결정하거나, 직접 분류를 "
        "풀거나, 다수결을 근거로 사용하거나, 합의를 강제하지 마라.\n\n"
        "원래 과업 맥락:\n{task_context}\n\n"
        "최근 에이전트 응답:\n{responses}\n\n"
        "제시된 근거를 기사 원문과 대조하고 다음 섹션만 작성하라:\n"
        "[후보 입장] 에이전트와 후보 라벨\n"
        "[기사 자체의 근거] 제목·기자·매체 프레이밍을 보여주는 짧은 원문과 출처 에이전트\n"
        "[인용 화자의 근거] 기사 입장으로 자동 간주하면 안 되는 화자 발언\n"
        "[반대 근거] 다른 라벨을 지지하는 근거\n"
        "[오류 또는 미확인 주장] 기사에 없거나 귀속이 잘못됐거나 충돌하는 주장\n"
        "[미해결 쟁점] 에이전트가 다시 확인해야 할 사항\n"
        "구체적인 기사 근거가 있는 소수 의견을 보존하고 간결하게 작성하라."
    ),
    memory_debate_template=(
        "공유 메모리 작성자가 최근 응답을 다음 근거 원장으로 압축했다:\n\n"
        "{memory}\n\n"
        "이 원장을 정답이 아니라 확인할 주장 목록으로 취급하라. 원래 한국어 기사와 "
        "자신의 이전 분석을 기준으로 관련 주장을 다시 검증하라. 에이전트 수와 자신감은 "
        "근거가 아니다. 구체적인 기사 문구나 프레이밍 근거가 있을 때만 라벨을 변경하고, "
        "그렇지 않으면 기존 라벨을 유지하라.\n\n"
        "다음 형식으로 답하라:\n이전 입장:\n결정적인 검증 근거:\n"
        "배제하거나 귀속을 수정한 근거:\n입장 변경: <예|아니오>\n"
        + LABEL_LINE_KO
    ),
)


# ------------------------------------------------------- agent role diversity
#
# Every run so far gave all three agents the same instruction and differed them
# only by sampling. Measured on EXAONE-4.0-1.2B at round 0: accuracies 0.4735,
# 0.4725 and 0.4745, pairwise agreement 0.8651, unanimous on 800 of 1001 items,
# and all three wrong on 45.4% where independent failures at those accuracies
# would give 14.6%. The candidate pool -- how often any agent names the gold
# label -- is stuck at 0.5465, and that is the hard ceiling on every aggregation
# rule downstream. `single` and `majority k=3` scoring identically is the same
# fact seen from the other side.
#
# ChatEval (Chan et al., ICLR 2024) reports that reusing one role description
# across agents degrades multi-agent evaluation, and M-MAD (Feng et al., ACL
# 2025) attributes most of its gain to decoupling the decision across agents
# rather than to the debate. These personas decompose the judgement the way the
# stance prompt already asks for it to be decomposed: the journalist's own
# framing, the opinions merely quoted, and the wording. Each still answers the
# same three-way question and still votes, so the aggregation and the parser are
# unchanged; only what each agent attends to differs.

STANCE_PERSONAS = (
    "You are a news analyst who reads for narrative framing: which claims the "
    "article foregrounds, what it puts in the headline and lead, what it leaves "
    "to the end, and what it omits. Judge the article's stance from its own "
    "construction.",
    "You are a news analyst who reads for sourcing: who is quoted, in what "
    "order, at what length, and whether the article endorses or distances "
    "itself from what they say. An opinion a source holds is not the article's "
    "stance unless the article adopts it.",
    "You are a news analyst who reads for wording: the verbs, modifiers and "
    "labels the journalist chooses when describing each side, and whether that "
    "choice is evaluative or neutral. Judge the article's stance from its "
    "language.",
)


def stance_personas(n_agents, enabled=True):
    """One system prompt per agent, or None to keep the shared prompt.

    Returning None rather than a list of copies keeps the old runs bit-identical
    when personas are off.
    """
    if not enabled:
        return None
    if n_agents > len(STANCE_PERSONAS):
        raise ValueError(
            f"{n_agents} agents but only {len(STANCE_PERSONAS)} personas are defined"
        )
    return list(STANCE_PERSONAS[:n_agents])


PROFILES = {
    profile.name: profile
    for profile in (
        LEGACY_KO,
        STANCE_V2_EN,
        STANCE_V2_EN_GENERIC_MEMORY,
        STANCE_MINIMAL_EN,
        STANCE_TWOSTEP_EN,
        STANCE_GATE_EN,
        STANCE_V2_KO,
    )
}


def get_stance_prompt_profile(name):
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown stance prompt profile {name!r}; choose one of: {choices}") from exc

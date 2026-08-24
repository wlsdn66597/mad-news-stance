"""Closed-set role planning followed by the existing three-agent debate."""
import json
import re

from . import methods
from .llm import build_messages, chat, strip_think
from .prompts.stance import STANCE_PERSONA_ROLES


JOURNALISM_ROLE_LIBRARY = {
    "issue_alignment": {
        "use_when": "the article covers related actors or events that may distract from the target proposition",
        "prompt": (
            "You are the Issue-alignment analyst. Restate the specified issue as one "
            "proposition and test only the journalist's treatment of that proposition. "
            "Separate related actors, procedures, and events. Do not default to neutral "
            "merely because the relationship is indirect."
        ),
    },
    "foregrounding": {
        "use_when": "placement, repetition, omission, or the opening and ending appear decisive",
        "prompt": (
            "You are the Foregrounding analyst. Examine structural salience: headline, "
            "lead, repetition, conclusion, and meaningful omission. Determine whether "
            "the body reinforces or qualifies the dominant frame."
        ),
    },
    "sourcing": {
        "use_when": "the article relies on multiple quoted or attributed speakers",
        "prompt": (
            "You are the Sourcing analyst. Examine who is quoted, in what order and "
            "at what length, and whether the journalist endorses, challenges, or "
            "distances those claims. A source's stance is not automatically the article's."
        ),
    },
    "wording": {
        "use_when": "evaluative labels, reporting verbs, modifiers, or metaphors may carry the stance",
        "prompt": (
            "You are the Wording analyst. Examine journalist-authored verbs, modifiers, "
            "labels, and metaphors. Keep quoted language separate from the journalist's "
            "own lexical choices."
        ),
    },
    "evidence_balance": {
        "use_when": "substantial evidence for competing positions is present",
        "prompt": (
            "You are the Evidence-balance analyst. Compare the strongest evidence for "
            "each side, its prominence, and its treatment. The presence of both sides "
            "does not automatically make an article neutral."
        ),
    },
    "causal_consequence": {
        "use_when": "the framing turns on causes, responsibility, harms, benefits, or remedies",
        "prompt": (
            "You are the Causal-consequence analyst. Track how the article assigns the "
            "problem, cause, responsibility, benefits or harms, and preferred remedy. "
            "Judge how that causal frame bears on the specified issue."
        ),
    },
}

LEGACY_USE_WHEN = {
    "foregrounding": "headline, lead, placement, or omission appears to organize the article",
    "sourcing": "quoted speakers and the journalist's distance from them are important",
    "wording": "journalist-authored lexical choices may carry evaluation",
    "balance": "the relative space and legitimacy given to competing sides must be compared",
    "headline": "the opening frame may be reinforced or qualified by the body",
    "issue": "the article contains strong treatment of topics adjacent to the specified issue",
}

LEGACY_ROLE_LIBRARY = {
    name: {"use_when": LEGACY_USE_WHEN[name], "prompt": prompt}
    for name, prompt in STANCE_PERSONA_ROLES.items()
}

ROLE_LIBRARIES = {
    "journalism": JOURNALISM_ROLE_LIBRARY,
    "legacy": LEGACY_ROLE_LIBRARY,
}
FALLBACK_ROLES = ("foregrounding", "sourcing", "wording")
PLANNER_MODES = ("free3", "sw_plus_one")
SW_FIXED_ROLES = ("sourcing", "wording")

PLANNER_SYSTEM_PROMPT = (
    "You select complementary analytical roles for news analysis. You do not "
    "classify the article and must not mention a stance label."
)


def build_planner_prompt(item, role_library, selection_count=3):
    cards = "\n".join(
        f"- {name}: {card['use_when']}"
        for name, card in role_library.items()
    )
    count_text = "one role" if selection_count == 1 else "three distinct and complementary roles"
    output_roles = '["role_1"]' if selection_count == 1 else '["role_1", "role_2", "role_3"]'
    return (
        f"Select exactly {count_text} from the closed list "
        "below. Choose roles based on observable article structure, not on a predicted "
        "answer. Do not create new roles and do not state whether the article supports "
        "or opposes the issue. Return JSON only.\n\n"
        f"Available roles:\n{cards}\n\n"
        f"Issue: {item['issue']}\n"
        f"Headline: {item.get('headline', '')}\n"
        f"Article (Korean):\n{item.get('article', '')}\n\n"
        f'Output: {{"selected_roles": {output_roles}, '
        '"observed_features": ["feature 1", "feature 2"]}'
    )


def _json_object(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner output contains no JSON object")
    return json.loads(cleaned[start:end + 1])


def parse_planner_output(text, role_library, expected_count=3):
    payload = _json_object(text)
    roles = payload.get("selected_roles")
    if not isinstance(roles, list) or len(roles) != expected_count:
        raise ValueError(
            f"selected_roles must contain exactly {expected_count} role ids"
        )
    roles = [str(role).strip() for role in roles]
    if len(set(roles)) != expected_count:
        raise ValueError("selected_roles must be distinct")
    unknown = [role for role in roles if role not in role_library]
    if unknown:
        raise ValueError(f"unknown planner roles: {', '.join(unknown)}")
    # Role order should not become an uncontrolled fourth planner decision.
    canonical = [role for role in role_library if role in roles]
    features = payload.get("observed_features", [])
    if not isinstance(features, list):
        features = [str(features)]
    return canonical, [str(feature) for feature in features]


def select_roles(
    model,
    tokenizer,
    item,
    role_pool="journalism",
    planner_mode="free3",
    max_new_tokens=128,
    temperature=0.0,
    enable_thinking=None,
):
    role_library = ROLE_LIBRARIES[role_pool]
    if planner_mode not in PLANNER_MODES:
        raise ValueError(
            f"unknown planner mode {planner_mode!r}; choose one of {', '.join(PLANNER_MODES)}"
        )
    if planner_mode == "sw_plus_one":
        fixed_roles = list(SW_FIXED_ROLES)
        candidate_library = {
            name: card for name, card in role_library.items()
            if name not in SW_FIXED_ROLES
        }
        selection_count = 1
        fallback_planner_roles = ["foregrounding"]
    else:
        fixed_roles = []
        candidate_library = role_library
        selection_count = 3
        fallback_planner_roles = list(FALLBACK_ROLES)
    prompt = build_planner_prompt(
        item, candidate_library, selection_count=selection_count
    )
    messages = build_messages(prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
    raw = strip_think(
        chat(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    )
    error = None
    try:
        planner_roles, features = parse_planner_output(
            raw, candidate_library, expected_count=selection_count
        )
        fallback = False
    except (ValueError, json.JSONDecodeError) as exc:
        planner_roles, features = fallback_planner_roles, []
        fallback = True
        error = str(exc)
    roles = planner_roles + fixed_roles
    leakage = bool(re.search(r"\b(supportive|oppositional|neutral)\b", raw, re.I))
    return {
        "selected_roles": roles,
        "planner_selected_roles": planner_roles,
        "fixed_roles": fixed_roles,
        "observed_features": features,
        "raw": raw,
        "prompt": prompt,
        "messages": messages,
        "fallback": fallback,
        "parse_error": error,
        "stance_label_leakage": leakage,
        "role_pool": role_pool,
        "planner_mode": planner_mode,
    }


def run_planner_debate(
    model,
    tokenizer,
    task,
    item,
    max_new_tokens=1024,
    n_rounds=4,
    temperature=1.0,
    planner_temperature=0.0,
    planner_max_new_tokens=128,
    role_pool="journalism",
    planner_mode="free3",
    debate_protocol="reasoned_exchange_full",
    enable_thinking=None,
):
    """Plan three roles once, then run the unchanged 3-agent debate and vote."""
    plan = select_roles(
        model,
        tokenizer,
        item,
        role_pool=role_pool,
        planner_mode=planner_mode,
        max_new_tokens=planner_max_new_tokens,
        temperature=planner_temperature,
        enable_thinking=enable_thinking,
    )
    library = ROLE_LIBRARIES[role_pool]
    system_prompts = [library[name]["prompt"] for name in plan["selected_roles"]]
    debate_result = methods.run_debate(
        model,
        tokenizer,
        task,
        item,
        system_prompts,
        max_new_tokens,
        n_agents=3,
        n_rounds=n_rounds,
        temperature=temperature,
        enable_thinking=enable_thinking,
        debate_protocol=debate_protocol,
    )
    debate_result["planner"] = plan
    debate_result["selected_roles"] = plan["selected_roles"]
    debate_result["call_counts"] = {
        "planner_generation_calls": 1,
        "debate_agent_generation_calls": 3 * n_rounds,
        "total_generation_calls": 1 + 3 * n_rounds,
    }
    return debate_result

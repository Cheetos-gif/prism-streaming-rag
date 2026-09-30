"""
Multi-Intent Decomposer — splits compound utterances into sub-queries.

Given a user utterance like:
    "I need a venue for 30 people and the cancellation policy and catering options"

Produces three SubQuery objects, each independently searchable.

Design principle: DON'T over-fragment.
    "What's the weight and dimensions of Model 7?" → ONE query (same topic)
    "Venue capacity, cancellation policy, catering" → THREE queries (different topics)

Uses Gemini for decomposition with a rule-based fast-path for obviously
single-intent utterances.
"""

from __future__ import annotations

import re

from shared.schemas import SubQuery

# ---------------------------------------------------------------------------
# Fast-path heuristic: skip LLM for obviously single-intent utterances
# ---------------------------------------------------------------------------

_CONJUNCTION_PATTERN = re.compile(
    r"\b(?:and\s+(?:I\s+need|also|what|the)|"
    r"also\s+(?:need|want|tell)|"
    r"as\s+well\s+as|"
    r"plus\s+(?:the|I)|"
    r"what\s+about)\b",
    re.IGNORECASE,
)

_MULTI_QUESTION_MARK = re.compile(r"\?.*\?")


def _likely_multi_intent(text: str) -> bool:
    """Quick check: does the text look like it contains multiple questions?"""
    if _MULTI_QUESTION_MARK.search(text):
        return True
    # Count conjunction-like signals
    matches = _CONJUNCTION_PATTERN.findall(text)
    return len(matches) >= 1


# ---------------------------------------------------------------------------
# Decomposer prompt
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a query decomposition engine for a retrieval-augmented generation system.
Given a user utterance, extract independent sub-questions that can each be searched separately.

RULES:
1. If the utterance contains only ONE question or topic, return a single sub-query.
2. Do NOT over-fragment: multiple attributes of the SAME object are ONE query.
   Example: "weight and dimensions of Model 7" → one query about Model 7 specs.
3. Each sub-query must be independently searchable against a document corpus.
4. The sub_intent should be a short snake_case label (e.g., "venue_capacity").
5. The search_query should be optimised for keyword/semantic search (no pronouns, include key entities).
6. Return a JSON array of objects, each with: sub_intent, search_query, original_span.

EXAMPLES:
Input: "I need a venue for 30 people in Pune and the cancellation policy"
Output: [
  {"sub_intent": "venue_capacity", "search_query": "workshop venue Pune capacity 30 people", "original_span": "I need a venue for 30 people in Pune"},
  {"sub_intent": "cancellation_policy", "search_query": "cancellation policy refund terms", "original_span": "the cancellation policy"}
]

Input: "What are the weight and dimensions of the Model 7 compressor?"
Output: [
  {"sub_intent": "model7_specifications", "search_query": "Model 7 compressor weight dimensions specifications", "original_span": "weight and dimensions of the Model 7 compressor"}
]
"""


def _build_prompt(utterance: str) -> str:
    return f'Decompose this utterance into independent sub-queries:\n\n"{utterance}"'


# ---------------------------------------------------------------------------
# Main decomposer
# ---------------------------------------------------------------------------


def decompose(utterance: str, use_llm: bool = True) -> list[SubQuery]:
    """Decompose an utterance into one or more SubQuery objects.

    Parameters
    ----------
    utterance : str
        The full accumulated transcript text.
    use_llm : bool
        If True (default), uses Gemini for decomposition when multi-intent
        is detected. If False, uses only the rule-based fallback.

    Returns
    -------
    list[SubQuery]
        One or more search-ready sub-queries.
    """
    utterance = utterance.strip()
    if not utterance:
        return []

    # Fast path: if obviously single-intent, skip LLM
    if not _likely_multi_intent(utterance):
        return [_single_subquery(utterance)]

    if not use_llm:
        return _rule_based_decompose(utterance)

    # LLM path
    try:
        return _llm_decompose(utterance)
    except Exception:
        # Fallback to rule-based on any LLM failure
        return _rule_based_decompose(utterance)


def _single_subquery(utterance: str) -> SubQuery:
    """Wrap a single-intent utterance as one SubQuery."""
    intent = _infer_intent_label(utterance)
    return SubQuery(
        sub_intent=intent,
        search_query=_clean_for_search(utterance),
        original_span=utterance,
    )


def _llm_decompose(utterance: str) -> list[SubQuery]:
    """Use Gemini to decompose a compound utterance."""
    from shared.llm import generate_json

    result = generate_json(
        prompt=_build_prompt(utterance),
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.1,
    )

    if isinstance(result, dict) and "sub_queries" in result:
        result = result["sub_queries"]

    if not isinstance(result, list) or not result:
        return [_single_subquery(utterance)]

    sub_queries = []
    for item in result:
        if isinstance(item, dict) and "sub_intent" in item and "search_query" in item:
            sub_queries.append(
                SubQuery(
                    sub_intent=item["sub_intent"],
                    search_query=item["search_query"],
                    original_span=item.get("original_span", ""),
                )
            )

    return sub_queries if sub_queries else [_single_subquery(utterance)]


def _rule_based_decompose(utterance: str) -> list[SubQuery]:
    """Fallback decomposition using conjunction splitting.

    Splits on 'and I need', 'and the', 'and also', etc.
    Not perfect, but good enough when LLM is unavailable.
    """
    # Split on major conjunction patterns
    parts = re.split(
        r",?\s+and\s+(?:I\s+need\s+)?(?:the\s+)?|"
        r",?\s+also\s+|"
        r",?\s+as\s+well\s+as\s+|"
        r",?\s+plus\s+",
        utterance,
        flags=re.IGNORECASE,
    )

    # Filter empty parts and deduplicate
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]

    if len(parts) <= 1:
        return [_single_subquery(utterance)]

    return [
        SubQuery(
            sub_intent=_infer_intent_label(part),
            search_query=_clean_for_search(part),
            original_span=part,
        )
        for part in parts
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_intent_label(text: str) -> str:
    """Generate a snake_case intent label from text."""
    text_lower = text.lower()

    # Known intent patterns
    intent_map = [
        (r"cancel|refund", "cancellation_policy"),
        (r"cater|food|menu|meal", "catering_options"),
        (r"venue|capacity|room|seat|people|attendee", "venue_capacity"),
        (r"troubleshoot|knock|vibrat|noise|fault|problem|issue", "troubleshooting"),
        (r"maintenance|schedule|inspect", "maintenance_schedule"),
        (r"safety|protocol|ppe|lockout", "safety_protocols"),
        (r"spare|part|order|replacement", "spare_parts"),
        (r"warranty|service\s+agreement", "warranty_terms"),
        (r"emergency|shutdown", "emergency_procedures"),
        (r"travel|reimburse|per\s+diem|expense", "travel_reimbursement"),
        (r"international|foreign|abroad", "international_travel"),
        (r"book|reservation", "booking"),
        (r"spec|dimension|weight|rated|power|capacity", "specifications"),
        (r"run|operate|continue|friday|until|till", "continued_operation"),
        (r"compressor|model\s+\d", "equipment_info"),
    ]

    for pattern, label in intent_map:
        if re.search(pattern, text_lower):
            return label

    # Generic fallback: first few words, snake_cased
    words = re.findall(r"[a-z]+", text_lower)[:3]
    return "_".join(words) if words else "general_query"


def _clean_for_search(text: str) -> str:
    """Clean utterance text for use as a search query.

    Removes filler words, pronouns, and conversational noise.
    """
    # Remove common filler
    cleaned = re.sub(
        r"\b(um|uh|I\s+need|I\s+want|can\s+you|please|could\s+you|"
        r"tell\s+me|I'm\s+looking\s+at|let\s+me\s+know|also)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Remove leading/trailing punctuation
    cleaned = cleaned.strip(".,;:!?")
    return cleaned if cleaned else text

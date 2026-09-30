"""
Grounding Verifier — checks that claims are actually supported by their cited chunks.

Catches citation hallucination: when a claim states a fact that isn't present
in the chunks it cites. This is Gate G4's core check.

Approach:
  1. Extract key factual assertions from the claim text
  2. For each assertion, check if any cited chunk contains supporting text
  3. Flag claims where an assertion has no chunk support
"""

from __future__ import annotations

import re

from shared.schemas import Chunk, ChunkRecord, Claim, GroundingResult


def verify_claim(
    claim: Claim,
    chunk_map: dict[str, ChunkRecord | Chunk],
) -> GroundingResult:
    """Verify that a claim's text is supported by its cited chunks.

    Parameters
    ----------
    claim : Claim
        The claim to verify.
    chunk_map : dict
        Mapping from chunk_id to chunk object (must have .text attribute).

    Returns
    -------
    GroundingResult
    """
    if claim.status == "superseded":
        return GroundingResult(
            claim_id=claim.id,
            is_grounded=False,
            supporting_chunks=[],
            unsupported_assertions=["claim is superseded"],
        )

    if claim.status == "unverified":
        return GroundingResult(
            claim_id=claim.id,
            is_grounded=False,
            supporting_chunks=[],
            unsupported_assertions=["claim is self-reported as unverified"],
        )

    # Gather the text of all cited chunks
    cited_texts: list[str] = []
    supporting_chunks: list[str] = []
    for cid in claim.chunk_ids:
        chunk = chunk_map.get(cid)
        if chunk is not None:
            cited_texts.append(chunk.text.lower())
            supporting_chunks.append(cid)

    if not cited_texts:
        return GroundingResult(
            claim_id=claim.id,
            is_grounded=False,
            supporting_chunks=[],
            unsupported_assertions=["no cited chunks found in index"],
        )

    # Extract key assertions from claim text
    assertions = _extract_assertions(claim.text)
    unsupported: list[str] = []

    for assertion in assertions:
        if not _assertion_supported(assertion, cited_texts):
            unsupported.append(assertion)

    is_grounded = len(unsupported) == 0

    return GroundingResult(
        claim_id=claim.id,
        is_grounded=is_grounded,
        supporting_chunks=supporting_chunks,
        unsupported_assertions=unsupported,
    )


def verify_all_claims(
    claims: list[Claim],
    chunk_map: dict[str, ChunkRecord | Chunk],
) -> list[GroundingResult]:
    """Verify all claims in a list."""
    return [verify_claim(c, chunk_map) for c in claims]


def grounding_score(results: list[GroundingResult]) -> float:
    """Compute the fraction of claims that are grounded (0.0-1.0)."""
    if not results:
        return 1.0
    grounded = sum(1 for r in results if r.is_grounded)
    return grounded / len(results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_assertions(claim_text: str) -> list[str]:
    """Extract key factual phrases from a claim's text.

    Splits on sentence boundaries and extracts noun phrases with numbers
    or proper nouns — the things most likely to be hallucinated.
    """
    # Remove citation markers like [Doc_02 §3]
    clean = re.sub(r"\[Doc_\w+\s*§\w+\]", "", claim_text)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", clean.strip())

    assertions: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        # Extract number-bearing phrases (most falsifiable)
        number_phrases = re.findall(
            r"(?:\w+\s+){0,3}\d+[\d,.]*\s*(?:\w+\s*){0,3}",
            sentence,
        )
        assertions.extend(p.strip() for p in number_phrases if len(p.strip()) > 3)

        # If no number phrases, use the whole sentence as an assertion
        if not number_phrases:
            assertions.append(sentence)

    return assertions if assertions else [clean.strip()]


def _assertion_supported(assertion: str, chunk_texts: list[str]) -> bool:
    """Check if an assertion is supported by any of the chunk texts.

    Uses keyword overlap as a proxy for entailment. A more sophisticated
    version could use NLI, but for the competition this is sufficient
    and doesn't require extra model loading.
    """
    assertion_lower = assertion.lower().strip()

    # Extract significant keywords (3+ chars, not stop words)
    stop_words = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "been",
        "will",
        "can",
        "may",
        "with",
        "from",
        "that",
        "this",
        "not",
        "but",
        "its",
        "also",
        "into",
        "than",
        "then",
        "when",
        "which",
        "each",
        "such",
        "must",
        "does",
        "more",
        "most",
    }

    keywords = {
        w
        for w in re.findall(r"\b[a-z0-9]+\b", assertion_lower)
        if len(w) >= 3 and w not in stop_words
    }

    if not keywords:
        return True  # nothing to verify

    # Check each chunk for keyword coverage
    for chunk_text in chunk_texts:
        chunk_words = set(re.findall(r"\b[a-z0-9]+\b", chunk_text))
        overlap = keywords & chunk_words
        coverage = len(overlap) / len(keywords) if keywords else 1.0
        if coverage >= 0.5:  # at least 50% keyword overlap
            return True

    return False

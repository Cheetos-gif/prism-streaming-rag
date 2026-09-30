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

import os
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


# ---------------------------------------------------------------------------
# Corpus coverage — "can this corpus answer that question at all?"
# ---------------------------------------------------------------------------

# Two out-of-corpus content words were the point at which the test set in
# tests/test_evidence_coverage.py separates — and the measurement is also why this is a
# **reported signal and not a gate**. It was briefly wired to downgrade claims to
# `unverified` and to warn the model in the synthesis prompt; that was reverted after two
# live failures, both reproduced against the real model:
#
#   1. "how long until a technician arrives when the machine is critical" and "what does the
#      warranty say about using a different oil" are answerable, yet they carry two and
#      three out-of-corpus words, so they were flagged and downgraded.
#   2. With a sub-intent label the decomposer can produce (a generic one), the added prompt
#      sentence made the model refuse answerable questions outright:
#        old prompt + generic sub-intent -> grounded=true  ("cancellation refund tiers")
#        new prompt + generic sub-intent -> grounded=false ("Insufficient evidence …")
#        either prompt + specific sub-intent -> grounded=true
#
# Sparse vocabulary is not evidence of an unanswerable question, and neither is the dense
# cosine: a legitimate "zone A international per diem pre-approval" scores 0.459 while
# "corporate policy for travelling to the moon" scores 0.454, so no floor separates them.
# The terms are therefore logged (`evidence_coverage`) and nothing more. Surfacing
# out-of-corpus questions as uncertainty is still an open problem; it needs a signal that
# survives ordinary phrasing, which this corpus/vocabulary pair does not provide.
DEFAULT_UNKNOWN_TERM_LIMIT = int(os.getenv("EVIDENCE_UNKNOWN_TERM_LIMIT", "2"))

_QUERY_FUNCTION_WORDS = {
    # interrogatives and pronouns only: hedges and adjectives ("best", "much") are left
    # in, because in "best pizza in Pune" the hedge is part of what makes the question
    # unanswerable from a compressor/travel corpus.
    "what",
    "who",
    "whom",
    "whose",
    "where",
    "why",
    "how",
    "when",
    "does",
    "did",
    "are",
    "was",
    "were",
    "the",
    "and",
    "for",
    "with",
}


def _content_terms(text: str) -> list[str]:
    """Lowercase words of 3+ characters that carry subject matter.

    Split on non-word characters, exactly as `retrieval.indexer.tokenize` does, so a
    hyphenated word like "on-site" is compared as "on"+"site" — the corpus vocabulary is
    built with that tokenizer, and comparing "on-site" against it reported a word as
    missing when only the tokenization differed.
    """
    return [
        word
        for word in re.split(r"\W+", text.lower())
        if len(word) >= 3 and word not in _QUERY_FUNCTION_WORDS
    ]


def out_of_corpus_terms(query: str, vocabulary: set[str]) -> list[str]:
    """Return the query's content words that occur nowhere in the corpus.

    `vocabulary` is `CorpusIndex.vocabulary`. Order is preserved and duplicates are
    dropped so the caller can log the terms as they were asked.
    """
    seen: list[str] = []
    for term in _content_terms(query):
        if term not in vocabulary and term not in seen:
            seen.append(term)
    return seen


def evidence_is_weak(terms: list[str], limit: int = DEFAULT_UNKNOWN_TERM_LIMIT) -> bool:
    """Whether a list of out-of-corpus terms is enough to downgrade claims to unverified."""
    return len(set(terms)) >= limit

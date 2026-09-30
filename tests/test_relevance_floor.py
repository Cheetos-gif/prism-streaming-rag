"""Tests for Issue #3: Relevance floor for out-of-corpus questions.

Acceptance criteria:
- Negative test: "What is the warranty coverage for a lunar habitat airlock module?"
  must return the insufficient-evidence claim or claims marked `unverified` with the
  affected sub-intent in `answer.uncertainty`.
- Four regression questions that must stay grounded:
  1. "how long until a technician arrives when the machine is critical"
  2. "what does the warranty say about using a different oil"
  3. "What is the on-site response time for critical severity issues?"
  4. "cancellation refund tiers"
"""

from __future__ import annotations

import pytest


def test_lunar_habitat_airlock_is_unverified(client):
    """Negative test: lunar habitat airlock is not in corpus, so claims must be unverified."""
    session_id = client.post("/session").json()["session_id"]
    client.post(
        f"/session/{session_id}/stream",
        json={
            "timestamp_s": 0.0,
            "text": "What is the warranty coverage for a lunar habitat airlock module?",
            "is_final": False,
        },
    )
    client.post(f"/session/{session_id}/utterance_end")
    state = client.get(f"/session/{session_id}").json()

    answer = state["answer"]
    assert answer is not None, "An answer snapshot should be produced"
    claims = answer["claims"]
    assert claims, "Claims should be produced"

    # Must return insufficient-evidence claim or unverified claims in answer.uncertainty
    has_unverified_or_insufficient = any(
        c["status"] == "unverified" or "insufficient evidence" in c["text"].lower() for c in claims
    )
    assert has_unverified_or_insufficient, (
        f"Expected unverified claim or insufficient evidence, got: {claims}"
    )
    assert answer["uncertainty"], (
        f"Expected non-empty answer.uncertainty, got {answer['uncertainty']}"
    )


@pytest.mark.parametrize(
    "question",
    [
        "how long until a technician arrives when the machine is critical",
        "what does the warranty say about using a different oil",
        "What is the on-site response time for critical severity issues?",
        "cancellation refund tiers",
    ],
)
def test_regression_questions_stay_grounded(client, question):
    """Regression test: answerable questions must remain grounded without false refusals."""
    session_id = client.post("/session").json()["session_id"]
    client.post(
        f"/session/{session_id}/stream",
        json={
            "timestamp_s": 0.0,
            "text": question,
            "is_final": False,
        },
    )
    client.post(f"/session/{session_id}/utterance_end")
    state = client.get(f"/session/{session_id}").json()

    answer = state["answer"]
    assert answer is not None, f"Answer should be produced for: {question}"
    claims = answer["claims"]
    assert claims, f"Claims should be present for: {question}"
    for c in claims:
        assert c["status"] == "grounded", f"Claim should be grounded for {question!r}: {c}"
        assert "insufficient evidence" not in c["text"].lower(), (
            f"Question {question!r} should not have insufficient evidence claim: {c['text']}"
        )
    assert not answer["uncertainty"], (
        f"Expected uncertainty to be empty for {question!r}, got {answer['uncertainty']}"
    )


def test_check_corpus_relevance_direct():
    """Unit test for check_corpus_relevance in local/conservative mode."""
    from ledger.synthesizer import check_corpus_relevance

    negatives = [
        "What is the warranty coverage for a lunar habitat airlock module?",
        "corporate policy for travelling to the moon",
        "best pizza in Pune",
        "quantum computing error correction techniques",
        "recipe for sourdough bread",
    ]
    for q in negatives:
        rel, reason = check_corpus_relevance(q, use_llm=False)
        assert not rel, f"Expected {q} to be judged not relevant, got reason: {reason}"

    positives = [
        "how long until a technician arrives when the machine is critical",
        "what does the warranty say about using a different oil",
        "What is the on-site response time for critical severity issues?",
        "cancellation refund tiers",
        "Venue A capacity and Hinjewadi booking",
        "per diem meal allowance for Mumbai trip",
    ]
    for q in positives:
        rel, reason = check_corpus_relevance(q, use_llm=False)
        assert rel, f"Expected {q} to be judged relevant, got reason: {reason}"


def test_get_default_section_titles():
    from ledger.synthesizer import get_default_section_titles

    titles = get_default_section_titles()
    assert len(titles) > 10
    assert any("Warranty" in t for t in titles)
    assert any("Compressor" in t for t in titles)


def test_check_corpus_relevance_llm_path():
    """Verify that when use_llm=True and provider is LLM, LLM output is parsed correctly."""
    from unittest.mock import patch

    from ledger.synthesizer import check_corpus_relevance

    with patch("shared.llm.get_provider", return_value="groq"):
        with patch(
            "shared.llm.generate_json",
            return_value={"relevant": False, "reason": "lunar habitat not in corpus"},
        ):
            rel, reason = check_corpus_relevance("lunar habitat airlock", use_llm=True)
            assert rel is False
            assert "lunar" in reason

        with patch(
            "shared.llm.generate_json",
            return_value={"relevant": True, "reason": "covered by warranty section"},
        ):
            rel, reason = check_corpus_relevance("warranty terms", use_llm=True)
            assert rel is True
            assert "covered" in reason

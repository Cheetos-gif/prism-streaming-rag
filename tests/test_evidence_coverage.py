"""Tests for the corpus-coverage signal.

This is a *reported* signal, not a gate. It was briefly wired to mark claims
`unverified` and to warn the model in the synthesis prompt; that was reverted after it
misread ordinary phrasing and the model, warned, refused to answer an answerable
question (see the comment above `DEFAULT_UNKNOWN_TERM_LIMIT` in ledger/grounding.py).

So these tests pin two things: the signal catches subject matter the corpus cannot
discuss, and it also fires on legitimate English — which is exactly why it is not
allowed to change an answer.
"""

import pytest

from ledger.grounding import (
    DEFAULT_UNKNOWN_TERM_LIMIT,
    evidence_is_weak,
    out_of_corpus_terms,
)
from retrieval.indexer import CorpusIndex

UNANSWERABLE = [
    "corporate policy for travelling to the moon",
    "what is the warranty coverage for a lunar habitat airlock module",
    "best pizza in Pune",
    "quantum computing error correction techniques",
    "recipe for sourdough bread",
]

ANSWERABLE_WITH_CORPUS_VOCABULARY = [
    "cancellation refund tiers",
    "catering minimum order size and advance notice",
    "warranty response time for critical severity",
    "run till failure policy for minor faults",
]

# Ordinary phrasing for answerable questions that the signal still flags. Kept as a
# test so nobody wires this signal back into the answer path without seeing them.
ANSWERABLE_BUT_FLAGGED = [
    "how long until a technician arrives when the machine is critical",
    "what does the warranty say about using a different oil",
]


@pytest.fixture(scope="module")
def vocabulary() -> set[str]:
    return CorpusIndex.build("data/corpus").vocabulary


def test_absent_subject_matter_is_named(vocabulary):
    terms = out_of_corpus_terms("corporate policy for travelling to the moon", vocabulary)

    assert "moon" in terms
    assert "travelling" in terms
    assert "policy" not in terms, "words the corpus does use stay out of the list"


def test_unanswerable_questions_exceed_the_limit(vocabulary):
    for question in UNANSWERABLE:
        assert evidence_is_weak(out_of_corpus_terms(question, vocabulary)), question


def test_hyphenated_words_are_split_like_the_corpus_vocabulary(vocabulary):
    # The corpus vocabulary comes from retrieval.indexer.tokenize, which splits on
    # non-word characters; comparing "on-site" against it reported a missing word when
    # only the tokenization differed.
    assert "site" in vocabulary and "on-site" not in vocabulary
    assert "on-site" not in out_of_corpus_terms("on-site response time", vocabulary)
    assert out_of_corpus_terms("on-site response time", vocabulary) == []


def test_typical_corpus_phrasing_does_not_trip_the_limit(vocabulary):
    for question in ANSWERABLE_WITH_CORPUS_VOCABULARY:
        assert not evidence_is_weak(out_of_corpus_terms(question, vocabulary)), question


def test_ordinary_phrasing_is_flagged_too_which_is_why_it_is_not_a_gate(vocabulary):
    for question in ANSWERABLE_BUT_FLAGGED:
        assert evidence_is_weak(out_of_corpus_terms(question, vocabulary)), (
            f"{question!r} is answerable, yet the vocabulary signal calls it weak"
        )


def test_the_limit_is_configurable():
    assert DEFAULT_UNKNOWN_TERM_LIMIT == 2
    assert evidence_is_weak(["a", "b"], limit=2)
    assert not evidence_is_weak(["a"], limit=2)


def test_query_function_words_are_not_counted_as_missing(vocabulary):
    terms = out_of_corpus_terms("who knows what happened where", vocabulary)

    assert not (set(terms) & {"who", "what", "where"})


def test_answers_are_not_downgraded_by_the_signal(client):
    """The regression this file exists for: a flagged question still gets its answer."""
    session_id = client.post("/session").json()["session_id"]
    client.post(
        f"/session/{session_id}/stream",
        json={
            "timestamp_s": 0.0,
            "text": "how long until a technician arrives when the machine is critical",
            "is_final": False,
        },
    )
    client.post(f"/session/{session_id}/utterance_end")
    state = client.get(f"/session/{session_id}").json()

    assert state["answer"]["claims"], "the answer is still produced"
    assert {c["status"] for c in state["answer"]["claims"]} == {"grounded"}
    assert any(
        event.get("event_type") == "evidence_coverage"
        for event in client.get(f"/session/{session_id}/telemetry").json()["events"]
    ), "the signal is still recorded, it just does not change the answer"

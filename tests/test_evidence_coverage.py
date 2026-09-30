"""Tests for the corpus-coverage signal and the uncertainty it produces.

Vocabulary membership, not vector similarity, is what separates "the corpus answers
this" from "the corpus does not discuss this" — see the measurement in
ledger/grounding.py. These tests pin both the signal and the behaviour it drives.
"""

import pytest

from ledger.grounding import (
    DEFAULT_UNKNOWN_TERM_LIMIT,
    evidence_is_weak,
    out_of_corpus_terms,
)
from retrieval.indexer import CorpusIndex

ANSWERABLE = [
    "CMP7-BRG-001 bearing cost and lead time",
    "warranty response time for critical severity",
    "what are the emergency shutdown triggers",
    "Pune workshop venue capacity for 50 people",
    "catering minimum order size and advance notice",
    "cancellation refund tiers",
    "tier 1 per diem and hotel cap",
    "zone A international per diem pre-approval",
]
UNANSWERABLE = [
    "corporate policy for travelling to the moon",
    "what is the warranty coverage for a lunar habitat airlock module",
    "best pizza in Pune",
    "quantum computing error correction techniques",
    "recipe for sourdough bread",
]


@pytest.fixture(scope="module")
def vocabulary() -> set[str]:
    return CorpusIndex.build("data/corpus").vocabulary


def test_absent_subject_matter_is_named(vocabulary):
    terms = out_of_corpus_terms("corporate policy for travelling to the moon", vocabulary)

    assert "moon" in terms
    assert "travelling" in terms
    # words the corpus does use stay out of the list
    assert "policy" not in terms


def test_answerable_questions_stay_under_the_limit(vocabulary):
    for question in ANSWERABLE:
        terms = out_of_corpus_terms(question, vocabulary)
        assert len(terms) < DEFAULT_UNKNOWN_TERM_LIMIT, (question, terms)
        assert not evidence_is_weak(terms), (question, terms)


def test_unanswerable_questions_exceed_the_limit(vocabulary):
    for question in UNANSWERABLE:
        terms = out_of_corpus_terms(question, vocabulary)
        assert evidence_is_weak(terms), (question, terms)


def test_query_function_words_are_not_counted_as_missing(vocabulary):
    terms = out_of_corpus_terms("who knows what happened where", vocabulary)

    assert not (set(terms) & {"who", "what", "where"})


def _claims_after_asking(client, question: str) -> dict:
    session_id = client.post("/session").json()["session_id"]
    client.post(
        f"/session/{session_id}/stream",
        json={"timestamp_s": 0.0, "text": question, "is_final": False},
    )
    client.post(f"/session/{session_id}/utterance_end")
    return client.get(f"/session/{session_id}").json()


def test_unanswerable_question_is_kept_as_uncertainty_not_as_a_grounded_fact(client):
    state = _claims_after_asking(
        client, "What is the warranty coverage for a lunar habitat airlock module?"
    )
    answer = state["answer"]

    assert answer["claims"], "the pipeline still answers — it just flags the answer"
    assert {c["status"] for c in answer["claims"]} == {"unverified"}
    assert answer["uncertainty"], "the weak evidence has to show up in the snapshot"
    assert all(c["chunk_ids"] for c in answer["claims"]), "citations are kept, not dropped"


def test_answerable_question_stays_grounded(client):
    state = _claims_after_asking(client, "What is the on-site response time for critical severity?")
    answer = state["answer"]

    assert answer["claims"]
    assert {c["status"] for c in answer["claims"]} == {"grounded"}
    assert answer["uncertainty"] == []

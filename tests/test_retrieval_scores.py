"""Tests for retrieval score semantics.

The score a caller reads has to mean something. It used to be the reciprocal rank
fusion value, whose ceiling is 2/(k+1) and which therefore sits at ~0.033 for *every*
query, relevant or not.
"""

import pytest

from retrieval.engine import HybridRetriever
from retrieval.indexer import CorpusIndex

RRF_CEILING = 2.0 / (60 + 1)  # two ranked lists contribute at most 1/(k+1) each


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    return HybridRetriever(CorpusIndex.build("data/corpus"))


def test_score_is_a_cosine_similarity_and_rrf_is_reported_separately(retriever):
    results = retriever.search("CMP7-BRG-001 bearing cost and lead time", top_k=5)

    assert results, "the corpus contains this part number, so it must be found"
    assert results[0].doc_id == "doc_05_spare_parts_catalog"

    for chunk in results:
        assert 0.0 <= chunk.score <= 1.0
        assert 0.0 < chunk.rrf_score <= RRF_CEILING + 1e-9

    # The two are different quantities: `score` is the cosine, which for these hits runs
    # 0.2-0.5, while every RRF value sits in a band of ~0.03 regardless of relevance.
    assert results[0].score > 0.2
    assert max(c.score for c in results) > results[0].rrf_score * 5
    assert results[0].bm25_score > 0.0


def test_rank_fusion_still_orders_the_results(retriever):
    results = retriever.search("warranty response time for critical severity", top_k=5)

    rrf_values = [c.rrf_score for c in results]
    assert rrf_values == sorted(rrf_values, reverse=True)


def test_unanswerable_questions_score_lower_than_answerable_ones(retriever):
    def best_cosine(question: str) -> float:
        return max(c.score for c in retriever.search(question, top_k=5))

    answerable = best_cosine("catering minimum order size and advance notice")
    unanswerable = best_cosine("recipe for sourdough bread")

    # Measured on this corpus: answerable questions reach 0.46-0.77, questions the corpus
    # cannot answer 0.09-0.46. The two sets overlap, which is why this is a comparison and
    # not a threshold the retriever enforces.
    assert answerable > 0.5
    assert unanswerable < 0.3
    assert answerable > unanswerable


def test_api_search_exposes_all_three_scores(client):
    res = client.post("/api/search", json={"query": "tier 1 per diem and hotel cap", "top_k": 2})

    assert res.status_code == 200
    hit = res.json()["results"][0]
    assert hit["score"] > 0.4, hit
    assert 0 < hit["rrf_score"] <= RRF_CEILING + 1e-6, hit
    assert "bm25_score" in hit

"""
Hybrid retriever for combining BM25 and dense embedding search.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from retrieval.indexer import CorpusIndex, tokenize
from shared.schemas import Chunk, SubQuery


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between matrix a and vector b."""
    if len(a) == 0:
        return np.array([])
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1e-10, norm_a)
    norm_b = 1e-10 if norm_b == 0 else norm_b
    return np.dot(a, b) / (norm_a * norm_b)


class HybridRetriever:
    """
    Retrieval engine that combines BM25 and dense embeddings with Reciprocal Rank Fusion.
    """

    def __init__(self, index: CorpusIndex):
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Search for the query in the corpus index, returning top-k chunks.
        """
        if not self.index.chunks:
            return []

        # BM25 Search
        tokenized_query = tokenize(query)
        bm25_scores = (
            self.index.bm25.get_scores(tokenized_query)
            if self.index.bm25
            else np.zeros(len(self.index.chunks))
        )

        # Dense Search
        query_embedding = self.index.model.encode(query, convert_to_numpy=True)
        dense_scores = cosine_similarity(self.index.embeddings, query_embedding)

        # Rank both
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        dense_ranks = np.argsort(dense_scores)[::-1]

        rrf_k = 60
        rrf_scores: dict[int, float] = defaultdict(float)

        for rank, chunk_idx in enumerate(bm25_ranks):
            rrf_scores[chunk_idx] += 1.0 / (rrf_k + rank + 1)

        for rank, chunk_idx in enumerate(dense_ranks):
            rrf_scores[chunk_idx] += 1.0 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results: list[Chunk] = []
        seen: set[str] = set()
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            chunk_record = self.index.chunks[idx]
            if chunk_record.chunk_id in seen:
                continue

            seen.add(chunk_record.chunk_id)
            results.append(
                Chunk(
                    doc_id=chunk_record.doc_id,
                    section=chunk_record.section,
                    text=chunk_record.text,
                    # Ordering is RRF, but the score a caller reads is the cosine: the RRF
                    # sum is rank-based (max 2/(60+1) ≈ 0.033 for every query, relevant or
                    # not), so it cannot express how well a chunk matches. Measured on this
                    # corpus: real questions score 0.46-0.77, questions the corpus cannot
                    # answer 0.09-0.46.
                    score=float(dense_scores[idx]),
                    chunk_id=chunk_record.chunk_id,
                    sub_intent=None,
                    rrf_score=float(rrf_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                )
            )

        return results

    def search_multi(self, queries: list[SubQuery], top_k: int = 5) -> dict[str, list[Chunk]]:
        """
        Perform search for multiple subqueries.
        Returns a mapping from sub_intent to a list of retrieved Chunks.
        """
        results: dict[str, list[Chunk]] = {}
        for q in queries:
            chunks = self.search(q.search_query, top_k=top_k)
            for chunk in chunks:
                chunk.sub_intent = q.sub_intent
            results[q.sub_intent] = chunks
        return results

"""
Pipeline Orchestrator — wires every stage together.

This is the single function that processes each transcript chunk through
the full pipeline:

    chunk → controller → decomposer → retriever → synthesizer → ledger

Each stage logs telemetry. The pipeline never calls external APIs
except through shared.llm, and never accesses data outside the corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from controller.decomposer import decompose
from controller.stream_simulator import TranscriptChunk
from ledger.grounding import evidence_is_weak, out_of_corpus_terms
from ledger.synthesizer import synthesize_claims
from shared.schemas import (
    AnswerSnapshot,
    Chunk,
    ControllerDecision,
    SubQuery,
)

if TYPE_CHECKING:
    from controller.session import Session
    from retrieval.engine import HybridRetriever


@dataclass
class PipelineResult:
    """Result of processing a single transcript chunk."""

    decision: ControllerDecision
    sub_queries: list[SubQuery] = field(default_factory=list)
    retrieved_chunks: dict[str, list[Chunk]] = field(default_factory=dict)
    answer: AnswerSnapshot | None = None
    telemetry: dict = field(default_factory=dict)


class Pipeline:
    """End-to-end streaming RAG pipeline.

    Usage:
        index = CorpusIndex.build("data/corpus")
        retriever = HybridRetriever(index)
        pipeline = Pipeline(retriever)

        session = Session()
        for chunk in simulator.chunks():
            result = pipeline.process_chunk(session, chunk)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        use_llm: bool = True,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.use_llm = use_llm
        self.top_k = top_k

    def process_chunk(
        self,
        session: Session,
        chunk: TranscriptChunk,
    ) -> PipelineResult:
        """Process a single incoming transcript chunk through the full pipeline.

        Parameters
        ----------
        session : Session
            The active conversation session.
        chunk : TranscriptChunk
            The incoming transcript fragment.

        Returns
        -------
        PipelineResult
            Contains the decision, any sub-queries, retrieved chunks, and
            the current answer snapshot.
        """
        t_start = time.time()

        # Record the chunk
        session.append_transcript(chunk.timestamp_s, chunk.text)
        if chunk.is_final or chunk.text == "[Utterance End]":
            session.logger.log("utterance_end", timestamp_s=chunk.timestamp_s)

        # Step 1: Controller decides what to do
        decision = session.controller.on_chunk(chunk)

        session.logger.log(
            "controller_decision",
            timestamp_s=chunk.timestamp_s,
            action=decision.action,
            reason=decision.reason,
            confidence=decision.confidence,
        )

        result = PipelineResult(decision=decision)

        if decision.action == "wait":
            return result

        if decision.action == "suppress":
            session.logger.suppressed(reason=decision.reason)
            result.answer = session.ledger.current_answer()
            return result

        if decision.action == "reretrieve":
            return self._handle_reretrieve(session, chunk, decision, result, t_start)

        if decision.action == "retrieve":
            return self._handle_retrieve(session, chunk, decision, result, t_start)

        return result

    def process_utterance_end(self, session: Session) -> PipelineResult:
        """Force processing at utterance end.

        Creates a synthetic final chunk to trigger retrieval.
        """
        final_chunk = TranscriptChunk(
            timestamp_s=time.time(),
            text="[Utterance End]",
            is_final=True,
        )
        session.logger.log("utterance_end", timestamp_s=final_chunk.timestamp_s)
        return self.process_chunk(session, final_chunk)

    # ------------------------------------------------------------------
    # Internal: handle retrieve action
    # ------------------------------------------------------------------

    def _handle_retrieve(
        self,
        session: Session,
        chunk: TranscriptChunk,
        decision: ControllerDecision,
        result: PipelineResult,
        t_start: float,
    ) -> PipelineResult:
        """Full retrieval path: decompose → search → synthesize → store."""

        accumulated = decision.accumulated_text

        # Step 2: Decompose into sub-queries
        t_decompose = time.time()
        sub_queries = decompose(accumulated, use_llm=self.use_llm)
        result.sub_queries = sub_queries

        session.logger.log(
            "decomposition_completed",
            timestamp_s=chunk.timestamp_s,
            sub_intents=[sq.sub_intent for sq in sub_queries],
            latency_ms=(time.time() - t_decompose) * 1000,
        )

        # Step 3: Retrieve for each sub-query
        all_chunks: dict[str, list[Chunk]] = {}
        for sq in sub_queries:
            t_retrieve = time.time()
            trigger = (
                "provisional" if decision.reason == "provisional_entity_match" else "sub_intent"
            )
            session.logger.retrieval_started(
                query=sq.search_query,
                trigger=trigger,
                timestamp_s=chunk.timestamp_s,
            )

            chunks = self.retriever.search(sq.search_query, top_k=self.top_k)
            # Tag chunks with the sub-intent
            for c in chunks:
                c.sub_intent = sq.sub_intent

            latency = (time.time() - t_retrieve) * 1000
            session.logger.retrieval_completed(
                query=sq.search_query,
                chunk_ids=[c.chunk_id for c in chunks],
                latency_ms=latency,
            )

            all_chunks[sq.sub_intent] = chunks

        result.retrieved_chunks = all_chunks

        # Step 4: Synthesize claims from chunks
        vocabulary = self.retriever.index.vocabulary
        all_claims = []
        for sq in sub_queries:
            chunks = all_chunks.get(sq.sub_intent, [])
            missing = out_of_corpus_terms(sq.search_query, vocabulary)
            if missing:
                session.logger.log(
                    "evidence_coverage",
                    timestamp_s=chunk.timestamp_s,
                    query=sq.search_query,
                    out_of_corpus_terms=missing,
                    weak=evidence_is_weak(missing),
                )
            claims = synthesize_claims(
                sub_intent=sq.sub_intent,
                chunks=chunks,
                version=session.ledger.version + 1,
                use_llm=self.use_llm,
            )
            all_claims.extend(claims)

        # Step 5: Store in ledger
        if all_claims:
            snapshot = session.ledger.add_claims(all_claims)
            result.answer = snapshot
            session.controller.mark_answered()
            session.controller.reset_buffer()

            session.logger.log(
                "answer_rendered",
                timestamp_s=chunk.timestamp_s,
                version=snapshot.version,
                claim_ids=[c.id for c in snapshot.claims],
            )

        result.telemetry = {
            "total_latency_ms": (time.time() - t_start) * 1000,
            "sub_queries": len(sub_queries),
            "chunks_retrieved": sum(len(v) for v in all_chunks.values()),
            "claims_generated": len(all_claims),
        }

        return result

    # ------------------------------------------------------------------
    # Internal: handle reretrieve (refinement) action
    # ------------------------------------------------------------------

    def _handle_reretrieve(
        self,
        session: Session,
        chunk: TranscriptChunk,
        decision: ControllerDecision,
        result: PipelineResult,
        t_start: float,
    ) -> PipelineResult:
        """Refinement path: identify affected claims → re-search → update."""

        constraint_text = decision.accumulated_text

        # Determine which sub-intents are affected
        # Use decomposer to identify what the constraint changes
        sub_queries = decompose(constraint_text, use_llm=self.use_llm)
        affected_intents = [sq.sub_intent for sq in sub_queries]

        # If decomposer doesn't match existing intents, try to match
        existing_intents = {c.sub_intent for c in session.ledger.get_active_claims()}
        matched_intents = [i for i in affected_intents if i in existing_intents]

        # If no direct match, search across all existing intents
        if not matched_intents:
            matched_intents = list(existing_intents)

        # Step 1: Mark old claims superseded, get refinement plan
        plan = session.ledger.refine(constraint_text, matched_intents)

        # Step 2: Re-retrieve for affected intents
        all_chunks: dict[str, list[Chunk]] = {}
        for sq in plan.queries_to_rerun:
            t_retrieve = time.time()
            session.logger.retrieval_started(
                query=sq.search_query,
                trigger="constraint_update",
                timestamp_s=chunk.timestamp_s,
            )

            chunks = self.retriever.search(sq.search_query, top_k=self.top_k)
            for c in chunks:
                c.sub_intent = sq.sub_intent

            latency = (time.time() - t_retrieve) * 1000
            session.logger.retrieval_completed(
                query=sq.search_query,
                chunk_ids=[c.chunk_id for c in chunks],
                latency_ms=latency,
            )
            all_chunks[sq.sub_intent] = chunks

        result.retrieved_chunks = all_chunks
        result.sub_queries = plan.queries_to_rerun

        # Step 3: Synthesize replacement claims
        vocabulary = self.retriever.index.vocabulary
        new_claims = []
        for sq in plan.queries_to_rerun:
            chunks = all_chunks.get(sq.sub_intent, [])
            missing = out_of_corpus_terms(sq.search_query, vocabulary)
            if missing:
                session.logger.log(
                    "evidence_coverage",
                    timestamp_s=chunk.timestamp_s,
                    query=sq.search_query,
                    out_of_corpus_terms=missing,
                    weak=evidence_is_weak(missing),
                )
            claims = synthesize_claims(
                sub_intent=sq.sub_intent,
                chunks=chunks,
                existing_context=constraint_text,
                version=session.ledger.version + 1,
                use_llm=self.use_llm,
            )
            new_claims.extend(claims)

        # Step 4: Add refined claims
        if new_claims:
            snapshot = session.ledger.add_refined_claims(
                new_claims=new_claims,
                superseded_ids=plan.claims_to_supersede,
            )
            result.answer = snapshot
            session.controller.reset_buffer()

            session.logger.log(
                "answer_rendered",
                timestamp_s=chunk.timestamp_s,
                version=snapshot.version,
                claim_ids=[c.id for c in snapshot.claims],
            )

        result.telemetry = {
            "total_latency_ms": (time.time() - t_start) * 1000,
            "refinement": True,
            "affected_intents": matched_intents,
            "claims_superseded": plan.claims_to_supersede,
            "claims_generated": len(new_claims),
        }

        return result

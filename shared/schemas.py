"""
Frozen interfaces. Anyone changing these fields tells the other three people first.

All dataclasses used across modules live here so import cycles never happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Core retrieval types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A scored passage returned from the retrieval engine.

    `score` is the dense cosine similarity between the query and the chunk, which is
    comparable across queries and meaningful on its own. `rrf_score` is the reciprocal
    rank fusion value that actually orders the results: it is rank-based, so its ceiling
    is 2/(k+1) — about 0.033 — for *every* query and it must not be read as relevance.
    """

    doc_id: str
    section: str
    text: str
    score: float
    chunk_id: str = ""
    sub_intent: str | None = None
    rrf_score: float = 0.0
    bm25_score: float = 0.0


@dataclass
class ChunkRecord:
    """A corpus chunk with full metadata, stored in the index."""

    chunk_id: str
    doc_id: str
    section: str
    text: str
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Claim / ledger types
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """One atomic factual assertion with provenance."""

    id: str
    text: str
    chunk_ids: list[str]
    sub_intent: str
    version: int
    status: Literal["grounded", "unverified", "superseded"]
    supersedes: str | None = None  # id of the claim this replaces


# ---------------------------------------------------------------------------
# Controller types
# ---------------------------------------------------------------------------


@dataclass
class SubQuery:
    """A decomposed sub-question ready for retrieval."""

    sub_intent: str
    search_query: str
    original_span: str = ""


@dataclass
class ControllerDecision:
    """The controller's decision on what to do with incoming text."""

    action: Literal["wait", "retrieve", "suppress", "reretrieve"]
    reason: str
    accumulated_text: str
    confidence: float = 1.0
    affected_sub_intents: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer / snapshot types
# ---------------------------------------------------------------------------


@dataclass
class AnswerSnapshot:
    """Immutable view of the answer at a specific version."""

    version: int
    claims: list[Claim]
    citations: dict[str, list[str]]  # claim_id -> chunk_ids
    uncertainty: list[str]  # sub_intents with insufficient evidence
    timestamp_s: float = 0.0


@dataclass
class RefinementPlan:
    """What needs to change when a late constraint arrives."""

    constraint_text: str
    affected_intents: list[str]
    claims_to_supersede: list[str]  # claim_ids
    queries_to_rerun: list[SubQuery] = field(default_factory=list)


@dataclass
class GroundingResult:
    """Result of verifying a claim against its cited chunks."""

    claim_id: str
    is_grounded: bool
    supporting_chunks: list[str]
    unsupported_assertions: list[str] = field(default_factory=list)

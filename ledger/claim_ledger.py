"""
Claim Ledger — THE DIFFERENTIATOR.

Versioned, sub-intent-keyed store of grounded claims. This is what makes
PRISM's answer updates surgical rather than full restarts.

Data model:
    Session → AnswerVersion(v1, v2, …) → Claim[]

Each Claim is an atomic factual assertion tagged with:
    - which corpus chunks support it (chunk_ids)
    - which sub-question it answers (sub_intent)
    - its version number
    - its grounding status (grounded / unverified / superseded)

Key operations:
    add_claims()   → stores new claims, bumps version
    refine()       → marks affected claims superseded, plans re-retrieval
    current_answer() → snapshot of all active claims
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from shared.schemas import (
    AnswerSnapshot,
    Claim,
    RefinementPlan,
    SubQuery,
)

if TYPE_CHECKING:
    from telemetry.logger import TelemetryLogger


class ClaimLedger:
    """Versioned claim store scoped to a single conversation session.

    Thread-safety: not thread-safe. One ledger per session, one session
    per request chain. If we ever need concurrency, wrap mutations in a
    lock.
    """

    def __init__(self, session_id: str, logger: TelemetryLogger | None = None):
        self.session_id = session_id
        self.version = 0
        self._claims: dict[str, Claim] = {}  # claim_id → Claim
        self._history: list[AnswerSnapshot] = []  # ordered snapshots
        self._logger = logger

    # ------------------------------------------------------------------
    # Core mutations
    # ------------------------------------------------------------------

    def add_claims(self, new_claims: list[Claim]) -> AnswerSnapshot:
        """Add new claims and bump the answer version.

        Parameters
        ----------
        new_claims : list[Claim]
            Claims to add. Each must have a unique id.

        Returns
        -------
        AnswerSnapshot
            The new answer state after adding these claims.
        """
        self.version += 1
        added_ids = []

        for claim in new_claims:
            claim.version = self.version
            self._claims[claim.id] = claim
            added_ids.append(claim.id)

        snapshot = self._build_snapshot()
        self._history.append(snapshot)

        # Telemetry
        if self._logger:
            self._logger.answer_version(
                version=self.version,
                claims_added=added_ids,
                claims_superseded=[],
            )
            for claim in new_claims:
                self._logger.log(
                    "claim_drafted",
                    claim_id=claim.id,
                    sub_intent=claim.sub_intent,
                    chunk_ids=claim.chunk_ids,
                    status=claim.status,
                )

        return snapshot

    def refine(
        self,
        constraint_text: str,
        affected_intents: list[str],
    ) -> RefinementPlan:
        """Mark claims on affected sub-intents as superseded.

        Does NOT perform re-retrieval — returns a RefinementPlan that
        tells the pipeline what to re-search. The pipeline calls
        add_claims() again with the replacement claims.

        Parameters
        ----------
        constraint_text : str
            The late-arriving user constraint (e.g. "actually 50 people").
        affected_intents : list[str]
            Sub-intents whose claims need updating.

        Returns
        -------
        RefinementPlan
            Plan indicating what to supersede and re-search.
        """
        claims_to_supersede: list[str] = []
        queries_to_rerun: list[SubQuery] = []

        for claim_id, claim in self._claims.items():
            if claim.sub_intent in affected_intents and claim.status != "superseded":
                claims_to_supersede.append(claim_id)
                # Mark superseded immediately
                claim.status = "superseded"

        # Build re-retrieval queries from the affected intents
        for intent in affected_intents:
            queries_to_rerun.append(
                SubQuery(
                    sub_intent=intent,
                    search_query=f"{constraint_text} {intent.replace('_', ' ')}",
                    original_span=constraint_text,
                )
            )

        # Telemetry
        if self._logger and claims_to_supersede:
            self._logger.log(
                "claims_superseded",
                claims=claims_to_supersede,
                reason=constraint_text,
            )

        return RefinementPlan(
            constraint_text=constraint_text,
            affected_intents=affected_intents,
            claims_to_supersede=claims_to_supersede,
            queries_to_rerun=queries_to_rerun,
        )

    def add_refined_claims(
        self,
        new_claims: list[Claim],
        superseded_ids: list[str],
    ) -> AnswerSnapshot:
        """Add replacement claims after a refinement cycle.

        Like add_claims() but also records which old claims were superseded.
        """
        self.version += 1
        added_ids = []

        for claim in new_claims:
            claim.version = self.version
            if superseded_ids:
                claim.supersedes = superseded_ids[0]  # primary supersession
            self._claims[claim.id] = claim
            added_ids.append(claim.id)

        snapshot = self._build_snapshot()
        self._history.append(snapshot)

        if self._logger:
            self._logger.answer_version(
                version=self.version,
                claims_added=added_ids,
                claims_superseded=superseded_ids,
            )
            for claim in new_claims:
                self._logger.log(
                    "claim_drafted",
                    claim_id=claim.id,
                    sub_intent=claim.sub_intent,
                    chunk_ids=claim.chunk_ids,
                    status=claim.status,
                    supersedes=claim.supersedes,
                )

        return snapshot

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def current_answer(self) -> AnswerSnapshot:
        """Return the current answer state (only active claims)."""
        return self._build_snapshot()

    def get_claim(self, claim_id: str) -> Claim | None:
        """Look up a single claim by ID."""
        return self._claims.get(claim_id)

    def get_claims_by_intent(self, sub_intent: str) -> list[Claim]:
        """Return all claims (including superseded) for a sub-intent."""
        return [c for c in self._claims.values() if c.sub_intent == sub_intent]

    def get_active_claims(self) -> list[Claim]:
        """Return only non-superseded claims."""
        return [c for c in self._claims.values() if c.status != "superseded"]

    def get_history(self) -> list[AnswerSnapshot]:
        """Return the full version history."""
        return list(self._history)

    @property
    def claim_count(self) -> int:
        return len(self._claims)

    @property
    def active_claim_count(self) -> int:
        return len(self.get_active_claims())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> AnswerSnapshot:
        """Build an AnswerSnapshot from current state."""
        active = self.get_active_claims()
        citations: dict[str, list[str]] = {}
        uncertainty: list[str] = []

        seen_intents: set[str] = set()
        for claim in active:
            citations[claim.id] = claim.chunk_ids
            seen_intents.add(claim.sub_intent)
            if claim.status == "unverified" and claim.sub_intent not in uncertainty:
                uncertainty.append(claim.sub_intent)

        return AnswerSnapshot(
            version=self.version,
            claims=active,
            citations=citations,
            uncertainty=uncertainty,
            timestamp_s=time.time(),
        )

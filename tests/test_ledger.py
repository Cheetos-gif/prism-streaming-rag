"""Tests for ledger/claim_ledger.py — the core differentiator."""

from ledger.claim_ledger import ClaimLedger
from shared.schemas import Claim


def _make_claim(sub_intent: str, version: int = 1, text: str = "") -> Claim:
    return Claim(
        id=f"claim_v{version}_{sub_intent}",
        text=text or f"Test claim for {sub_intent}",
        chunk_ids=[f"chunk_{sub_intent}_1", f"chunk_{sub_intent}_2"],
        sub_intent=sub_intent,
        version=version,
        status="grounded",
    )


class TestAddClaims:
    def test_add_claims_bumps_version(self):
        ledger = ClaimLedger("test_session")
        claims = [_make_claim("venue_capacity"), _make_claim("cancellation_policy")]
        snapshot = ledger.add_claims(claims)

        assert ledger.version == 1
        assert snapshot.version == 1
        assert len(snapshot.claims) == 2

    def test_claims_stored_correctly(self):
        ledger = ClaimLedger("test_session")
        claims = [_make_claim("venue_capacity")]
        ledger.add_claims(claims)

        stored = ledger.get_claim("claim_v1_venue_capacity")
        assert stored is not None
        assert stored.sub_intent == "venue_capacity"
        assert stored.status == "grounded"

    def test_multiple_add_rounds_increment_version(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims([_make_claim("venue")])
        ledger.add_claims([_make_claim("catering")])

        assert ledger.version == 2
        assert ledger.claim_count == 2


class TestRefine:
    def test_refine_supersedes_affected_claims(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims(
            [
                _make_claim("venue_capacity"),
                _make_claim("cancellation_policy"),
                _make_claim("catering_options"),
            ]
        )

        plan = ledger.refine(
            constraint_text="actually 50 people not 30",
            affected_intents=["venue_capacity"],
        )

        assert len(plan.claims_to_supersede) == 1
        assert "claim_v1_venue_capacity" in plan.claims_to_supersede
        assert len(plan.queries_to_rerun) == 1

        # Verify the old claim is superseded
        old_claim = ledger.get_claim("claim_v1_venue_capacity")
        assert old_claim.status == "superseded"

    def test_refine_preserves_unaffected_claims(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims(
            [
                _make_claim("venue_capacity"),
                _make_claim("cancellation_policy"),
                _make_claim("catering_options"),
            ]
        )

        ledger.refine(
            constraint_text="actually 50 people",
            affected_intents=["venue_capacity"],
        )

        # Cancellation and catering should be untouched
        cancel = ledger.get_claim("claim_v1_cancellation_policy")
        catering = ledger.get_claim("claim_v1_catering_options")
        assert cancel.status == "grounded"
        assert catering.status == "grounded"

    def test_add_refined_claims_creates_new_version(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims(
            [
                _make_claim("venue_capacity"),
                _make_claim("cancellation_policy"),
            ]
        )

        plan = ledger.refine("50 people", ["venue_capacity"])

        new_claim = _make_claim("venue_capacity", version=2, text="Updated venue claim")
        new_claim.id = "claim_v2_venue_capacity"
        ledger.add_refined_claims([new_claim], plan.claims_to_supersede)

        assert ledger.version == 2
        # Active claims: new venue + old cancellation
        active = ledger.get_active_claims()
        assert len(active) == 2
        active_ids = {c.id for c in active}
        assert "claim_v2_venue_capacity" in active_ids
        assert "claim_v1_cancellation_policy" in active_ids


class TestCurrentAnswer:
    def test_current_answer_excludes_superseded(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims(
            [
                _make_claim("venue_capacity"),
                _make_claim("cancellation_policy"),
            ]
        )

        ledger.refine("50 people", ["venue_capacity"])

        new_claim = _make_claim("venue_capacity", version=2)
        new_claim.id = "claim_v2_venue_capacity"
        ledger.add_refined_claims([new_claim], ["claim_v1_venue_capacity"])

        answer = ledger.current_answer()
        answer_ids = {c.id for c in answer.claims}
        assert "claim_v1_venue_capacity" not in answer_ids
        assert "claim_v2_venue_capacity" in answer_ids

    def test_citations_populated(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims([_make_claim("venue_capacity")])

        answer = ledger.current_answer()
        assert "claim_v1_venue_capacity" in answer.citations
        assert len(answer.citations["claim_v1_venue_capacity"]) == 2

    def test_uncertainty_flags_unverified_claims(self):
        ledger = ClaimLedger("test_session")
        unverified = Claim(
            id="claim_v1_parking",
            text="Parking info not found",
            chunk_ids=[],
            sub_intent="parking",
            version=1,
            status="unverified",
        )
        ledger.add_claims([unverified])

        answer = ledger.current_answer()
        assert "parking" in answer.uncertainty


class TestHistory:
    def test_history_tracks_all_versions(self):
        ledger = ClaimLedger("test_session")
        ledger.add_claims([_make_claim("a")])
        ledger.add_claims([_make_claim("b")])

        history = ledger.get_history()
        assert len(history) == 2
        assert history[0].version == 1
        assert history[1].version == 2

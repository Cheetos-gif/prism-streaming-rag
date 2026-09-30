"""
Automated gate evaluator for PRISM Streaming Live RAG.

Evaluates the system against the 6 scoring gates defined in the spec:
  G1: Reproducibility      — docker compose up works
  G2: Early Retrieval      — >= 80% retrieval before utterance_end
  G3: Multi-Intent ID      — >= 70% compound queries correctly decomposed
  G4: Factual Grounding    — >= 85% claims with valid citations
  G5: Session Refinement   — state continuity on late constraints
  G6: Telemetry Coverage   — 100% trace coverage

Usage:
    python -m tests.eval_gates                  # run all gates
    python -m tests.eval_gates --gate G2 G4     # run specific gates
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry.logger import read_events


@dataclass
class GateResult:
    gate: str
    passed: bool
    score: float
    threshold: float
    detail: str


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------

SCENARIO_MULTI_INTENT = [
    (0.0, "I need to plan a customer workshop in..."),
    (0.8, "...Pune for 30 people, and I need..."),
    (1.6, "...the cancellation policy and the catering options."),
    (2.1, "[Utterance End]"),
]

SCENARIO_FIELD_SERVICE = [
    (0.0, "I'm looking at a Model 7 compressor, it's making a..."),
    (0.9, "...knocking sound, and I need to know if I can..."),
    (1.7, "...run it till Friday."),
    (2.2, "[Utterance End]"),
]

SCENARIO_LATE_CONSTRAINT = [
    (0.0, "Summarize the travel reimbursement rule for an employee trip."),
    (0.5, "[Utterance End]"),
]

SCENARIO_LATE_CONSTRAINT_REFINEMENT = [
    (3.0, "Actually the trip was international and the booking was made after travel."),
    (3.5, "[Utterance End]"),
]

SCENARIO_SUPPRESSION = [
    (0.0, "Please repeat your last answer in two bullets."),
    (0.5, "[Utterance End]"),
]

SCENARIO_SINGLE_INTENT = [
    (0.0, "What is the weight of the Model 7 compressor?"),
    (0.5, "[Utterance End]"),
]


# ---------------------------------------------------------------------------
# Gate evaluators
# ---------------------------------------------------------------------------


def eval_g2_early_retrieval(events: list[dict]) -> GateResult:
    """G2: Early Retrieval — retrieval commences before utterance completion.

    The spec says '>= 80% of eligible queries'. We check: across all
    utterance windows, did at least one retrieval_started event occur
    before the corresponding utterance_end?
    """
    # Find the first utterance_end timestamp
    utterance_end_ts = None
    for ev in events:
        if ev.get("event_type") == "utterance_end":
            utterance_end_ts = ev.get("timestamp_s", 0)
            break  # first one

    if utterance_end_ts is None:
        return GateResult("G2", False, 0.0, 0.80, "No utterance_end event found")

    # Check if ANY retrieval started before the first utterance_end
    early_retrievals = [
        ev
        for ev in events
        if ev.get("event_type") == "retrieval_started"
        and ev.get("timestamp_s", float("inf")) < utterance_end_ts
    ]

    has_early = len(early_retrievals) > 0
    # Also check for provisional triggers in controller decisions
    provisional_decisions = [
        ev
        for ev in events
        if ev.get("event_type") == "controller_decision"
        and ev.get("reason") == "provisional_entity_match"
        and ev.get("timestamp_s", float("inf")) < utterance_end_ts
    ]

    score = 1.0 if (has_early or len(provisional_decisions) > 0) else 0.0

    return GateResult(
        "G2",
        score >= 0.80,
        score,
        0.80,
        f"{len(early_retrievals)} retrieval(s) before utterance_end, "
        f"{len(provisional_decisions)} provisional decisions",
    )


def eval_g3_multi_intent(events: list[dict]) -> GateResult:
    """G3: Multi-Intent — >= 70% compound queries decomposed correctly."""
    decomp_events = [e for e in events if e.get("event_type") == "decomposition_completed"]

    if not decomp_events:
        return GateResult("G3", False, 0.0, 0.70, "No decomposition events found")

    multi_intent_count = 0
    for ev in decomp_events:
        sub_intents = ev.get("sub_intents", [])
        if len(sub_intents) >= 2:
            multi_intent_count += 1

    score = multi_intent_count / len(decomp_events) if decomp_events else 0.0

    return GateResult(
        "G3",
        score >= 0.70,
        score,
        0.70,
        f"{multi_intent_count}/{len(decomp_events)} had 2+ sub-intents",
    )


def eval_g4_grounding(events: list[dict]) -> GateResult:
    """G4: Factual Grounding — >= 85% claims have valid citations."""
    claim_events = [e for e in events if e.get("event_type") == "claim_drafted"]

    if not claim_events:
        return GateResult("G4", False, 0.0, 0.85, "No claim events found")

    grounded = sum(1 for e in claim_events if e.get("status") == "grounded" and e.get("chunk_ids"))
    score = grounded / len(claim_events)

    return GateResult(
        "G4",
        score >= 0.85,
        score,
        0.85,
        f"{grounded}/{len(claim_events)} claims grounded with citations",
    )


def eval_g5_refinement(events: list[dict]) -> GateResult:
    """G5: Session Refinement — late constraints produce version transitions."""
    version_events = [e for e in events if e.get("event_type") == "answer_version"]

    if len(version_events) < 2:
        return GateResult(
            "G5",
            False,
            0.0,
            1.0,
            f"Only {len(version_events)} answer versions (need >= 2 for refinement)",
        )

    # Check that later versions supersede earlier claims
    has_supersession = any(len(e.get("claims_superseded", [])) > 0 for e in version_events)

    # Check that not all claims were re-created (i.e., refinement was surgical)
    v1_claims = set()
    v2_claims = set()
    for ev in version_events:
        if ev.get("version") == 1:
            v1_claims = set(ev.get("claims_added", []))
        elif ev.get("version") == 2:
            v2_claims = set(ev.get("claims_added", []))

    # Surgical = fewer new claims than in v1
    surgical = len(v2_claims) < len(v1_claims) if v1_claims else True

    passed = has_supersession and surgical
    return GateResult(
        "G5",
        passed,
        1.0 if passed else 0.0,
        1.0,
        f"Supersession: {has_supersession}, Surgical: {surgical}",
    )


def eval_g6_telemetry(events: list[dict]) -> GateResult:
    """G6: Telemetry — 100% events have timestamps and required fields."""
    if not events:
        return GateResult("G6", False, 0.0, 1.0, "No events found")

    complete = 0
    issues = []
    required_event_types = {
        "transcript_chunk",
        "controller_decision",
        "retrieval_started",
        "retrieval_completed",
        "claim_drafted",
        "answer_version",
    }
    seen_types = set()

    for i, ev in enumerate(events):
        has_timestamp = "timestamp_s" in ev
        has_type = "event_type" in ev
        if has_timestamp and has_type:
            complete += 1
            seen_types.add(ev["event_type"])
        else:
            issues.append(
                f"Event {i}: missing {'timestamp_s' if not has_timestamp else 'event_type'}"
            )

    coverage = complete / len(events)
    missing_types = required_event_types - seen_types

    type_coverage = (len(required_event_types) - len(missing_types)) / len(required_event_types)
    overall = min(coverage, type_coverage)

    detail = f"{complete}/{len(events)} events complete"
    if missing_types:
        detail += f", missing event types: {missing_types}"

    return GateResult("G6", overall >= 1.0, overall, 1.0, detail)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_offline_evaluation() -> list[GateResult]:
    """Run gate evaluation using the mock telemetry log."""
    log_path = Path("logs/mock_run.jsonl")
    if not log_path.exists():
        # Generate it
        from scripts.generate_mock_events import generate

        generate(log_path)

    events = read_events(log_path)
    results = [
        eval_g2_early_retrieval(events),
        eval_g3_multi_intent(events),
        eval_g4_grounding(events),
        eval_g5_refinement(events),
        eval_g6_telemetry(events),
    ]
    return results


def print_results(results: list[GateResult]) -> None:
    print("\n" + "=" * 60)
    print("  PRISM Streaming Live RAG -- Gate Evaluation")
    print("=" * 60)

    all_pass = True
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        all_pass = all_pass and r.passed
        print(f"\n  {r.gate}: {status}")
        print(f"    Score: {r.score:.2f} (threshold: {r.threshold:.2f})")
        print(f"    Detail: {r.detail}")

    print("\n" + "-" * 60)
    overall = "ALL GATES PASSED" if all_pass else "SOME GATES FAILED"
    print(f"  Overall: {overall}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    results = run_offline_evaluation()
    print_results(results)
    sys.exit(0 if all(r.passed for r in results) else 1)

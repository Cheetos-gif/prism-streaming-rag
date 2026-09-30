"""
Generate a realistic fake telemetry log (logs/mock_run.jsonl) so the
dashboard has something real to render against without waiting on the
rest of the pipeline to exist.

Simulates one full utterance from the travel-workshop demo scenario:

    "I need to plan a customer workshop in Pune for 30 people, and I need
    the cancellation policy and the catering options."

...followed by a late-arriving correction ("actually, 50 people") that
triggers a second retrieval round and a superseding answer version --
the scenario the ledger's versioning logic exists to handle.

Usage:
    python scripts/generate_mock_events.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telemetry.logger import TelemetryLogger

DEFAULT_OUTPUT_PATH = Path("logs") / "mock_run.jsonl"


def generate(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    logger = TelemetryLogger(output_path)

    # --- Controller: transcript streams in, chunk by chunk ---------------
    logger.log(
        "transcript_chunk",
        timestamp_s=0.0,
        text="I need to plan a customer workshop in...",
    )
    logger.log(
        "transcript_chunk",
        timestamp_s=0.8,
        text="...Pune for 30 people, and I need...",
    )

    # --- Controller: provisional early retrieval triggered ----------------
    # Entities detected (Pune, 30 people) before utterance completes.
    # This is what makes it "streaming" — retrieval starts mid-speech.
    logger.log(
        "controller_decision",
        timestamp_s=0.8,
        action="retrieve",
        reason="provisional_entity_match",
    )
    logger.retrieval_started(
        query="workshop venue Pune 30 people",
        trigger="provisional",
        timestamp_s=0.85,
    )
    logger.retrieval_completed(
        query="workshop venue Pune 30 people",
        chunk_ids=["chunk_venue_capacity_1", "chunk_venue_capacity_2"],
        latency_ms=140.0,
    )

    logger.log(
        "transcript_chunk",
        timestamp_s=1.6,
        text="...the cancellation policy and the catering options.",
    )
    logger.log("utterance_end", timestamp_s=2.1)

    # --- Controller: decides to decompose + trigger retrieval ------------
    logger.log(
        "controller_decision",
        timestamp_s=2.1,
        action="decompose_and_retrieve",
        reason="utterance_end",
    )

    # --- Decomposer: splits into independent sub-intents ------------------
    logger.log(
        "decomposition_started",
        timestamp_s=2.12,
        utterance="I need to plan a customer workshop in Pune for 30 people, "
        "and I need the cancellation policy and the catering options.",
    )
    sub_intents = [
        {"sub_intent": "venue_capacity", "query": "workshop venue Pune 30 people"},
        {"sub_intent": "cancellation_policy", "query": "cancellation policy"},
        {"sub_intent": "catering_options", "query": "catering options"},
    ]
    logger.log(
        "decomposition_completed",
        timestamp_s=2.18,
        sub_intents=[s["sub_intent"] for s in sub_intents],
        latency_ms=60.0,
    )

    # --- Retrieval: one round per sub-intent ------------------------------
    claim_ids = []
    t = 2.2
    for i, sub in enumerate(sub_intents):
        logger.retrieval_started(query=sub["query"], trigger="sub_intent", timestamp_s=t)
        chunk_ids = [f"chunk_{sub['sub_intent']}_{j}" for j in range(1, 3)]
        logger.retrieval_completed(
            query=sub["query"], chunk_ids=chunk_ids, latency_ms=140.0 + i * 15
        )
        claim_id = f"claim_v1_{sub['sub_intent']}"
        claim_ids.append(claim_id)
        logger.log(
            "claim_drafted",
            timestamp_s=t + 0.16,
            claim_id=claim_id,
            sub_intent=sub["sub_intent"],
            chunk_ids=chunk_ids,
            status="grounded",
        )
        t += 0.3

    # --- Ledger: first answer version, all three sub-intents answered ----
    logger.answer_version(version=1, claims_added=claim_ids, claims_superseded=[])
    logger.log("answer_rendered", timestamp_s=t + 0.05, version=1, claim_ids=claim_ids)

    # --- Late constraint arrives mid-conversation: "actually, 50 people" -
    late_t = t + 1.2
    logger.log(
        "transcript_chunk",
        timestamp_s=late_t,
        text="Sorry, actually make that 50 people, not 30.",
    )
    logger.log(
        "controller_decision",
        timestamp_s=late_t + 0.02,
        action="reretrieve",
        reason="constraint_updated:group_size",
    )

    refine_query = "workshop venue Pune 50 people"
    logger.retrieval_started(
        query=refine_query, trigger="constraint_update", timestamp_s=late_t + 0.05
    )
    refine_chunk_ids = ["chunk_venue_capacity_3", "chunk_venue_capacity_4"]
    logger.retrieval_completed(query=refine_query, chunk_ids=refine_chunk_ids, latency_ms=132.0)

    superseded_claim_id = claim_ids[0]  # venue_capacity claim from v1
    refined_claim_id = "claim_v2_venue_capacity"
    logger.log(
        "claim_drafted",
        timestamp_s=late_t + 0.2,
        claim_id=refined_claim_id,
        sub_intent="venue_capacity",
        chunk_ids=refine_chunk_ids,
        status="grounded",
        supersedes=superseded_claim_id,
    )

    # --- Ledger: second answer version, venue claim superseded -----------
    logger.answer_version(
        version=2,
        claims_added=[refined_claim_id],
        claims_superseded=[superseded_claim_id],
    )
    logger.log(
        "answer_rendered",
        timestamp_s=late_t + 0.3,
        version=2,
        claim_ids=[refined_claim_id, claim_ids[1], claim_ids[2]],
    )

    # --- A suppressed low-confidence aside, for dashboard coverage -------
    logger.log(
        "transcript_chunk",
        timestamp_s=late_t + 1.0,
        text="Oh, and do you have parking nearby?",
    )
    logger.suppressed(reason="low_confidence_retrieval")

    logger.close()
    return logger.output_path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    written_path = generate(out)
    print(f"Wrote mock telemetry log to {written_path}")

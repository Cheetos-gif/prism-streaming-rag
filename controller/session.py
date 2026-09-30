"""
Ephemeral session state — one per conversation.

Holds everything needed to process a streaming conversation:
  - The claim ledger (versioned answer state)
  - The retrieval controller (decision logic)
  - The telemetry logger (event recording)
  - Accumulated transcript text

Session-bound: no cross-session persistence, no user tracking.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from controller.retrieval_controller import RetrievalController
from ledger.claim_ledger import ClaimLedger
from telemetry.logger import TelemetryLogger


class Session:
    """A single conversation session with ephemeral state."""

    def __init__(
        self,
        session_id: str | None = None,
        log_dir: str | Path = "logs",
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.created_at = time.time()

        # Per-session telemetry log
        log_path = Path(log_dir) / f"session_{self.session_id}.jsonl"
        self.logger = TelemetryLogger(log_path)

        # Core components
        self.controller = RetrievalController()
        self.ledger = ClaimLedger(self.session_id, self.logger)

        # Transcript accumulation
        self.transcript_chunks: list[dict] = []
        self.full_transcript: str = ""

    def append_transcript(self, timestamp_s: float, text: str) -> None:
        """Record a transcript chunk."""
        self.transcript_chunks.append({
            "timestamp_s": timestamp_s,
            "text": text,
        })
        self.full_transcript = " ".join(
            chunk["text"] for chunk in self.transcript_chunks
        )
        self.logger.log(
            "transcript_chunk",
            timestamp_s=timestamp_s,
            text=text,
        )

    def close(self) -> None:
        """Close the session's telemetry logger."""
        self.logger.close()

    def to_dict(self) -> dict:
        """Serialize session state for API responses."""
        answer = self.ledger.current_answer()
        return {
            "session_id": self.session_id,
            "version": self.ledger.version,
            "active_claims": self.ledger.active_claim_count,
            "total_claims": self.ledger.claim_count,
            "transcript_chunks": len(self.transcript_chunks),
            "answer": {
                "version": answer.version,
                "claims": [
                    {
                        "id": c.id,
                        "text": c.text,
                        "sub_intent": c.sub_intent,
                        "chunk_ids": c.chunk_ids,
                        "status": c.status,
                        "version": c.version,
                    }
                    for c in answer.claims
                ],
                "citations": answer.citations,
                "uncertainty": answer.uncertainty,
            },
        }


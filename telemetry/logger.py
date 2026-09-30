"""
Structured event logger for Streaming Live RAG observability.

Every pipeline stage (controller, decomposer, retrieval, ledger) calls
TelemetryLogger to emit one JSON object per line (JSONL). Zero
dependencies on the logic of those stages -- this module only knows how
to serialize events and read them back.

Stdlib only: json, time, pathlib.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TelemetryLogger:
    """Append-only JSONL event logger.

    Every `.log()` call writes exactly one JSON object as a line and
    flushes immediately, so a crash mid-run never loses events already
    emitted.
    """

    def __init__(self, output_path: str | Path | None = None):
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path("logs") / f"run_{timestamp}.jsonl"
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered text mode; we still call flush() explicitly after
        # every write to guarantee durability across process crashes.
        self._file = self.output_path.open("a", encoding="utf-8")

    def log(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Write one JSON line: {"event_type", "timestamp_s", **fields}.

        `fields` may itself include a `timestamp_s` override (e.g. a
        caller-supplied transcript timestamp); it takes precedence over
        the wall-clock time captured here.
        """
        event = {"event_type": event_type, "timestamp_s": time.time(), **fields}
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()
        return event

    def retrieval_started(self, query: str, trigger: str, timestamp_s: float) -> dict[str, Any]:
        return self.log(
            "retrieval_started",
            query=query,
            trigger=trigger,
            timestamp_s=timestamp_s,
        )

    def retrieval_completed(
        self, query: str, chunk_ids: list[str], latency_ms: float
    ) -> dict[str, Any]:
        return self.log(
            "retrieval_completed",
            query=query,
            chunk_ids=chunk_ids,
            latency_ms=latency_ms,
        )

    def answer_version(
        self, version: int, claims_added: list[str], claims_superseded: list[str]
    ) -> dict[str, Any]:
        return self.log(
            "answer_version",
            version=version,
            claims_added=claims_added,
            claims_superseded=claims_superseded,
        )

    def suppressed(self, reason: str) -> dict[str, Any]:
        return self.log("suppressed", reason=reason)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TelemetryLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @classmethod
    def read_events(cls, path: str | Path) -> list[dict[str, Any]]:
        """Read a JSONL event log back into a list of dicts."""
        events = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return events


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Module-level convenience wrapper around TelemetryLogger.read_events."""
    return TelemetryLogger.read_events(path)

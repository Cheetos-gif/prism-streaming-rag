"""Tests for telemetry/logger.py."""
import json

from telemetry.logger import TelemetryLogger


def test_logged_event_round_trips_through_read_events(tmp_path):
    log_path = tmp_path / "run.jsonl"
    logger = TelemetryLogger(log_path)

    written = logger.retrieval_completed(
        query="cancellation policy Pune",
        chunk_ids=["c1", "c2"],
        latency_ms=123.4,
    )
    logger.close()

    events = TelemetryLogger.read_events(log_path)

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "retrieval_completed"
    assert event["query"] == written["query"] == "cancellation policy Pune"
    assert event["chunk_ids"] == ["c1", "c2"]
    assert event["latency_ms"] == 123.4
    assert "timestamp_s" in event


def test_log_file_is_valid_jsonl_one_object_per_line(tmp_path):
    log_path = tmp_path / "run.jsonl"
    logger = TelemetryLogger(log_path)

    logger.retrieval_started(query="q1", trigger="utterance_pause", timestamp_s=0.8)
    logger.answer_version(version=1, claims_added=["a1"], claims_superseded=[])
    logger.suppressed(reason="low_confidence")
    logger.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [e["event_type"] for e in parsed] == [
        "retrieval_started",
        "answer_version",
        "suppressed",
    ]
    assert parsed[0]["timestamp_s"] == 0.8

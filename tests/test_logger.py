"""Tests for telemetry/logger.py."""
import json

from telemetry.logger import TelemetryLogger, read_events


def test_logged_events_round_trip_through_read_events(tmp_path):
    log_path = tmp_path / "run.jsonl"
    logger = TelemetryLogger(log_path)

    logger.retrieval_started(query="cancellation policy Pune", trigger="sub_intent", timestamp_s=0.8)
    written = logger.retrieval_completed(
        query="cancellation policy Pune",
        chunk_ids=["c1", "c2"],
        latency_ms=123.4,
    )
    logger.answer_version(version=1, claims_added=["a1"], claims_superseded=[])
    logger.suppressed(reason="low_confidence")
    logger.close()

    events = read_events(log_path)

    assert len(events) == 4
    completed = events[1]
    assert completed["event_type"] == "retrieval_completed"
    assert completed["query"] == written["query"] == "cancellation policy Pune"
    assert completed["chunk_ids"] == ["c1", "c2"]
    assert completed["latency_ms"] == 123.4
    assert "timestamp_s" in completed

    assert events[0]["timestamp_s"] == 0.8  # explicit override honored
    assert events[2]["event_type"] == "answer_version"
    assert events[2]["claims_added"] == ["a1"]
    assert events[3]["event_type"] == "suppressed"
    assert events[3]["reason"] == "low_confidence"


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
    assert all(isinstance(e, dict) for e in parsed)

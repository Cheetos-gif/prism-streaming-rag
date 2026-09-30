"""Tests for controller/stream_simulator.py."""

from controller.stream_simulator import (
    FIELD_SERVICE_SCRIPT,
    TRAVEL_WORKSHOP_SCRIPT,
    StreamSimulator,
)


def test_chunks_come_back_in_timestamp_order():
    simulator = StreamSimulator(TRAVEL_WORKSHOP_SCRIPT)
    chunks = list(simulator.chunks())

    timestamps = [chunk.timestamp_s for chunk in chunks]
    assert timestamps == sorted(timestamps)
    assert len(chunks) == len(TRAVEL_WORKSHOP_SCRIPT)


def test_last_chunk_is_flagged_final_and_others_are_not():
    simulator = StreamSimulator(TRAVEL_WORKSHOP_SCRIPT)
    chunks = list(simulator.chunks())

    assert chunks[-1].is_final is True
    assert all(chunk.is_final is False for chunk in chunks[:-1])


def test_field_service_script_also_orders_and_finalizes_correctly():
    simulator = StreamSimulator(FIELD_SERVICE_SCRIPT)
    chunks = list(simulator.chunks())

    timestamps = [chunk.timestamp_s for chunk in chunks]
    assert timestamps == sorted(timestamps)
    assert chunks[-1].is_final is True


def test_run_dispatches_chunks_synchronously_in_order_by_default():
    simulator = StreamSimulator(TRAVEL_WORKSHOP_SCRIPT)
    received = []

    simulator.run(received.append)

    assert [c.text for c in received] == [text for _, text in TRAVEL_WORKSHOP_SCRIPT]
    assert received[-1].is_final is True

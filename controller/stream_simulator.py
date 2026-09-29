"""
Transcript stream simulator for Streaming Live RAG.

Replays a scripted utterance as timestamped chunks, mimicking a user
speaking in real time. Used to drive the controller pipeline in unit
tests / offline eval (synchronous) and in demos (real-time playback).

Example (matches the brief):

    0.0s  "I need to plan a customer workshop in..."
    0.8s  "...Pune for 30 people, and I need..."
    1.6s  "...the cancellation policy and the catering options."
    2.1s  [Utterance End]
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

# shared/schemas.py defines the Chunk/Claim types the retrieval and ledger
# subsystems trade in. This module doesn't depend on them -- it only ever
# emits raw transcript text -- but importing here documents the relationship
# and keeps the type checkable if a caller wants to build Chunk/Claim from
# TranscriptChunk.text downstream.
from shared.schemas import Chunk, Claim  # noqa: F401


@dataclass
class TranscriptChunk:
    """A single piece of a streamed transcript."""

    timestamp_s: float
    text: str
    is_final: bool


# A script is an ordered list of (timestamp_s, text) pairs. The final entry
# conventionally marks the end of the utterance (e.g. text="[Utterance End]").
Script = Sequence[tuple[float, str]]


class StreamSimulator:
    """Replays a scripted utterance as a sequence of TranscriptChunks."""

    def __init__(self, script: Script):
        if not script:
            raise ValueError("script must contain at least one (timestamp, text) pair")
        self.script: list[tuple[float, str]] = list(script)

    def chunks(self) -> Iterator[TranscriptChunk]:
        """Yield chunks synchronously, in script order, with no sleeping.

        Intended for unit tests and offline evaluation where real-time
        pacing is irrelevant.
        """
        last_index = len(self.script) - 1
        for index, (timestamp_s, text) in enumerate(self.script):
            yield TranscriptChunk(
                timestamp_s=timestamp_s,
                text=text,
                is_final=(index == last_index),
            )

    def run(
        self,
        on_chunk: Callable[[TranscriptChunk], None],
        real_time: bool = False,
        speed: float = 1.0,
    ) -> None:
        """Dispatch each chunk to `on_chunk`, in order.

        If `real_time` is True, sleeps between chunks so that wall-clock
        gaps match the differences between successive `timestamp_s` values,
        scaled by `1 / speed` (speed=2.0 plays twice as fast).
        """
        if speed <= 0:
            raise ValueError("speed must be positive")

        previous_timestamp_s = 0.0
        for chunk in self.chunks():
            if real_time:
                gap_s = (chunk.timestamp_s - previous_timestamp_s) / speed
                if gap_s > 0:
                    time.sleep(gap_s)
                previous_timestamp_s = chunk.timestamp_s
            on_chunk(chunk)


# Matches the brief's travel example verbatim.
TRAVEL_WORKSHOP_SCRIPT: Script = [
    (0.0, "I need to plan a customer workshop in..."),
    (0.8, "...Pune for 30 people, and I need..."),
    (1.6, "...the cancellation policy and the catering options."),
    (2.1, "[Utterance End]"),
]

# Field-service demo scenario.
FIELD_SERVICE_SCRIPT: Script = [
    (0.0, "I'm looking at a Model 7 compressor, it's making a..."),
    (0.9, "...knocking sound, and I need to know if I can..."),
    (1.7, "...run it till Friday."),
    (2.2, "[Utterance End]"),
]


if __name__ == "__main__":
    simulator = StreamSimulator(FIELD_SERVICE_SCRIPT)
    for chunk in simulator.chunks():
        marker = " [FINAL]" if chunk.is_final else ""
        print(f"{chunk.timestamp_s:.1f}s  {chunk.text!r}{marker}")

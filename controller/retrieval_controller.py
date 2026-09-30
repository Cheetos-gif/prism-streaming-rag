"""
Retrieval Controller — the brain that decides WHEN to search.

Evaluates incoming transcript chunks and makes one of four decisions:
  WAIT       – text is incomplete, wait for more
  RETRIEVE   – stable intent detected, trigger corpus search
  SUPPRESS   – presentation-only request (reformat, repeat, shorten)
  RERETRIEVE – late constraint modifies an existing answer

Design: heuristic-first for speed (<1ms), with optional LLM fallback
for ambiguous cases. This means 80%+ of decisions are instantaneous.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from shared.schemas import ControllerDecision

if TYPE_CHECKING:
    from controller.stream_simulator import TranscriptChunk


_SUPPRESS_PATTERNS = re.compile(
    r"\b(repeat\s+(?:your\s+)?(?:last\s+answer|that)|"
    r"say\s+(?:that|it)\s+again|"
    r"make\s+(?:that|it)\s+shorter|"
    r"shorter\s+please|"
    r"reformat\s+(?:that|it)?\s*(?:as|in)?\s*bullet|"
    r"in\s+two\s+(?:bullets|points)|"
    r"in\s+\d+\s+(?:bullets|points)|"
    r"bullet\s*points?\s+only|"
    r"summarize\s+(?:your\s+last\s+answer|that|the\s+above))\b",
    re.IGNORECASE,
)

_REFINEMENT_PATTERNS = re.compile(
    r"\b(actually|correction|sorry|change\s+(?:that|it)|"
    r"not\s+\d+.*\b\d+|make\s+(?:that|it)|instead\s+of|"
    r"update|modify|wait.*(?:meant|mean)|no.*(?:meant|mean))\b",
    re.IGNORECASE,
)

_ENTITY_PATTERN = re.compile(
    r"(?:"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"  # proper nouns
    r"|\b\d+\s*(?:people|persons|pax|attendees|units|hours|days|kg|lbs|psi|cfm)\b"  # quantities
    r"|\bmodel\s+\d+\b"  # model numbers
    r"|\b(?:Pune|Mumbai|Delhi|Bangalore|Chennai|Hyderabad)\b"  # known cities
    r")",
    re.IGNORECASE,
)

# Signals that an utterance is a complete thought
_COMPLETENESS_SIGNALS = re.compile(
    r"[.?!]$|"
    r"\b(?:can\s+(?:I|you)|what\s+(?:is|are)|how\s+(?:do|does|much|many|long)|"
    r"tell\s+me|I\s+need|please|could\s+you)\b",
    re.IGNORECASE,
)


class RetrievalController:
    """Stateful controller scoped to a single session.

    Accumulates transcript text and emits decisions per chunk.
    """

    def __init__(
        self,
        entity_threshold: int = 2,
        confidence_threshold: float = 0.6,
    ):
        self._buffer: list[str] = []
        self._entity_threshold = entity_threshold
        self._confidence_threshold = confidence_threshold
        self._has_answered = False  # True after first RETRIEVE completes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_chunk(self, chunk: TranscriptChunk) -> ControllerDecision:
        """Process a single transcript chunk and decide what to do.

        Parameters
        ----------
        chunk : TranscriptChunk
            The incoming transcript fragment with timestamp and is_final flag.

        Returns
        -------
        ControllerDecision
        """
        text = chunk.text.strip()

        # Utterance end marker — always force action
        if chunk.is_final or text == "[Utterance End]":
            return self._on_utterance_end()

        self._buffer.append(text)
        accumulated = " ".join(self._buffer)

        # 1) Check suppression first (cheapest)
        if self._is_suppression(accumulated):
            return ControllerDecision(
                action="suppress",
                reason="presentation_restructure",
                accumulated_text=accumulated,
                confidence=0.95,
            )

        # 2) Check refinement (late constraint)
        if self._has_answered and self._is_refinement(accumulated):
            return ControllerDecision(
                action="reretrieve",
                reason="constraint_updated",
                accumulated_text=accumulated,
                confidence=0.85,
            )

        # 3) Check entity density + completeness → provisional retrieve
        entity_count = len(_ENTITY_PATTERN.findall(accumulated))
        looks_complete = bool(_COMPLETENESS_SIGNALS.search(accumulated))

        if entity_count >= self._entity_threshold and looks_complete:
            return ControllerDecision(
                action="retrieve",
                reason="stable_intent_detected",
                accumulated_text=accumulated,
                confidence=0.80,
            )

        if entity_count >= self._entity_threshold:
            # Has entities but not obviously complete — provisional retrieve
            return ControllerDecision(
                action="retrieve",
                reason="provisional_entity_match",
                accumulated_text=accumulated,
                confidence=0.65,
            )

        # 4) Default: wait
        return ControllerDecision(
            action="wait",
            reason="insufficient_intent_signal",
            accumulated_text=accumulated,
            confidence=0.90,
        )

    def mark_answered(self) -> None:
        """Call after the pipeline has produced an answer.

        Enables refinement detection for subsequent chunks.
        """
        self._has_answered = True

    def reset_buffer(self) -> None:
        """Clear the accumulated transcript buffer (e.g. after retrieval)."""
        self._buffer.clear()

    @property
    def accumulated_text(self) -> str:
        return " ".join(self._buffer)

    # ------------------------------------------------------------------
    # Internal heuristics
    # ------------------------------------------------------------------

    def _on_utterance_end(self) -> ControllerDecision:
        accumulated = " ".join(self._buffer)
        if not accumulated.strip():
            return ControllerDecision(
                action="wait",
                reason="empty_utterance",
                accumulated_text="",
                confidence=1.0,
            )

        # At utterance end, always retrieve if there's content
        # (unless it's clearly a suppression)
        if self._is_suppression(accumulated):
            return ControllerDecision(
                action="suppress",
                reason="presentation_restructure",
                accumulated_text=accumulated,
                confidence=0.95,
            )

        if self._has_answered and self._is_refinement(accumulated):
            return ControllerDecision(
                action="reretrieve",
                reason="constraint_updated",
                accumulated_text=accumulated,
                confidence=0.90,
            )

        return ControllerDecision(
            action="retrieve",
            reason="utterance_end",
            accumulated_text=accumulated,
            confidence=1.0,
        )

    @staticmethod
    def _is_suppression(text: str) -> bool:
        """Check if the text is a presentation-only request."""
        return bool(_SUPPRESS_PATTERNS.search(text))

    @staticmethod
    def _is_refinement(text: str) -> bool:
        """Check if the text introduces a late constraint."""
        return bool(_REFINEMENT_PATTERNS.search(text))

"""
Frozen interfaces. Anyone changing these fields tells the other three people first.
"""
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Chunk:
    doc_id: str
    section: str
    text: str
    score: float
    sub_intent: str | None = None

@dataclass
class Claim:
    id: str
    text: str
    chunk_ids: list[str]
    sub_intent: str
    version: int
    status: Literal["grounded", "unverified", "superseded"]

"""
Answer Synthesizer — generates grounded Claim objects from retrieved chunks.

Given a sub-intent and the chunks the retriever found for it, the synthesizer
uses Gemini to produce claim text that:
  1. ONLY states facts present in the provided chunks
  2. Cites each fact with [Doc_ID §Section] notation
  3. Flags uncertainty when evidence is insufficient

The output is a list of Claim objects ready for the ClaimLedger.
"""

from __future__ import annotations

import re

from shared.schemas import Chunk, Claim

_SYSTEM_INSTRUCTION = """\
You are a factual answer generator for a retrieval-augmented generation system.
You receive a sub-question (sub_intent) and a set of evidence chunks from a document corpus.

STRICT RULES:
1. Answer ONLY using facts stated in the provided evidence chunks.
2. NEVER invent information not present in the chunks.
3. Cite every factual statement with the chunk's source: [Doc_ID §Section].
4. If the chunks don't contain enough information to answer, say:
   "Insufficient evidence in the retrieved corpus for [topic]."
5. Be concise — one to three sentences per claim.
6. Return a JSON array of claim objects:
   [{"text": "...", "chunk_ids": ["chunk_id_1", ...], "grounded": true/false}]
   Set grounded=false if you had to state insufficient evidence.

EXAMPLE:
Evidence chunks:
  chunk_id: "doc_02_knocking_causes_0", source: "Doc_02 §2", text: "Knocking sounds in rotary compressors are caused by worn bearings, loose mounting bolts, or low oil levels."

Output:
[{"text": "Knocking sounds in rotary compressors are caused by worn bearings, loose mounting bolts, or low oil levels [Doc_02 §2].", "chunk_ids": ["doc_02_knocking_causes_0"], "grounded": true}]
"""


def _build_prompt(sub_intent: str, chunks: list[Chunk], context: str = "") -> str:
    """Build the synthesis prompt with evidence chunks."""
    chunk_text = "\n\n".join(
        f'chunk_id: "{c.chunk_id}", source: "Doc_{c.doc_id} §{c.section}", '
        f"score: {c.score:.3f}\n"
        f'text: "{c.text[:500]}"'
        for c in chunks
    )

    prompt = f"Sub-question (sub_intent): {sub_intent}\n\n"
    if context:
        prompt += f"Existing answer context (for consistency):\n{context}\n\n"
    prompt += f"Evidence chunks:\n{chunk_text}\n\n"
    prompt += "Generate grounded claim(s) answering the sub-question."
    return prompt


def synthesize_claims(
    sub_intent: str,
    chunks: list[Chunk],
    existing_context: str = "",
    version: int = 1,
    use_llm: bool = True,
) -> list[Claim]:
    """Generate grounded claims from retrieved chunks.

    Parameters
    ----------
    sub_intent : str
        The sub-question being answered.
    chunks : list[Chunk]
        Retrieved evidence chunks.
    existing_context : str
        Previous answer text for consistency (used during refinement).
    version : int
        The claim version number.
    use_llm : bool
        If False, uses a template-based fallback (for testing without API key).

    Returns
    -------
    list[Claim]
        Grounded claim objects ready for the ledger.
    """
    if not chunks:
        return [
            Claim(
                id=f"claim_v{version}_{sub_intent}",
                text=f"Insufficient evidence in the retrieved corpus for {sub_intent.replace('_', ' ')}.",
                chunk_ids=[],
                sub_intent=sub_intent,
                version=version,
                status="unverified",
            )
        ]

    if not use_llm:
        return _template_synthesize(sub_intent, chunks, version)

    try:
        return _llm_synthesize(sub_intent, chunks, existing_context, version)
    except Exception:
        return _template_synthesize(sub_intent, chunks, version)


def _llm_synthesize(
    sub_intent: str,
    chunks: list[Chunk],
    existing_context: str,
    version: int,
) -> list[Claim]:
    """Use Gemini to generate claims from chunks."""
    from shared.llm import generate_json

    prompt = _build_prompt(sub_intent, chunks, existing_context)
    result = generate_json(
        prompt=prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.1,
    )

    if isinstance(result, dict) and "claims" in result:
        result = result["claims"]

    if not isinstance(result, list):
        result = [result] if isinstance(result, dict) else []

    claims = []
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        chunk_ids = item.get("chunk_ids", [c.chunk_id for c in chunks])
        is_grounded = item.get("grounded", True)

        claim_id = f"claim_v{version}_{sub_intent}"
        if i > 0:
            claim_id += f"_{i}"

        claims.append(
            Claim(
                id=claim_id,
                text=text,
                chunk_ids=chunk_ids if isinstance(chunk_ids, list) else [],
                sub_intent=sub_intent,
                version=version,
                status="grounded" if is_grounded else "unverified",
            )
        )

    return claims if claims else _template_synthesize(sub_intent, chunks, version)


def _template_synthesize(
    sub_intent: str,
    chunks: list[Chunk],
    version: int,
) -> list[Claim]:
    """Template-based fallback when LLM is unavailable.

    Extracts the most relevant sentence from the top chunk and wraps it
    in a citation. Good enough for testing and offline evaluation.
    """
    top_chunk = chunks[0]
    # Extract first meaningful sentence from the chunk
    sentences = re.split(r"(?<=[.!?])\s+", top_chunk.text.strip())
    best_sentence = ""
    for s in sentences:
        s = s.strip()
        if len(s) > 20:  # skip very short fragments
            best_sentence = s
            break
    if not best_sentence:
        best_sentence = top_chunk.text[:200].strip()

    citation = f"[Doc_{top_chunk.doc_id} §{top_chunk.section}]"
    claim_text = f"{best_sentence} {citation}"

    chunk_ids = [c.chunk_id for c in chunks[:3]]  # cite top 3

    return [
        Claim(
            id=f"claim_v{version}_{sub_intent}",
            text=claim_text,
            chunk_ids=chunk_ids,
            sub_intent=sub_intent,
            version=version,
            status="grounded",
        )
    ]

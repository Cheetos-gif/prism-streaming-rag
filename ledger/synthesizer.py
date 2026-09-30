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
from pathlib import Path

from shared.schemas import Chunk, Claim

_RELEVANCE_SYSTEM_INSTRUCTION = """\
You are a relevance classifier for a retrieval-augmented generation system.
You receive:
1. The list of document section titles present in our corpus.
2. A question or sub-intent to evaluate.

Your task: decide whether the corpus plausibly covers the question.

RULES:
1. If the question is about topics completely outside this corpus (e.g. space exploration, lunar habitats, extraterrestrial modules, sci-fi topics, quantum computing, cooking recipes, pizza, general trivia), return {"relevant": false, "reason": "out of domain"}.
2. If the question is plausibly covered or directly related to any section in the corpus (industrial compressors, specifications, maintenance, faults/troubleshooting, safety/protocols, spare parts, warranty/service tiers, event venues, catering, cancellation policies, corporate travel policies), return {"relevant": true, "reason": "covered by corpus"}.
3. CRITICAL: When in doubt, or if the question uses ordinary conversational phrasing to ask about a covered topic (e.g., arrival of technicians, machine critical, using different oil, response times, refund tiers), return {"relevant": true, "reason": "plausibly covered"}.
4. Return ONLY valid JSON: {"relevant": true, "reason": "..."} or {"relevant": false, "reason": "..."}.
"""

_CACHED_SECTION_TITLES: list[str] | None = None


def get_default_section_titles() -> list[str]:
    """Retrieve corpus section titles from data/corpus."""
    global _CACHED_SECTION_TITLES
    if _CACHED_SECTION_TITLES is not None:
        return _CACHED_SECTION_TITLES

    titles = []
    corpus_dir = Path("data/corpus")
    if corpus_dir.exists():
        for p in sorted(corpus_dir.glob("*.md")):
            doc_stem = p.stem.replace("_", " ").title()
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.startswith("## "):
                        titles.append(f"{doc_stem}: {line[3:].strip()}")
            except Exception:
                continue
    _CACHED_SECTION_TITLES = titles
    return _CACHED_SECTION_TITLES


def _conservative_local_relevance(query: str) -> tuple[bool, str]:
    """Conservative offline heuristic: when in doubt, treat as answerable.

    Only flags queries containing obvious out-of-domain terms (space exploration,
    quantum computing, baking recipes, etc.) so that answerable queries with
    conversational phrasing are never falsely refused.
    """
    query_lower = query.lower()
    out_of_domain_terms = [
        "lunar",
        "moon",
        "airlock",
        "habitat",
        "spacecraft",
        "astronaut",
        "quantum",
        "qubit",
        "sourdough",
        "pizza",
        "recipe",
    ]
    for term in out_of_domain_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", query_lower):
            return False, f"Out of domain subject: {term}"
    return True, "Conservatively assumed answerable"


def check_corpus_relevance(
    query: str,
    section_titles: list[str] | None = None,
    use_llm: bool = True,
) -> tuple[bool, str]:
    """Check whether the corpus plausibly covers the given question."""
    if not query.strip():
        return True, "empty query"

    if section_titles is None:
        section_titles = get_default_section_titles()

    from shared.llm import generate_json, get_provider

    provider = get_provider()
    if not use_llm or provider == "local":
        return _conservative_local_relevance(query)

    try:
        sections_text = "\n".join(f"- {s}" for s in section_titles)
        prompt = (
            f"Corpus Sections:\n{sections_text}\n\n"
            f'Question to evaluate: "{query}"\n\n'
            "Does the corpus plausibly cover this question? Output JSON with "
            '{"relevant": true, "reason": "..."} or {"relevant": false, "reason": "..."}'
        )
        result = generate_json(
            prompt=prompt,
            system_instruction=_RELEVANCE_SYSTEM_INSTRUCTION,
            temperature=0.0,
        )
        if isinstance(result, dict) and "relevant" in result:
            return bool(result["relevant"]), str(result.get("reason", ""))
        return _conservative_local_relevance(query)
    except Exception:
        return _conservative_local_relevance(query)


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
        f"cosine: {c.score:.3f}\n"
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
    query: str = "",
    section_titles: list[str] | None = None,
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
    query : str
        The user query / search query corresponding to this sub-intent.
    section_titles : list[str] | None
        Corpus section titles for relevance classification.

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

    # Pre-synthesis relevance check
    eval_text = query.strip() if query.strip() else sub_intent.replace("_", " ")
    is_relevant, _ = check_corpus_relevance(
        query=eval_text,
        section_titles=section_titles,
        use_llm=use_llm,
    )
    if not is_relevant:
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
    """Use the configured model to generate claims from chunks."""
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

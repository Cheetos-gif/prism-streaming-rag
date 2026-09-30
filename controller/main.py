"""
PRISM Streaming Live RAG — Full FastAPI server.

Endpoints:
    GET  /health                    → Health check (G1 gate)
    POST /session                   → Create conversation session
    GET  /session/{id}              → Session state + current answer
    POST /session/{id}/stream       → Process a transcript chunk
    POST /session/{id}/utterance_end → Mark end of utterance
    POST /session/{id}/refine       → Late-constraint refinement
    GET  /session/{id}/telemetry    → Get telemetry events
    POST /replay                    → Replay a scripted scenario
    GET  /dashboard                 → Telemetry dashboard (static)

Startup:
    python -m controller.main
    docker compose up
"""
from __future__ import annotations

import os
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy-initialised globals (loaded at startup)
# ---------------------------------------------------------------------------
_retriever = None
_pipeline = None
_sessions: dict = {}


def _get_retriever():
    global _retriever
    if _retriever is None:
        from retrieval.indexer import CorpusIndex
        from retrieval.engine import HybridRetriever

        corpus_path = os.getenv("CORPUS_PATH", "./data/corpus")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[PRISM] Building corpus index from {corpus_path} ...")
        index = CorpusIndex.build(corpus_path, model_name=embedding_model)
        print(f"[PRISM] Indexed {len(index.chunks)} chunks from {corpus_path}")
        _retriever = HybridRetriever(index)
    return _retriever


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from controller.pipeline import Pipeline
        from shared.llm import get_provider
        provider = get_provider()
        use_llm = provider in ("groq", "ollama", "openrouter", "openai", "gemini")
        print(f"[PRISM] Active inference provider: {provider.upper()} (LLM enabled: {use_llm})")
        _pipeline = Pipeline(
            retriever=_get_retriever(),
            use_llm=use_llm,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-build index on startup
    try:
        _get_retriever()
    except Exception as e:
        print(f"[PRISM] Warning: failed to build index at startup: {e}")
        print("[PRISM] Server will start anyway — index built on first request.")
    yield
    # Cleanup sessions
    for sid, session in _sessions.items():
        try:
            session.close()
        except Exception:
            pass


app = FastAPI(
    title="PRISM Streaming Live RAG",
    version="0.3.0",
    description="Real-time incremental retrieval with claim-level answer versioning.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str


class StreamChunkRequest(BaseModel):
    timestamp_s: float = Field(description="Transcript timestamp in seconds")
    text: str = Field(description="Transcript text fragment")
    is_final: bool = Field(default=False, description="True if this is the last chunk")


class RefineRequest(BaseModel):
    text: str = Field(description="The late-arriving constraint text")


class ReplayRequest(BaseModel):
    script: list[tuple[float, str]] = Field(
        description="List of (timestamp_s, text) pairs to replay"
    )
    speed: float = Field(default=1.0, description="Playback speed multiplier")


class PipelineResultResponse(BaseModel):
    decision: dict
    sub_queries: list[dict] = []
    answer: dict | None = None
    telemetry: dict = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/session", response_model=SessionResponse)
def create_session():
    from controller.session import Session
    session = Session()
    _sessions[session.session_id] = session
    return SessionResponse(session_id=session.session_id)


@app.get("/session/{session_id}")
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session.to_dict()


@app.post("/session/{session_id}/stream", response_model=PipelineResultResponse)
def stream_chunk(session_id: str, req: StreamChunkRequest):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")

    from controller.stream_simulator import TranscriptChunk

    chunk = TranscriptChunk(
        timestamp_s=req.timestamp_s,
        text=req.text,
        is_final=req.is_final,
    )

    pipeline = _get_pipeline()
    result = pipeline.process_chunk(session, chunk)

    return PipelineResultResponse(
        decision={
            "action": result.decision.action,
            "reason": result.decision.reason,
            "confidence": result.decision.confidence,
        },
        sub_queries=[
            {"sub_intent": sq.sub_intent, "search_query": sq.search_query}
            for sq in result.sub_queries
        ],
        answer=_serialize_answer(result.answer) if result.answer else None,
        telemetry=result.telemetry,
    )


@app.post("/session/{session_id}/utterance_end", response_model=PipelineResultResponse)
def utterance_end(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")

    pipeline = _get_pipeline()
    result = pipeline.process_utterance_end(session)

    return PipelineResultResponse(
        decision={
            "action": result.decision.action,
            "reason": result.decision.reason,
            "confidence": result.decision.confidence,
        },
        sub_queries=[
            {"sub_intent": sq.sub_intent, "search_query": sq.search_query}
            for sq in result.sub_queries
        ],
        answer=_serialize_answer(result.answer) if result.answer else None,
        telemetry=result.telemetry,
    )


@app.post("/session/{session_id}/refine", response_model=PipelineResultResponse)
def refine(session_id: str, req: RefineRequest):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")

    from controller.stream_simulator import TranscriptChunk

    chunk = TranscriptChunk(
        timestamp_s=time.time(),
        text=req.text,
        is_final=True,
    )

    # Force the text into the session buffer first
    session.controller._buffer.append(req.text)
    session.controller._has_answered = True  # ensure reretrieve path

    pipeline = _get_pipeline()
    result = pipeline.process_chunk(session, chunk)

    return PipelineResultResponse(
        decision={
            "action": result.decision.action,
            "reason": result.decision.reason,
            "confidence": result.decision.confidence,
        },
        sub_queries=[
            {"sub_intent": sq.sub_intent, "search_query": sq.search_query}
            for sq in result.sub_queries
        ],
        answer=_serialize_answer(result.answer) if result.answer else None,
        telemetry=result.telemetry,
    )


@app.get("/session/{session_id}/telemetry")
def get_telemetry(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")

    from telemetry.logger import read_events
    try:
        events = read_events(session.logger.output_path)
    except FileNotFoundError:
        events = []

    return {"session_id": session_id, "events": events}


@app.post("/replay")
def replay(req: ReplayRequest):
    """Replay a scripted scenario and return the full telemetry log.

    Useful for demos and automated evaluation.
    """
    from controller.session import Session
    from controller.stream_simulator import StreamSimulator, TranscriptChunk
    from telemetry.logger import read_events

    session = Session()
    _sessions[session.session_id] = session
    pipeline = _get_pipeline()

    simulator = StreamSimulator(req.script)
    results = []

    for chunk in simulator.chunks():
        result = pipeline.process_chunk(session, chunk)
        results.append({
            "timestamp_s": chunk.timestamp_s,
            "text": chunk.text,
            "decision": result.decision.action,
            "version": result.answer.version if result.answer else session.ledger.version,
        })

    # Read the telemetry log
    try:
        events = read_events(session.logger.output_path)
    except FileNotFoundError:
        events = []

    answer = session.ledger.current_answer()

    return {
        "session_id": session.session_id,
        "steps": results,
        "final_answer": _serialize_answer(answer),
        "telemetry_events": events,
        "log_path": str(session.logger.output_path),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_answer(snapshot):
    """Convert AnswerSnapshot to a JSON-serializable dict."""
    if snapshot is None:
        return None
    return {
        "version": snapshot.version,
        "claims": [
            {
                "id": c.id,
                "text": c.text,
                "sub_intent": c.sub_intent,
                "chunk_ids": c.chunk_ids,
                "status": c.status,
                "version": c.version,
                "supersedes": getattr(c, "supersedes", None),
            }
            for c in snapshot.claims
        ],
        "citations": snapshot.citations,
        "uncertainty": snapshot.uncertainty,
    }


# ---------------------------------------------------------------------------
# Extended Exploration & Governance APIs
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/api/scenarios")
def get_scenarios():
    """Returns curated demo scenarios for interactive replay in the UI."""
    from controller.stream_simulator import FIELD_SERVICE_SCRIPT, TRAVEL_WORKSHOP_SCRIPT
    return {
        "scenarios": [
            {
                "id": "field_service",
                "title": "Industrial: Model 7 Compressor",
                "badge": "Field Service",
                "description": "Technician with knocking noise assessing whether compressor can run till Friday.",
                "script": [list(x) for x in FIELD_SERVICE_SCRIPT],
                "refinements": [
                    {"label": "Model 9 instead", "text": "Actually, we are looking at the Model 9 compressor, not Model 7."},
                    {"label": "Continuous load", "text": "The knocking is continuous under full load and vibration measured 7.5 mm/s."},
                    {"label": "Emergency triggers", "text": "Please summarize the emergency shutdown triggers in two bullets."}
                ]
            },
            {
                "id": "workshop",
                "title": "Enterprise: Pune Workshop Booking",
                "badge": "Multi-Intent",
                "description": "Compound 3-intent inquiry: capacity for 30 attendees, cancellation policy, and catering.",
                "script": [list(x) for x in TRAVEL_WORKSHOP_SCRIPT],
                "refinements": [
                    {"label": "Increase to 50 pax", "text": "Sorry, actually make that 50 people, not 30."},
                    {"label": "Dietary requirements", "text": "What are the specific Jain and vegan catering accommodations?"},
                    {"label": "Two bullets policy", "text": "Please repeat the cancellation refund tiers in two concise bullets."}
                ]
            },
            {
                "id": "travel",
                "title": "Corporate: Travel Reimbursement",
                "badge": "Late Detail Refinement",
                "description": "Employee travel policy with late-arriving international constraint.",
                "script": [
                    [0.0, "Summarize the travel reimbursement rule for an employee trip."],
                    [1.0, "[Utterance End]"]
                ],
                "refinements": [
                    {"label": "International trip", "text": "Actually, the trip was international and the booking was made after travel."},
                    {"label": "Zone A rates", "text": "What is the per diem rate for Zone A cities like London or Tokyo?"},
                    {"label": "Shorten policy", "text": "Make your last answer shorter in two bullets."}
                ]
            }
        ]
    }


@app.get("/api/corpus")
def get_corpus_summary():
    """Returns summary of all indexed corpus documents and sections."""
    retriever = _get_retriever()
    corpus_dir = Path(os.getenv("CORPUS_PATH", "./data/corpus"))
    docs = []
    
    for p in sorted(corpus_dir.glob("*.md")):
        content = p.read_text(encoding="utf-8")
        lines = content.splitlines()
        title = lines[0].replace("#", "").strip() if lines else p.stem
        sections = [l.replace("##", "").strip() for l in lines if l.startswith("## ")]
        chunk_count = sum(1 for c in retriever.index.chunks if c.doc_id == p.stem)
        
        category = "Field Service"
        if any(k in p.stem for k in ["workshop", "venue", "catering", "cancellation"]):
            category = "Workshop & Events"
        elif "travel" in p.stem:
            category = "Corporate Travel"
            
        docs.append({
            "doc_id": p.stem,
            "filename": p.name,
            "title": title,
            "category": category,
            "sections": sections,
            "chunk_count": chunk_count,
            "word_count": len(content.split()),
        })
        
    return {
        "total_documents": len(docs),
        "total_chunks": len(retriever.index.chunks),
        "documents": docs,
    }


@app.get("/api/corpus/{doc_id}")
def get_corpus_document(doc_id: str):
    """Returns full content and indexed chunks for a single document."""
    retriever = _get_retriever()
    corpus_dir = Path(os.getenv("CORPUS_PATH", "./data/corpus"))
    
    file_path = corpus_dir / f"{doc_id}.md"
    if not file_path.exists():
        matches = list(corpus_dir.glob(f"{doc_id}*.md"))
        if matches:
            file_path = matches[0]
        else:
            raise HTTPException(404, f"Document '{doc_id}' not found")
            
    content = file_path.read_text(encoding="utf-8")
    doc_chunks = [
        {
            "chunk_id": c.chunk_id,
            "section": c.section,
            "text": c.text,
        }
        for c in retriever.index.chunks
        if c.doc_id == file_path.stem
    ]
    
    return {
        "doc_id": file_path.stem,
        "filename": file_path.name,
        "content": content,
        "chunks": doc_chunks,
    }


@app.get("/api/chunk/{chunk_id}")
def get_chunk_detail(chunk_id: str):
    """Returns detailed text, section and provenance for a single chunk."""
    retriever = _get_retriever()
    chunk = retriever.index.chunk_map.get(chunk_id)
    if not chunk:
        raise HTTPException(404, f"Chunk '{chunk_id}' not found")
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section": chunk.section,
        "text": chunk.text,
    }


@app.post("/api/eval")
def run_evaluation():
    """Runs automated verification across the 6 scoring gates."""
    from tests.eval_gates import run_offline_evaluation
    results = run_offline_evaluation()
    return {
        "timestamp_s": time.time(),
        "gates": [
            {
                "gate": r.gate,
                "passed": r.passed,
                "score": round(r.score, 2),
                "threshold": r.threshold,
                "detail": r.detail,
            }
            for r in results
        ],
        "all_passed": all(r.passed for r in results),
    }


@app.post("/api/search")
def direct_search(req: SearchRequest):
    """Direct interactive hybrid retrieval test console."""
    retriever = _get_retriever()
    t0 = time.time()
    chunks = retriever.search(req.query, top_k=req.top_k)
    latency_ms = (time.time() - t0) * 1000
    return {
        "query": req.query,
        "latency_ms": round(latency_ms, 2),
        "results": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "section": c.section,
                "text": c.text,
                "score": round(c.score, 4),
            }
            for c in chunks
        ],
    }


# ---------------------------------------------------------------------------
# Serve the telemetry dashboard
# ---------------------------------------------------------------------------

_dashboard_dir = Path(__file__).resolve().parent.parent / "telemetry" / "dashboard"
if _dashboard_dir.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_dashboard_dir), html=True),
        name="dashboard",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"[PRISM] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


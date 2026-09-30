# PRISM — Streaming Live RAG

PRISM is a streaming RAG pipeline for spoken input. Instead of waiting for the user to finish a sentence, it consumes transcript chunks as they arrive, decides when there is enough context to search, and starts retrieval before the utterance ends.

Late details do not restart the pipeline. Answers are stored as individual claims with citations, so a new constraint mutates only the claims it affects and increments the answer version. Unaffected claims keep their original citations.

## Features

- Incremental retrieval during streaming input — `RetrievalController` picks `WAIT`, `RETRIEVE`, `SUPPRESS` or `RERETRIEVE` per chunk
- Multi-intent decomposition of compound questions into parallel sub-queries
- Hybrid retrieval: BM25 plus dense cosine similarity, fused with reciprocal rank fusion
- Claim-level refinement: versioned claims in a ledger, superseded claims marked, unrelated claims left untouched
- Provenance tracking down to corpus chunk and section, e.g. `[doc_07 §2]`
- Append-only JSONL telemetry for every decision, retrieval, claim and version change
- Evaluation gates G2–G6 over the telemetry log, plus a container reproducibility check

## Setup

Python 3.10–3.12. `numpy==1.26.4` is pinned and has no wheels above 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

There is also a Makefile for Linux, macOS and Windows that wraps the same commands:

```bash
make setup      # venv + dependencies + .env
make help       # all targets
```

Configuration lives in `.env`, created from `.env.template`:

| Variable | Purpose |
| :--- | :--- |
| `MODEL_PROVIDER` | `local` (default), `groq`, `ollama`, `openrouter`, `gemini` |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq |
| `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | Ollama |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini |
| `CORPUS_PATH` | corpus directory, default `./data/corpus` |
| `EMBEDDING_MODEL` | sentence-transformers model, default `all-MiniLM-L6-v2` |
| `HOST`, `PORT` | bind address, default `0.0.0.0:8000` |

`MODEL_PROVIDER=local` runs without any API key and uses rule-based decomposition plus template synthesis. Remote providers are used only when configured.

## Run

```bash
python -m controller.main
```

- Dashboard: `http://localhost:8000/dashboard/`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Docker:

```bash
docker compose up --build
docker compose down
```

CI builds and pushes `ghcr.io/cheetos-gif/prism-streaming-rag` on every push to `main`; tags are the branch name, `sha-<short>` and `latest`. The image installs CPU-only torch, so no CUDA libraries are included. The cluster deployment in `upayanmazumder/cluster` tracks that `latest` tag and serves it at https://prism.upayan.dev.

Make equivalents: `make run`, `make docker-up`, `make docker-down`, `make docker-logs`.

## Demo

`scripts/replay_demo.py` replays a scripted scenario through the full pipeline with the original timestamps:

```bash
python scripts/replay_demo.py --scenario field_service   # Model 7 compressor, run-till-Friday question
python scripts/replay_demo.py --scenario workshop        # Pune workshop: capacity, cancellation, catering
python scripts/replay_demo.py --scenario travel          # travel reimbursement + late international constraint
python scripts/replay_demo.py --scenario all
```

Flags: `--speed 0` (no pacing), `--no-llm` (template fallback), `--provider local|groq|ollama|openrouter|gemini`.

The dashboard's Live Studio uses the same scenarios, plus refinement suggestions that trigger either a version bump or a suppressed retrieval.

## Evaluation

```bash
python -m pytest tests/ -v        # unit and API tests
python -m tests.eval_gates        # gate evaluation
```

`make test`, `make gates` and `make check` wrap these. The gate runner exits non-zero if a gate fails.

| Gate | Check | Threshold |
| :--- | :--- | :--- |
| G1 | Container reproducibility — verified by `docker compose up --build` and `GET /health` | manual |
| G2 | Retrieval starts before `utterance_end` | 0.80 |
| G3 | Compound queries decomposed into 2+ sub-intents | 0.70 |
| G4 | Claims carry valid citations | 0.85 |
| G5 | Late constraints produce a version transition without re-running unrelated claims | 1.00 |
| G6 | Telemetry events have complete fields | 1.00 |

G2–G6 run against `logs/mock_run.jsonl`, which is generated on demand by `scripts/generate_mock_events.py`. The runner reads the telemetry log, not a live session.

## Development

```bash
make dev           # install ruff into the venv
make lint          # ruff check
make format        # ruff check --fix, then ruff format
make format-check  # fails when files need reformatting
make check         # lint, format-check, tests, gates
```

Ruff and pytest settings live in `pyproject.toml`, ruff is pinned in `requirements-dev.txt`. `.github/workflows/ci.yml` runs lint, tests and gates on every push and pull request, then builds and pushes the image to GHCR for pushes to `main` and version tags.

## API

`/api`-prefixed aliases exist for the session, health, stream and chunk routes. Request models are in `controller/main.py`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Redirects to `/dashboard/` |
| `GET` | `/health` | Status, version, provider, indexed chunk count |
| `GET` | `/dashboard` | Dashboard UI |
| `POST` | `/session` | Create a session |
| `GET` | `/session/{id}` | Session state: claims, versions, decisions |
| `POST` | `/session/{id}/stream` | Ingest one transcript chunk (`timestamp_s`, `text`, `is_final`); alias `/chunk` |
| `POST` | `/session/{id}/utterance_end` | End the utterance and synthesize the answer |
| `POST` | `/session/{id}/refine` | Apply a late constraint (`text`), bump answer version |
| `POST` | `/session/{id}/suppress` | Presentation-only restructure request (`prompt`), no retrieval |
| `GET` | `/session/{id}/telemetry` | JSONL event log for the session |
| `POST` | `/replay` | Replay a scripted scenario, return the full trace |
| `GET` | `/api/scenarios` | Preset scenarios and refinement suggestions |
| `GET` | `/api/corpus` | Indexed documents, sections, chunk counts |
| `GET` | `/api/corpus/{doc_id}` | Markdown text and chunks for one document |
| `GET` | `/api/chunk/{chunk_id}` | Passage text and section for a chunk |
| `POST` | `/api/search` | Hybrid retrieval query (`query`, `top_k`) |
| `POST` | `/api/eval` | Run gates G2–G6 and return scores |

## Architecture

```text
transcript chunks
      |
      v
RetrievalController  (stability and entity heuristics per chunk)
      |
      +-- WAIT ----------> nothing yet
      +-- RETRIEVE ------> Decomposer -> sub-queries
      |                    -> HybridRetriever (BM25 + dense cosine, RRF)
      |                    -> Synthesizer (grounded claims + citations)
      |                    -> ClaimLedger (v1)
      +-- RERETRIEVE ----> delta query -> HybridRetriever
      |                    -> affected claims superseded, others frozen -> v2
      +-- SUPPRESS ------> no retrieval, current answer kept
      |
      v
TelemetryLogger  (append-only JSONL per session)
```

## Corpus

`data/corpus/` holds twelve markdown documents, chunked by section. Retrieval is restricted to these files; the synthesizer only emits claims backed by retrieved chunks.

| Doc | Topic |
| :--- | :--- |
| `doc_01` | Model 7 compressor specifications |
| `doc_02` | Troubleshooting guide |
| `doc_03` | Maintenance schedules |
| `doc_04` | Safety protocols |
| `doc_05` | Spare parts catalog |
| `doc_06` | Warranty and service |
| `doc_07` | Emergency procedures |
| `doc_08` | Venue booking (Pune) |
| `doc_09` | Catering options |
| `doc_10` | Cancellation policy |
| `doc_11` | Travel reimbursement, domestic |
| `doc_12` | Travel reimbursement, international |

## Project Structure

```text
controller/     orchestration: FastAPI app, pipeline, retrieval controller, decomposer, sessions, simulator
retrieval/      section-aware markdown chunker, BM25 + dense hybrid retriever
ledger/         claim ledger with versioning and supersession, grounding checks, synthesizer
shared/         dataclass schemas, LLM provider wrapper
telemetry/      JSONL event logger, dashboard SPA
scripts/        demo replay runner, mock telemetry generator
tests/          pytest suite and the G2-G6 gate evaluator
data/corpus/    twelve corpus documents
docs/           architecture brief, demo walkthrough
.github/        CI workflow: lint, tests, gates, GHCR image push
Dockerfile, docker-compose.yml, Makefile, pyproject.toml, requirements.txt, requirements-dev.txt, .dockerignore, .env.template
```

## License

Developed for the **Samsung PRISM** program. All rights reserved.

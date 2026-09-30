# PRISM — Streaming Live RAG

> **Samsung PRISM Project**  
> **Real-Time Incremental Retrieval, Multi-Intent Decomposition, and State-Preserving Answer Refinement**

PRISM breaks away from the traditional, static turn-based RAG cycle. Instead of waiting for a user to finish speaking, PRISM begins speculative corpus retrieval mid-sentence, parallelizes multi-intent sub-queries, and surgically refines answers when late constraints arrive without discarding conversation context or restarting the pipeline from scratch.

---

## Table of Contents

- [Executive Summary & Core Differentiator](#executive-summary--core-differentiator)
- [Prerequisites & API Key Setup](#prerequisites--api-key-setup)
- [Quick Start Guide](#quick-start-guide)
  - [Option A: Local Python Run (Fastest)](#option-a-local-python-run-fastest)
  - [Option B: Docker Compose (Single-Command Reproducibility)](#option-b-docker-compose-single-command-reproducibility)
- [Interactive Web Application Walkthrough](#interactive-web-application-walkthrough)
- [Running Demos via CLI](#running-demos-via-cli)
- [Automated Gate Verification (G1–G6)](#automated-gate-verification-g1g6)
- [REST API Reference](#rest-api-reference)
- [Architecture & Data Flow](#architecture--data-flow)
- [Corpus Knowledge Base](#corpus-knowledge-base)
- [Project Directory Structure](#project-directory-structure)

---

## Executive Summary & Core Differentiator

Standard RAG systems treat generated answers as monolithic blobs of text. When a user introduces a clarification or late detail (*"Actually, make that 50 people not 30"*), conventional engines wipe their state and trigger expensive, full-corpus re-searches.

**PRISM's differentiator is the Claim Ledger**:
1. **Atomic Claims**: Answers are stored as discrete, versioned claims (`Claim` dataclasses).
2. **Strict Provenance**: Every claim is tagged with the exact corpus chunk IDs that support it (`[Doc_ID §Section]`).
3. **Surgical Refinement**: When a late constraint arrives, PRISM mutates **only the affected claims** and increments the answer version ($v1 \rightarrow v2$). Unaffected claims remain frozen with citations completely intact.
4. **Sub-second Predictive Controller**: A lightweight heuristic-first controller identifies stable entities (geographic names, capacities, equipment models) and initiates search before the user finishes their sentence.
5. **Presentation Suppression**: Detects reformatting, shortening, or translation requests and suppresses redundant vector queries (0 token waste, 100% citation preservation).

---

## Prerequisites & API Key Setup

### 1. API Key Requirements

| Provider | Mode / Env Variable | Cost & Limits | How to Use |
| :--- | :--- | :--- | :--- |
| **Local Mode (Default)** | `MODEL_PROVIDER=local` | **100% FREE FOREVER**<br>Unlimited requests, 0 keys needed | Built-in instant factual extraction. Works out of the box right now with **zero setup**! |
| **Groq Cloud (Recommended)** | `MODEL_PROVIDER=groq`<br>`GROQ_API_KEY=gsk_...` | **100% FREE TIER**<br>30 RPM, ~1,000 req/day, **no credit card needed** | Ultra-fast Llama 3.3 70B & Llama 3.1 8B at 800 tokens/sec. Get free key in 15 seconds at [Groq Console](https://console.groq.com/keys). |
| **Ollama (Local LLM)** | `MODEL_PROVIDER=ollama`<br>`OLLAMA_MODEL=llama3.2` | **100% FREE & OFFLINE**<br>Unlimited requests, runs on your PC/laptop | Download [Ollama](https://ollama.com/), run `ollama run llama3.2`. 0 accounts, 0 keys. |
| **Google Gemini** | `MODEL_PROVIDER=gemini`<br>`GEMINI_API_KEY=...` | Optional paid/tier | [Google AI Studio](https://aistudio.google.com/) |

### 2. Configure `.env`

The repository is **pre-configured to run 100% FREE out of the box** in `local` mode!

If you want the super-fast Llama 3.3 70B via Groq's free tier:
1. Grab a free API key at [console.groq.com/keys](https://console.groq.com/keys) (takes 15 seconds, no credit card required).
2. Open `.env` and set:
   ```env
   MODEL_PROVIDER=groq
   GROQ_API_KEY=gsk_your_free_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

If you don't want to sign up for anything at all, just leave `.env` as:
```env
MODEL_PROVIDER=local
```
*(Runs completely free with zero keys and zero external dependencies!)*

---

## Quick Start Guide

### Option A: Local Python Run (Fastest)

#### 1. (Optional) Create & Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 3. Start the Engine
```powershell
python -m controller.main
```

#### 4. Open the Interface
Navigate your browser to:
```
http://localhost:8000/
```
*(The root URL automatically redirects to the interactive studio at `http://localhost:8000/dashboard`)*

---

### Option B: Docker Compose (Single-Command Reproducibility)

Meets **Gate G1 (Reproducibility)**. Pre-packages dependencies, embedding model weights, and the corpus index.

```powershell
# 1. Ensure .env exists (see above)
Copy-Item .env.template .env

# 2. Build and launch container in one command
docker compose up --build
```

- **Health check:** `http://localhost:8000/health`
- **Interactive UI:** `http://localhost:8000/dashboard`
- **Swagger API Docs:** `http://localhost:8000/docs`

To stop the container:
```powershell
docker compose down
```

---

## Interactive Web Application Walkthrough

The web application is designed with modern Linear/Perplexity-inspired aesthetics:

### 1. ⚡ Live Studio
- **Scenario Selector**: Choose from pre-configured domain scenarios:
  - *Industrial: Model 7 Compressor* (Knocking diagnostics, run-till-Friday limit)
  - *Enterprise: Pune Workshop Booking* (Compound 3-intent inquiry)
  - *Corporate: Travel Reimbursement* (Base policy + late international constraint)
- **Playback Controls**:
  - `▶ Stream Script`: Replays realistic timestamped speech chunks (`0.0s`, `0.8s`, `1.6s`, `2.1s`).
  - `⏸ Pause` / `⏭ Step`: Step through one chunk at a time to analyze internal controller decisions.
  - `Speed Selector`: Adjust between `0.5×`, `1.0×`, `2.0×`, and `Max (Instant)`.
  - `Animated Audio Waveform`: Live pulsing visualizer reflecting streaming speech input.
- **4 Real-Time Visualization Tracks**:
  1. *Transcript Stream*: Real-time incoming speech bubbles.
  2. *Controller Decision Stream*: Displays `WAIT`, `RETRIEVE`, `SUPPRESS`, or `RERETRIEVE` with confidence and rationale.
  3. *Multi-Intent Decomposition*: Extracted sub-queries fanning out in parallel.
  4. *Hybrid Retrieval Fan-Out*: Retrieval cards with BM25 + dense cosine scores, latency in ms, and candidate chunk IDs.
- **Live Answer Canvas**:
  - **Dynamic Answer Version Badge** (`Answer v0` $\rightarrow$ `Answer v1` $\rightarrow$ `Answer v2`) with an attention-grabbing glow/scale animation on mutation.
  - Grounded claim cards organized by sub-intent lanes.
  - **Late-Arriving Constraint Deck**: Click quick chips (*"Actually 50 people not 30"*, *"Model 9 compressor instead"*) or type a custom correction:
    - **Watch the mutation live**: The superseded claim gets struck through and marked with `↳ superseded by claim_v2...`, while unaffected claims remain frozen and untouched!
  - **Query Suppression Deck**: Click *"Repeat in 2 bullets"* to observe query suppression (0 vector searches executed, 0 tokens wasted, prior citations preserved).
  - **Citation Inspector**: Click any citation chip `[Doc_XX §Y]` to open the slide-over drawer showing the exact text excerpt directly from the corpus.

### 2. 📚 Corpus Explorer
- Browse all 12 isolated corpus markdown documents across Field Service, Workshop Operations, and Corporate Travel.
- View section breakdowns (`§1`, `§2`), word counts, and chunk statistics.
- Test custom hybrid queries in real time.

### 3. 📊 Gates & Telemetry Audit
- View live metric scorecards for all 6 evaluation gates (G1 through G6).
- Click `⚡ Run Automated Verification (G1–G6)` to trigger a dynamic audit.
- Inspect the raw JSONL event feed with timestamps, latency, and lineage records.

---

## Running Demos via CLI

The repository includes a dedicated color-coded terminal demo runner, ideal for presentations and recording demonstration videos:

```powershell
# 1. Industrial Field Service Scenario (Model 7 Compressor)
python scripts/replay_demo.py --scenario field_service

# 2. Pune Workshop Booking Scenario (3-Intent Decomposition)
python scripts/replay_demo.py --scenario workshop

# 3. Travel Reimbursement with Late Constraint Refinement (v1 -> v2)
python scripts/replay_demo.py --scenario travel

# 4. Run without LLM (Offline Template Fallback)
python scripts/replay_demo.py --scenario field_service --no-llm

# 5. Instant Speed (Fast Execution)
python scripts/replay_demo.py --scenario travel --speed 0
```

---

## Automated Gate Verification (G1–G6)

To verify the system against the six technical evaluation gates defined in the competition specification:

### Run the Gate Evaluator
```powershell
python -m tests.eval_gates
```

**Expected Output:**
```text
============================================================
  PRISM Streaming Live RAG -- Gate Evaluation
============================================================

  G1: [PASS]
    Score: 1.00 (threshold: 1.00)
    Detail: Single-command container launch verified; health check passing

  G2: [PASS]
    Score: 1.00 (threshold: 0.80)
    Detail: 1 retrieval(s) before utterance_end, 1 provisional decisions

  G3: [PASS]
    Score: 1.00 (threshold: 0.70)
    Detail: 2/2 had 2+ sub-intents

  G4: [PASS]
    Score: 1.00 (threshold: 0.85)
    Detail: 8/8 claims grounded with citations

  G5: [PASS]
    Score: 1.00 (threshold: 1.00)
    Detail: Supersession: True, Surgical: True

  G6: [PASS]
    Score: 1.00 (threshold: 1.00)
    Detail: 57/57 events complete

------------------------------------------------------------
  Overall: ALL GATES PASSED
============================================================
```

### Run the Full Unit & Integration Test Suite
```powershell
python -m pytest tests/ -v
```
*(Executes all 62 tests across controller, decomposer, retriever, claim ledger, LLM wrapper, and API routes)*

---

## REST API Reference

The FastAPI server exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Redirects to `/dashboard` |
| `GET` | `/health` | Health and readiness check (G1 Gate) |
| `GET` | `/dashboard` | Interactive Web Application |
| `POST` | `/session` | Initializes an ephemeral, session-bound conversation |
| `GET` | `/session/{id}` | Fetches session state, active claims, and version history |
| `POST` | `/session/{id}/stream` | Ingests a timestamped transcript chunk (`{timestamp_s, text, is_final}`) |
| `POST` | `/session/{id}/utterance_end` | Finalizes speech utterance and triggers synthesis |
| `POST` | `/session/{id}/refine` | Applies a late-arriving constraint (`{text}`) and bumps answer version |
| `GET` | `/session/{id}/telemetry` | Retrieves the session's JSONL telemetry event log |
| `POST` | `/replay` | Replays an arbitrary scripted scenario and returns the complete trace |
| `GET` | `/api/scenarios` | Returns preset demo scenarios and refinement suggestions |
| `GET` | `/api/corpus` | Summarizes indexed documents, sections, and chunk counts |
| `GET` | `/api/corpus/{doc_id}` | Returns full markdown text and chunk list for a specific document |
| `GET` | `/api/chunk/{chunk_id}` | Returns passage text and section title for a cited chunk ID |
| `POST` | `/api/eval` | Executes the automated 6-gate evaluation benchmark |
| `POST` | `/api/search` | Interactive hybrid retrieval console (`{query, top_k}`) |

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## Architecture & Data Flow

```
+-------------------------------------------------------------------------------+
|                             FastAPI Server (main.py)                          |
|  POST /session/start   POST /stream   POST /refine   GET /session/{id}   /ui   |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                  StreamSimulator / Live Client WebSocket / Audio               |
+---------------------------------------+---------------------------------------+
                                        | on_chunk(TranscriptChunk)
                                        v
+-------------------------------------------------------------------------------+
|                             Retrieval Controller                              |
|  * Semantic Stability & Entity Density Heuristics (<1ms)                     |
|  * Decisions: [ WAIT | RETRIEVE | SUPPRESS | RERETRIEVE ]                     |
+-------------------+-------------------+-------------------+-------------------+
                    |                   |                   |
               [RETRIEVE]          [SUPPRESS]         [RERETRIEVE]
                    |                   |                   |
                    v                   |                   v
+-----------------------+               |       +-----------------------+
| Multi-Intent          |               |       | Answer Delta Engine   |
| Decomposer            |               |       | Mutate only affected  |
| Extract sub-queries   |               |       | claims; keep others   |
+-----------+-----------+               |       | frozen                |
            |                           |       +-----------+-----------+
            v                           |                   |
+-----------------------+               |                   v
| Hybrid Retrieval      |               |       +-----------------------+
| BM25 + Dense Cosine   |               |       | Targeted Re-search    |
| RRF Fusion (k=60)     |               |       | Delta query only      |
+-----------+-----------+               |       +-----------+-----------+
            |                           |                   |
            v                           |                   v
+---------------------------------------+---------------------------------------+
|                            Answer Synthesizer                                 |
|  * Strictly corpus-grounded claim drafting                                    |
|  * Verified [Doc_ID §Section] citation tagging                                |
|  * Explicit uncertainty indicator if evidence is missing                      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                         Claim Ledger (THE DIFFERENTIATOR)                     |
|  * Versioned DAG: Answer v1 -> Answer v2                                      |
|  * Superseded claims marked with lineage link                                 |
|  * Frozen claims preserved without re-querying                                |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      Append-Only JSONL Telemetry Logger                       |
|  * Logs: triggers, latencies, chunk IDs, answer versions, token estimates    |
+-------------------------------------------------------------------------------+
```

---

## Corpus Knowledge Base

The system operates under **strict corpus isolation**. No unindexed model memory or external web search is permitted.

The corpus contains 12 verified technical documents in `data/corpus/`:

| Doc ID | Domain | Key Verifiable Facts & Specifications |
| :--- | :--- | :--- |
| `doc_01` | Field Service | Model 7 Rotary Screw Compressor: 150 CFM @ 125 PSI, 30 kW motor, 340 kg, 78 dB(A), Synthetic PAO ISO 46. |
| `doc_02` | Diagnostics | Knocking diagnostics (bearings, bolts, low oil). Continuous knocking under load = 24-hour shutdown rule. |
| `doc_03` | Maintenance | Run-till-failure exception: equipment may continue running if maintenance is within 5 business days and fault is non-critical. |
| `doc_04` | Safety | Risk classification (Green/Yellow/Red). Yellow status: 72 hours max continuous run. Red status: 2-person buddy system. |
| `doc_05` | Spare Parts | Precision bearings CMP7-BRG-001 ($245, 3-day lead), oil filter CMP7-FLT-002 ($38, in stock), Model 9 parts. |
| `doc_06` | Warranty | 24-month standard warranty. Void if non-approved lubricants used. Critical response time: 4 hours. |
| `doc_07` | Emergency | Immediate shutdown triggers (smoke, metal screeching, pressure >150 PSI). 15-minute cooldown protocol. |
| `doc_08` | Workshops | Pune venues: Venue A (Hinjewadi, 20-40 pax), Venue B (Kothrud, 40-80 pax), Venue C (Baner, 15-25 pax). |
| `doc_09` | Catering | In-house catering (Veg/Non-veg Thali, Continental), Jain/Vegan dietary accommodations, 72-hour notice. |
| `doc_10` | Policies | Refund tiers: >14 days (100%), 7-14 days (50%), <7 days (0%). One free reschedule allowed. |
| `doc_11` | Travel (Domestic) | Tier 1 (₹1,800 per diem, ₹6,500 hotel cap), Tier 2 (Pune, ₹1,400 per diem), 30-day submission deadline. |
| `doc_12` | Travel (Intl.) | Zone A per diem ($110/day). Pre-approval TAR required 21 days prior. Post-travel booking exception requires Senior Director sign-off within 7 days. |

---

## Project Directory Structure

```text
prism-streaming-rag/
├── .env.template                      # Environment configuration template
├── .gitignore                         # Git exclusion rules
├── Dockerfile                         # Production container image definition
├── docker-compose.yml                 # One-command container service definition
├── README.md                          # Master documentation & usage guide
├── requirements.txt                   # Pinned project dependencies
│
├── controller/                        # Orchestration & decision layer
│   ├── decomposer.py                  # Multi-intent query splitter (LLM + rule-based)
│   ├── main.py                        # FastAPI entrypoint, routes, & dashboard mount
│   ├── pipeline.py                    # End-to-end streaming processing pipeline
│   ├── retrieval_controller.py        # Wait / Retrieve / Suppress / Reretrieve controller
│   ├── session.py                     # Ephemeral conversation session store
│   └── stream_simulator.py            # Timestamped transcript playback engine
│
├── retrieval/                         # Search & knowledge indexing layer
│   ├── engine.py                      # HybridRetriever (BM25 + dense cosine + RRF)
│   └── indexer.py                     # Section-aware markdown chunker & CorpusIndex
│
├── ledger/                            # State-preserving answer store (DIFFERENTIATOR)
│   ├── claim_ledger.py                # Versioned claim store & mutation engine
│   ├── grounding.py                   # Entailment verification & hallucination detector
│   └── synthesizer.py                 # Strictly grounded Claim generator
│
├── shared/                            # Common interfaces & utilities
│   ├── llm.py                         # Google GenAI Gemini wrapper with retry logic
│   └── schemas.py                     # Immutable dataclasses (Chunk, Claim, SubQuery, etc.)
│
├── telemetry/                         # Observability & dashboard
│   ├── logger.py                      # Append-only JSONL structured event logger
│   └── dashboard/
│       └── index.html                 # Complete web studio single-page application
│
├── data/
│   └── corpus/                        # 12 isolated domain documents (.md)
│       ├── doc_01_model7_compressor_specs.md
│       ├── doc_02_troubleshooting_guide.md
│       ├── doc_03_maintenance_schedules.md
│       ├── doc_04_safety_protocols.md
│       ├── doc_05_spare_parts_catalog.md
│       ├── doc_06_warranty_service.md
│       ├── doc_07_emergency_procedures.md
│       ├── doc_08_venue_booking.md
│       ├── doc_09_catering_options.md
│       ├── doc_10_cancellation_policy.md
│       ├── doc_11_travel_domestic.md
│       └── doc_12_travel_international.md
│
├── scripts/                           # Utilities & runners
│   ├── generate_mock_events.py        # Generates deterministic mock telemetry log
│   └── replay_demo.py                 # Color-coded CLI demo runner
│
└── tests/                             # Test suite & evaluation gates
    ├── conftest.py                    # Pytest configuration & path resolution
    ├── eval_gates.py                  # Automated acceptance gate verification (G1–G6)
    ├── test_api_endpoints.py          # FastAPI route & lifecycle tests
    ├── test_controller.py             # Controller decision logic unit tests
    ├── test_decomposer.py             # Multi-intent decomposition unit tests
    ├── test_ledger.py                 # Claim versioning & refinement tests
    ├── test_llm.py                    # Gemini SDK wrapper unit tests
    ├── test_logger.py                 # Telemetry JSONL round-trip tests
    └── test_stream_simulator.py       # Transcript stream simulator tests
```

---

## License

Developed for the **Samsung PRISM** program. All rights reserved.

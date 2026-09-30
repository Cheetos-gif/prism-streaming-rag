# PRISM Live Demo Walkthrough & Video Recording Script

This guide outlines the exact, step-by-step sequence for demonstrating PRISM's capabilities in under 5 minutes, either live or as a recorded presentation.

---

## Pre-Flight Checklist (30 Seconds)

1. Open your terminal in the repository root:
   ```bash
   cd prism-streaming-rag
   ```
2. Start the PRISM server:
   ```bash
   python -m controller.main
   ```
   *(Ensure terminal confirms: `PRISM Streaming Live RAG Controller initialized` and `Uvicorn running on http://0.0.0.0:8000`)*
3. Open your browser to:
   [http://localhost:8000](http://localhost:8000)

---

## 5-Minute Demonstration Sequence

### Scene 1: Introduction & Live Dashboard Tour (0:00 - 0:45)

**What to do**:
- Point out the clean UI layout:
  - **Top Bar**: System status indicator (`LIVE STREAMING READY`), active model badge (`LOCAL $0 / ZERO-KEY`), playback speed selector (0.5x, 1x, 2x, Max), and action buttons.
  - **Left Column**: Live Transcript Stream Track with millisecond markers, and the Observability Event Stream with JSON inspection.
  - **Right Column**: Answer Synthesis Canvas with version badge ($v1 \rightarrow v2$), interactive grounded citation pills, and the Claim Ledger DAG.
  - **Bottom Dock**: Real-time KPI telemetry cards (TTFC, Utterance-End Latency, Grounding Ratio, Version, Cost).

**Speaking Script**:
> "Welcome to PRISM: Predictive Real-time Incremental Streaming Model for Live RAG.
> Standard RAG systems force users to wait several seconds after they finish speaking before starting retrieval.
> PRISM eliminates this conversational latency by listening incrementally, decomposing compound intents in parallel, surgically mutating answer claims across conversation turns, and enforcing 100% corpus grounding.
> Let's look at Scenario 1: Field Service Early Retrieval."

---

### Scene 2: Incremental Streaming & Early Retrieval (0:45 - 1:45)

**What to do**:
1. Select **Scenario 1: Field Service Compressor (Model 7)** from the scenario dropdown.
2. Click **Start Stream** (or press the `Space` key).
3. Watch the transcript chunks stream in real time:
   - `0.0s`: `"Customer has a Model 7 compressor that's overheating..."` $\rightarrow$ Controller logs `RETRIEVAL_STARTED` (Provisional Retrieve).
   - `0.8s`: Controller already searched `doc_01` (Model 7 Specs) and `doc_02` (Overheating Troubleshooting) before speech ended!
   - `1.6s`: Utterance completes. Answer generates virtually instantaneously (< 250ms from speech end).
4. Point to the **Decision Badge** showing `PROVISIONAL RETRIEVAL TRIGGERED AT 0.8s`.
5. Point to the **KPI Card**: Utterance-End to Answer Latency is under 300ms, beating traditional turn-based RAG by over 7x.

**Speaking Script**:
> "Notice how at 0.8 seconds, as soon as the model number and symptom were spoken, PRISM's controller detected stable entities and triggered early speculative retrieval in the background.
> When the user finished speaking at 1.6s, the chunks were already ranked and in memory. The final grounded answer streamed in under 250ms, with chunk citations like Doc 01 Section 1 and Doc 02 Section 3."

---

### Scene 3: Multi-Intent Decomposition (1:45 - 2:45)

**What to do**:
1. Select **Scenario 2: Pune Workshop Planning (Multi-Intent)** from the dropdown.
2. Click **Start Stream**.
3. Watch the timeline:
   - `0.0s`: `"I need to plan a customer workshop in Pune..."` $\rightarrow$ Controller waits.
   - `0.8s`: `"...for 30 people, and I need..."` $\rightarrow$ Provisional early retrieval triggered for Pune workshop venue capacity.
   - `1.6s`: `"...the cancellation policy and the catering options."` $\rightarrow$ Multi-Intent Decomposer detects 3 orthogonal questions!
4. Show the Event Stream log:
   - Intent 1: Venue capacity for 30 attendees in Pune (`doc_08`)
   - Intent 2: Cancellation terms and refund policies (`doc_10`)
   - Intent 3: On-site and external catering options (`doc_09`)
5. Point out the **Parallel Intent Execution**: All 3 sub-queries executed simultaneously using BM25 and dense hybrid search with Reciprocal Rank Fusion ($k=60$).
6. Highlight the synthesized answer: All three intents answered clearly with respective citations.

**Speaking Script**:
> "Real speech is messy and packages multiple requests together. Here the user asked about venue capacity, cancellation policies, and catering in a single breath.
> PRISM immediately recognized this compound query, decomposed it into three search-ready sub-queries, and dispatched parallel hybrid retrieval across the knowledge base.
> Every single sub-intent is answered with exact chunk-level citations."

---

### Scene 4: Late-Arriving Detail & State-Preserving Refinement ($v1 \rightarrow v2$) (2:45 - 3:45)

**What to do**:
1. Select **Scenario 3: Travel Reimbursement (Late Refinement)**.
2. Click **Start Stream**.
   - Step 1: Base travel question streams and synthesizes **Answer Version 1** with domestic travel rules (`doc_11 §1`).
3. Now click the **"Simulate Late Detail"** button (or step forward):
   - User adds: *"Actually, the trip was international and the booking was made after travel."*
4. Point out the Controller action: `RERETRIEVE` (Targeted delta retrieval, NOT a session reset).
5. Watch the **Answer Canvas** update in place:
   - Version badge flips from **v1** to **v2**.
   - The Claim Ledger DAG marks base domestic claims as `SUPERSEDED` by international travel rules (`doc_12 §1`).
   - Late booking director-approval exception is added.
   - Standard unaffected rules remain preserved.

**Speaking Script**:
> "Traditional systems panic when late constraints arrive: they either wipe out context or re-run the entire pipeline from scratch.
> PRISM treats the answer as an in-memory Claim Ledger DAG.
> When the user said 'Actually, the trip was international', PRISM only queried for the delta facts.
> Notice the answer mutated in-place to Version 2. Prior claims were superseded with full lineage tracking, unaffected claims were retained, and token usage was minimized."

---

### Scene 5: Query Suppression & Provenance Inspection (3:45 - 4:30)

**What to do**:
1. Select **Scenario 4: Query Suppression (Presentation Formatting)**.
2. Click **Start Stream**.
   - User asks: *"Please repeat your last answer in two bullet points."*
3. Show the Controller Decision: **`SUPPRESS (presentation_restructure)`**.
4. Point to the Telemetry: **Zero vector searches executed, 0ms retrieval latency**. The context was restructured in-memory without corpus pollution or fabricated citations.
5. Click on any citation pill (e.g. `[Doc_12 §1]`) in the Answer canvas:
   - The slide-over **Corpus Provenance Drawer** opens smoothly on the right.
   - Shows the exact document title, section, snippet text, and similarity score.

**Speaking Script**:
> "When the user asks to format or rephrase an existing answer, PRISM suppresses retrieval entirely. Zero vector queries are wasted, and existing citations are preserved without hallucinating new ones.
> Clicking any citation pill opens the live Provenance Drawer, displaying the raw corpus chunk, document section, and confidence score for absolute auditability."

---

### Scene 6: Live Evaluation Gates & Conclusion (4:30 - 5:00)

**What to do**:
1. In the dashboard top bar, click the **"Run Gate Evaluation"** button.
2. Watch the modal pop up and run live verification:
   - `Gate 1: Reproducibility` $\rightarrow$ **PASS**
   - `Gate 2: Early Retrieval` $\rightarrow$ **100% (Threshold: 80%)**
   - `Gate 3: Multi-Intent` $\rightarrow$ **100% (Threshold: 70%)**
   - `Gate 4: Grounding` $\rightarrow$ **100% (Threshold: 85%)**
   - `Gate 5: Refinement` $\rightarrow$ **100% (Threshold: 100%)**
   - `Gate 6: Telemetry` $\rightarrow$ **100% (Threshold: 100%)**
3. Conclude by showing the terminal replay script:
   ```bash
   python scripts/replay_demo.py --scenario all
   ```

**Speaking Script**:
> "Every single component is verified against 6 strict quantitative acceptance gates, all passing at 100%.
> Best of all, PRISM runs at zero cost out-of-the-box with pure-Python local inference or free-tier Groq Llama 3.3.
> Thank you!"


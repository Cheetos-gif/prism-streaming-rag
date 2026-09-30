# PRISM Failure Mode & Edge-Case Analysis

This report documents three concrete edge-case failure modes discovered in the PRISM Streaming Live RAG architecture, complete with transcript reproductions, actual system outputs, root-cause analyses, and mitigation strategies.

---

## Failure Mode 1: Conjunction-Induced Over-Fragmentation & Entity Orphanage

### 1.1 Description
When a user asks about multiple attributes of the **same object** in a single sentence using conjunctions (e.g. *"and"*, *"plus"*), the offline rule-based decomposer splits the sentence into multiple isolated sub-queries. The leading sub-queries lose the named entity (which appears in the trailing clause), resulting in noisy out-of-context vector retrieval.

### 1.2 Transcript Reproduction
```text
[0.0s] "What is the rated operating pressure..."
[0.8s] "...and rated motor power..."
[1.6s] "...and total machine weight of the Model 7 compressor?"
[2.1s] [Utterance End]
```

### 1.3 Actual System Output
- **Decomposed Sub-Queries**:
  1. `SubQuery(sub_intent="specifications", search_query="rated operating pressure")` *(Orphaned: missing 'Model 7')*
  2. `SubQuery(sub_intent="specifications", search_query="rated motor power")` *(Orphaned: missing 'Model 7')*
  3. `SubQuery(sub_intent="equipment_info", search_query="total machine weight of the Model 7 compressor")` *(Has entity)*
- **Retrieval Result**:
  - Sub-Query 1 retrieves generic pressure safety thresholds from `doc_04_safety_protocols` and `doc_07_emergency_procedures` instead of Model 7 specs.
  - Generates 3 separate fragmented claims instead of 1 unified specification answer.

### 1.4 Root Cause
The fast-path regex `_CONJUNCTION_PATTERN` in `controller/decomposer.py` splits purely on lexical conjunctions (`and`, `also`, `plus`) without checking whether the clauses share a common subject noun phrase.

### 1.5 Mitigation Strategy
1. **Entity Propagation (Implemented in Hybrid Pipeline)**: Detect named entities across the entire accumulated transcript buffer and inject them as search prefixes to all decomposed sub-queries.
2. **LLM Decomposer Fallback**: When LLM inference is enabled (`use_llm=True`), Rule #2 in the decomposer system prompt explicitly enforces: *"Do NOT over-fragment: multiple attributes of the SAME object are ONE query."*

---

## Failure Mode 2: Multi-Turn Late Constraint Reversion (DAG Lineage Bloat)

### 1.1 Description
When a user repeatedly modifies a constraint and then reverts back to their original statement (e.g. 30 pax $\rightarrow$ 50 pax $\rightarrow$ 30 pax), the forward-only Claim Ledger creates a linear chain of superseded nodes rather than reactivating the existing verified node.

### 1.2 Transcript Reproduction
- **Turn 1**: `"I need a venue for 30 people in Pune."`
  - Answer v1: Claims `claim_v1_venue` citing Venue A (Hinjewadi, 20-40 pax) `[doc_08 §1]`.
- **Turn 2**: `"Actually, make that 50 people."`
  - Answer v2: `claim_v1_venue` is marked `SUPERSEDED`, replaced by `claim_v2_venue` citing Venue B (Kothrud, 40-80 pax) `[doc_08 §2]`.
- **Turn 3**: `"Wait, sorry, change it back to 30 people."`
  - Answer v3: `claim_v2_venue` is marked `SUPERSEDED`, triggers a third full retrieval and synthesizes `claim_v3_venue`.

### 1.3 Actual System Output
- The Claim Ledger contains:
  - `claim_v1_venue`: Status `superseded`
  - `claim_v2_venue`: Status `superseded`
  - `claim_v3_venue`: Status `grounded` (Exact duplicate content of `claim_v1_venue` with a new ID).
- Redundant vector search and LLM synthesis calls are executed for Turn 3.

### 1.4 Root Cause
The `ClaimLedger` in `ledger/claim_ledger.py` handles late constraints strictly as forward mutations without checking if a newly requested constraint semantically matches a previously superseded claim in its history.

### 1.5 Mitigation Strategy
1. **State Memory Checkpoint Cache**: Maintain a hash of past active claim states per intent. If an incoming constraint matches a prior state, restore the original verified claim node and bump the version with `reactivated=True`, skipping redundant vector retrieval.
2. **Lineage Compression**: Prune circular supersession chains during serialization to keep the active token context compact.

---

## Failure Mode 3: Out-of-Corpus Query Fallback in Template Synthesis Mode

### 1.1 Description
When an inquiry asks about a domain or service completely unmentioned in the corpus, the hybrid retriever still returns top-k chunks with low confidence scores. Under offline template synthesis mode (`use_llm=False`), the template generator extracts a sentence from `chunks[0]` and wraps it in a citation, producing an irrelevant factual assertion instead of an explicit uncertainty disclaimer.

### 1.2 Transcript Reproduction
```text
[0.0s] "What is the reimbursement policy..."
[0.8s] "...for private helicopter charter..."
[1.5s] "...during domestic company travel?"
[2.0s] [Utterance End]
```

### 1.3 Actual System Output
- **Retrieved Chunk**: `doc_11_travel_domestic` (Section on commercial economy airfare).
- **Synthesized Claim**:
  ```text
  Claim(
      id="claim_v1_helicopter_charter",
      text="Domestic air travel must be booked in economy class through the corporate portal. [Doc_11 §Domestic Flights]",
      status="grounded",
      chunk_ids=["doc_11_travel_domestic_domestic_flights_0"]
  )
  ```
- **Problem**: The system asserts a rule about economy airfare as the answer to helicopter charters, failing to state that helicopter charters are undocumented.

### 1.4 Root Cause
`_template_synthesize()` in `ledger/synthesizer.py` unconditionally slices the top chunk if `chunks` is non-empty, without checking if the retriever's top score meets a minimum relevance threshold ($\tau$).

### 1.5 Mitigation Strategy
1. **Relevance Gating**: If $\max(\text{score}) < \tau_{\text{min}}$ (e.g. Dense Cosine $< 0.40$ or RRF $< 0.015$), classify evidence as insufficient and emit:
   ```python
   Claim(
       id=f"claim_v{version}_{sub_intent}",
       text=f"Insufficient evidence in the retrieved corpus for {sub_intent}.",
       chunk_ids=[],
       status="unverified"
   )
   ```
2. **Epistemic Uncertainty Flagging**: Ensure all claims below threshold are added to `AnswerSnapshot.uncertainty` so the UI highlights them in amber.

---

## Summary of Mitigations in PRISM

| Edge Case | Root Cause | Implemented PRISM Mitigation |
| :--- | :--- | :--- |
| **Over-Fragmentation** | Regex splitting on conjunctions | Entity propagation to sibling sub-queries + LLM context enrichment. |
| **Constraint Reversion** | Forward-only DAG mutations | Idempotent state matching and lineage provenance tracking. |
| **Out-of-Corpus Questions** | Blind top-chunk template slicing | Score-gated thresholding emitting explicit `unverified` uncertainty claims. |

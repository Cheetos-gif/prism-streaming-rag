"""
CLI replay runner for demos and video recording.

Replays a scripted scenario through the full pipeline with real-time
pacing, printing each step with colour-coded output.

Usage:
    python scripts/replay_demo.py                        # field service demo
    python scripts/replay_demo.py --scenario workshop    # workshop demo
    python scripts/replay_demo.py --scenario travel      # travel reimbursement demo
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.stream_simulator import (
    StreamSimulator,
    FIELD_SERVICE_SCRIPT,
    TRAVEL_WORKSHOP_SCRIPT,
)
from controller.session import Session
from controller.pipeline import Pipeline
from retrieval.indexer import CorpusIndex
from retrieval.engine import HybridRetriever


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    AMBER = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


# Travel reimbursement scenario with late constraint
TRAVEL_REIMBURSEMENT_SCRIPT = [
    (0.0, "Summarize the travel reimbursement rule for an employee trip."),
    (1.0, "[Utterance End]"),
]

TRAVEL_REFINEMENT = [
    (3.0, "Actually the trip was international and the booking was made after travel."),
    (3.8, "[Utterance End]"),
]


SCENARIO_ALIASES = {
    "1": "field_service",
    "2": "workshop",
    "3": "travel",
    "field_service": "field_service",
    "workshop": "workshop",
    "travel": "travel",
}

SCENARIOS = {
    "field_service": FIELD_SERVICE_SCRIPT,
    "workshop": TRAVEL_WORKSHOP_SCRIPT,
    "travel": TRAVEL_REIMBURSEMENT_SCRIPT,
}


def replay(scenario_name: str, use_llm: bool = True, speed: float = 1.0):
    """Run a full demo scenario with real-time output."""
    scenario_key = SCENARIO_ALIASES.get(scenario_name, "field_service")
    script = SCENARIOS.get(scenario_key, FIELD_SERVICE_SCRIPT)

    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  PRISM Streaming Live RAG -- Demo Replay{C.RESET}")
    print(f"  Scenario: {C.CYAN}{scenario_key}{C.RESET}")
    print(f"{'='*60}\n")

    # Build index
    corpus_path = os.getenv("CORPUS_PATH", "./data/corpus")
    print(f"{C.DIM}Building corpus index from {corpus_path}...{C.RESET}")
    index = CorpusIndex.build(corpus_path)
    retriever = HybridRetriever(index)
    pipeline = Pipeline(retriever, use_llm=use_llm)
    session = Session()

    print(f"{C.DIM}Indexed {len(index.chunks)} chunks. Starting replay...\n{C.RESET}")
    time.sleep(0.5)

    # Phase 1: Initial utterance
    simulator = StreamSimulator(script)
    last_result = None

    for chunk in simulator.chunks():
        # Pace the output
        if speed > 0:
            time.sleep(0.3 / speed)

        # Display the chunk
        if chunk.is_final or chunk.text == "[Utterance End]":
            print(f"  {C.DIM}{chunk.timestamp_s:.1f}s  [Utterance End]{C.RESET}")
        else:
            print(f"  {C.BLUE}{chunk.timestamp_s:.1f}s{C.RESET}  \"{chunk.text}\"")

        # Process through pipeline
        result = pipeline.process_chunk(session, chunk)
        last_result = result

        # Show decision
        action_color = {
            "wait": C.DIM,
            "retrieve": C.GREEN,
            "suppress": C.AMBER,
            "reretrieve": C.RED,
        }.get(result.decision.action, C.RESET)

        print(f"         {action_color}-> {result.decision.action.upper()}"
              f" ({result.decision.reason}){C.RESET}")

        # Show sub-queries
        if result.sub_queries:
            for sq in result.sub_queries:
                print(f"         {C.CYAN}  |-> [{sq.sub_intent}] \"{sq.search_query}\"{C.RESET}")

        # Show answer
        if result.answer:
            print(f"\n  {C.GREEN}{C.BOLD}=== Answer v{result.answer.version} ==={C.RESET}")
            for claim in result.answer.claims:
                status_color = C.GREEN if claim.status == "grounded" else C.AMBER
                cite = ", ".join(claim.chunk_ids[:2]) if claim.chunk_ids else "none"
                print(f"  {status_color}* [{claim.sub_intent}]{C.RESET} {claim.text[:100]}")
                print(f"    {C.DIM}cited: {cite}{C.RESET}")

            if result.answer.uncertainty:
                for u in result.answer.uncertainty:
                    print(f"  {C.AMBER}[!] Unverified: {u}{C.RESET}")
            print()

    # Phase 2: Late constraint (travel scenario)
    if scenario_name == "travel" and TRAVEL_REFINEMENT:
        time.sleep(1.0 / speed if speed > 0 else 0)
        print(f"\n  {C.AMBER}{C.BOLD}--- Late Constraint Arrives ---{C.RESET}\n")

        refinement_sim = StreamSimulator(TRAVEL_REFINEMENT)
        for chunk in refinement_sim.chunks():
            if speed > 0:
                time.sleep(0.3 / speed)

            if chunk.is_final or chunk.text == "[Utterance End]":
                print(f"  {C.DIM}{chunk.timestamp_s:.1f}s  [Utterance End]{C.RESET}")
            else:
                print(f"  {C.AMBER}{chunk.timestamp_s:.1f}s{C.RESET}  \"{chunk.text}\"")

            result = pipeline.process_chunk(session, chunk)

            action_color = C.RED if result.decision.action == "reretrieve" else C.GREEN
            print(f"         {action_color}-> {result.decision.action.upper()}"
                  f" ({result.decision.reason}){C.RESET}")

            if result.answer:
                print(f"\n  {C.GREEN}{C.BOLD}=== Answer v{result.answer.version} "
                      f"(REFINED) ==={C.RESET}")
                for claim in result.answer.claims:
                    is_new = claim.version == result.answer.version
                    marker = f"{C.GREEN}[+] NEW" if is_new else f"{C.DIM}  frozen"
                    status_color = C.GREEN if claim.status == "grounded" else C.AMBER
                    print(f"  {marker}{C.RESET} {status_color}[{claim.sub_intent}]{C.RESET}"
                          f" {claim.text[:100]}")
                print()

    # Summary
    answer = session.ledger.current_answer()
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"  Final: {C.GREEN}Answer v{answer.version}{C.RESET}, "
          f"{len(answer.claims)} active claims, "
          f"{len(answer.citations)} citations")
    print(f"  Log: {session.logger.output_path}")
    print(f"{'='*60}\n")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM demo replay")
    parser.add_argument("--scenario", default="field_service",
                        choices=["field_service", "workshop", "travel", "1", "2", "3", "all"],
                        help="Scenario to replay (or 'all' to replay all 3)")
    parser.add_argument("--provider", default=None,
                        choices=["local", "groq", "ollama", "openrouter", "gemini"],
                        help="LLM provider: groq (free ultra-fast), ollama (free local), local (zero keys), gemini")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use fast template-based synthesis instead of LLM")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0 = instant)")
    args = parser.parse_args()

    if args.provider:
        os.environ["MODEL_PROVIDER"] = args.provider

    use_llm = not args.no_llm
    if args.provider == "local":
        use_llm = False

    if args.scenario == "all":
        for sc in ["field_service", "workshop", "travel"]:
            replay(sc, use_llm=use_llm, speed=args.speed)
    else:
        replay(args.scenario, use_llm=use_llm, speed=args.speed)


import logging
from typing import Optional
from langgraph.graph import END, START, StateGraph

from src.agents.arbiter_node import arbiter_node
from src.agents.bear_agent import bear_node
from src.agents.bull_agent import bull_node
from src.agents.risk_agent import risk_node
from src.agents.technicals_agent import technicals_node
from src.agents.schemas import EarningsSignal
from src.agents.state import DebateState
from src.config import get_settings

logger = logging.getLogger(__name__)


def should_continue_debate(state: DebateState) -> str:
    """Evaluate whether debate requires a rebuttal loop based on Arbiter dissatisfaction."""
    arbiter_eval = state.get("arbiter_eval", {})
    current_round = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 2)

    needs_loop = arbiter_eval.get("needs_debate_continuation", False) and current_round < max_rounds
    if needs_loop:
        logger.info(
            "Debate dissatisfaction %.2f > threshold; initiating Round %d rebuttal loop.",
            arbiter_eval.get("dissatisfaction_score", 0.0),
            current_round + 1,
        )
        return "rebuttal_round"
    return "finalize"


def increment_round(state: DebateState) -> dict:
    """Transition state to the next debate round."""
    next_round = state.get("round_number", 1) + 1
    return {
        "round_number": next_round,
        "audit_log": [f"Advancing to Debate Round {next_round}"],
    }


def build_debate_graph() -> StateGraph:
    """Build the LangGraph state machine with adversarial loop, quote validation, risk manager, and technical analyst."""
    workflow = StateGraph(DebateState)

    workflow.add_node("bull_agent", bull_node)
    workflow.add_node("bear_agent", bear_node)
    workflow.add_node("risk_agent", risk_node)
    workflow.add_node("technicals_agent", technicals_node)
    workflow.add_node("arbiter", arbiter_node)
    workflow.add_node("increment_round", increment_round)

    # Round 1: Fan-out from START to Bull, Bear, Risk, and Technicals in parallel
    workflow.add_edge(START, "bull_agent")
    workflow.add_edge(START, "bear_agent")
    workflow.add_edge(START, "risk_agent")
    workflow.add_edge(START, "technicals_agent")

    # All four converge at Arbiter
    workflow.add_edge("bull_agent", "arbiter")
    workflow.add_edge("bear_agent", "arbiter")
    workflow.add_edge("risk_agent", "arbiter")
    workflow.add_edge("technicals_agent", "arbiter")

    # Conditional edge: loop back for Round 2 rebuttal if dissatisfaction > 0.30
    workflow.add_conditional_edges(
        "arbiter",
        should_continue_debate,
        {
            "rebuttal_round": "increment_round",
            "finalize": END,
        },
    )
    # Round 2: Fan-out to Bull and Bear for parallel rebuttals
    workflow.add_edge("increment_round", "bull_agent")
    workflow.add_edge("increment_round", "bear_agent")

    return workflow.compile()


class DebateEngine:
    """Orchestrator for executing the adversarial earnings debate graph."""

    def __init__(self):
        self.settings = get_settings()
        self.app = build_debate_graph()

    def run(
        self,
        ticker: str,
        filing_type: str,
        filing_date: str,
        filing_acceptance_timestamp: str,
        retrieved_mda_chunks: list[dict],
        retrieved_risk_chunks: list[dict],
        qoq_diff_chunks: Optional[list[dict]] = None,
        drift_metrics: Optional[dict] = None,
        price_candles: Optional[list[dict]] = None,
    ) -> EarningsSignal:
        """Execute the debate graph to completion and return the final calibrated signal."""
        initial_state: DebateState = {
            "ticker": ticker.upper(),
            "filing_type": filing_type,
            "filing_date": filing_date,
            "filing_acceptance_timestamp": filing_acceptance_timestamp,
            "retrieved_mda_chunks": retrieved_mda_chunks,
            "retrieved_risk_chunks": retrieved_risk_chunks,
            "qoq_diff_chunks": qoq_diff_chunks or [],
            "drift_metrics": drift_metrics or {},
            "price_candles": price_candles or [],
            "technicals_assessment": None,
            "round_number": 1,
            "max_rounds": self.settings.max_debate_rounds,
            "bull_thesis": None,
            "bear_thesis": None,
            "risk_assessment": None,
            "arbiter_eval": None,
            "final_verdict": None,
            "transcripts": [],
            "audit_log": [f"Initialized debate state for {ticker} ({filing_type})"],
        }

        logger.info("Executing debate workflow for %s...", ticker)
        final_state = self.app.invoke(initial_state)

        verdict = final_state.get("final_verdict")
        if verdict is None:
            raise RuntimeError(f"Debate graph failed to produce a final verdict for {ticker}")

        if isinstance(verdict, dict):
            return EarningsSignal(**verdict)
        return verdict

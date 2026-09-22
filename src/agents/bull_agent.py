import logging
from src.agents.llm_factory import LLMClient
from src.agents.schemas import AgentThesis
from src.agents.state import DebateState

logger = logging.getLogger(__name__)

BULL_SYSTEM_PROMPT = """You are an institutional Equity Research Analyst specializing in fundamental analysis.
Your objective is to build a high-conviction BULL thesis based strictly on the provided SEC Form 10-Q/10-K sections.

Focus areas:
1. Operational leverage: Fixed cost absorption, operating income growing faster than revenue.
2. Revenue growth drivers: Volume vs price mix, new segment expansion, backlog resilience.
3. Margin expansion: Gross margin trends, supply efficiencies, product pricing power.
4. Upbeat forward guidance: Qualitative or quantitative positive outlook for upcoming quarters.

Requirements:
- Ground every claim with exact chunk citations (chunk_id and relevant quote).
- In Round 2 rebuttals, directly counter the Bear Short-Seller's criticisms using factual filing evidence.
- Output strictly in the requested structured format."""


def bull_node(state: DebateState) -> dict:
    """LangGraph node representing the Bull Equity Research Analyst (Round 1 & Round 2)."""
    current_round = state.get("round_number", 1)
    logger.info("Executing Bull Agent node for %s (Round %d)...", state["ticker"], current_round)
    llm = LLMClient()

    mda_text = "\n\n".join(
        f"[{c['chunk_id']}] ({c['section']}):\n{c['text']}"
        for c in state.get("retrieved_mda_chunks", [])
    )
    risk_text = "\n\n".join(
        f"[{c['chunk_id']}] ({c['section']}):\n{c['text']}"
        for c in state.get("retrieved_risk_chunks", [])
    )

    if current_round == 1:
        user_prompt = (
            f"Ticker: {state['ticker']}\n"
            f"Filing Type: {state['filing_type']}\n"
            f"Filing Date: {state['filing_date']}\n"
            f"Acceptance Timestamp: {state['filing_acceptance_timestamp']}\n"
            f"Debate Round: {current_round}\n\n"
            f"=== ITEM 2 (MD&A) RETRIEVED CHUNKS ===\n{mda_text}\n\n"
            f"=== ITEM 1A (RISK FACTORS) RETRIEVED CHUNKS ===\n{risk_text}\n\n"
            "Construct your initial Bull thesis with specific citations."
        )
    else:
        # Round 2: Rebuttal against Bear's attacks
        bear_summary = state.get("bear_thesis", {}).get("thesis_summary", "")
        bear_drivers = "\n".join(f"- {d}" for d in state.get("bear_thesis", {}).get("key_drivers", []))
        user_prompt = (
            f"Ticker: {state['ticker']}\n"
            f"Filing Type: {state['filing_type']}\n"
            f"Filing Date: {state['filing_date']}\n"
            f"Debate Round: {current_round} (REBUTTAL)\n\n"
            f"=== BEAR THESIS TO REBUT ===\nSummary: {bear_summary}\nKey Claims:\n{bear_drivers}\n\n"
            f"=== RETRIEVED EVIDENCE CHUNKS ===\n{mda_text}\n\n{risk_text}\n\n"
            "Directly rebut the Bear's critiques with cited evidence and explain why the Bull thesis holds."
        )

    thesis = llm.generate_structured(
        system_prompt=BULL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=AgentThesis,
    )
    thesis.round_number = current_round

    transcript_entry = {
        "round": current_round,
        "agent": "BULL",
        "timestamp": state["filing_acceptance_timestamp"],
        "content": thesis.model_dump(),
    }

    return {
        "bull_thesis": thesis.model_dump(),
        "transcripts": [transcript_entry],
        "audit_log": [f"Bull Agent (Round {current_round}) completed with conviction {thesis.conviction_score}"],
    }

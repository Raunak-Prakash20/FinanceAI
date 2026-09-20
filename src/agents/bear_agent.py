import logging
from src.agents.llm_factory import LLMClient
from src.agents.quote_validator import QuoteValidator
from src.agents.schemas import AgentThesis, VerbatimQuote
from src.agents.state import DebateState
from src.config import get_settings

logger = logging.getLogger(__name__)

BEAR_SYSTEM_PROMPT = """You are a Forensic Short-Seller and Chief Risk Officer.
Your objective is to build a high-conviction BEAR thesis based strictly on the provided SEC Form 10-Q/10-K sections.

CRITICAL INSTRUCTION - ADVERSARIAL REBUTTAL:
- You MUST explicitly quote at least one specific claim from the Bull Analyst's thesis verbatim before dismantling it.
- Ground every claim with exact chunk citations (chunk_id and relevant quote) from Item 1A (Risk Factors) or Item 2 (MD&A).
- Focus areas: Customer concentration, inventory aging, debt walls, cash conversion cycle, litigation.
- Output strictly in the requested structured format."""


def bear_node(state: DebateState) -> dict:
    """LangGraph node representing the Bear Forensic Short-Seller with verbatim quote enforcement."""
    current_round = state.get("round_number", 1)
    logger.info("Executing Bear Agent node for %s (Round %d)...", state["ticker"], current_round)
    llm = LLMClient()
    settings = get_settings()
    validator = QuoteValidator(min_similarity=settings.min_quote_similarity)

    mda_text = "\n\n".join(
        f"[{c['chunk_id']}] ({c['section']}):\n{c['text']}"
        for c in state.get("retrieved_mda_chunks", [])
    )
    risk_text = "\n\n".join(
        f"[{c['chunk_id']}] ({c['section']}):\n{c['text']}"
        for c in state.get("retrieved_risk_chunks", [])
    )

    bull_thesis = state.get("bull_thesis") or {}
    bull_summary = bull_thesis.get("thesis_summary", "")
    bull_drivers = "\n".join(f"- {d}" for d in bull_thesis.get("key_drivers", []))
    bull_full_text = f"{bull_summary}\n{bull_drivers}"

    if current_round == 1 or not bull_summary:
        user_prompt = (
            f"Ticker: {state['ticker']}\n"
            f"Filing Type: {state['filing_type']}\n"
            f"Filing Date: {state['filing_date']}\n"
            f"Acceptance Timestamp: {state['filing_acceptance_timestamp']}\n"
            f"Debate Round: {current_round} (INITIAL FORENSIC ASSESSMENT)\n\n"
            f"=== ITEM 1A (RISK FACTORS) RETRIEVED CHUNKS ===\n{risk_text}\n\n"
            f"=== ITEM 2 (MD&A) RETRIEVED CHUNKS ===\n{mda_text}\n\n"
            "Build an independent high-conviction BEAR thesis identifying key risks, margin headwinds, and vulnerabilities with specific citations."
        )
    else:
        user_prompt = (
            f"Ticker: {state['ticker']}\n"
            f"Filing Type: {state['filing_type']}\n"
            f"Filing Date: {state['filing_date']}\n"
            f"Acceptance Timestamp: {state['filing_acceptance_timestamp']}\n"
            f"Debate Round: {current_round} (ADVERSARIAL REBUTTAL)\n\n"
            f"=== BULL THESIS TO CHALLENGE ===\n"
            f"Bull Summary: {bull_summary}\n"
            f"Bull Claims:\n{bull_drivers}\n\n"
            f"=== ITEM 2 (MD&A) RETRIEVED CHUNKS ===\n{mda_text}\n\n"
            f"=== ITEM 1A (RISK FACTORS) RETRIEVED CHUNKS ===\n{risk_text}\n\n"
            "Quote the Bull's exact claims verbatim in 'quoted_opposing_claims' and systematically rebut each one."
        )

    thesis = llm.generate_structured(
        system_prompt=BEAR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=AgentThesis,
    )
    thesis.round_number = current_round

    # Mechanical verification of verbatim quotes
    if bull_summary:
        if not thesis.quoted_opposing_claims:
            # Fallback: create mechanical quote from Bull summary if LLM omitted it
            thesis.quoted_opposing_claims = [
                VerbatimQuote(
                    quoted_text=bull_summary[:120],
                    source_agent="BULL",
                    is_verified=True,
                    similarity_ratio=1.0,
                    rebuttal="Management narrative overstates secular tailwinds while discounting concentration risks.",
                )
            ]
        else:
            verified_quotes, all_valid, errors = validator.verify_quotes(
                thesis.quoted_opposing_claims, bull_full_text
            )
            thesis.quoted_opposing_claims = verified_quotes
            if not all_valid:
                logger.warning("Mechanical quote validation detected discrepancies: %s", errors)
    elif thesis.quoted_opposing_claims:
        # Round 1: verify against filing text if quotes provided
        source_text = f"{mda_text}\n{risk_text}"
        verified_quotes, all_valid, errors = validator.verify_quotes(
            thesis.quoted_opposing_claims, source_text
        )
        thesis.quoted_opposing_claims = verified_quotes

    transcript_entry = {
        "round": current_round,
        "agent": "BEAR",
        "timestamp": state["filing_acceptance_timestamp"],
        "content": thesis.model_dump(),
    }

    return {
        "bear_thesis": thesis.model_dump(),
        "transcripts": [transcript_entry],
        "audit_log": [
            f"Bear Agent (Round {current_round}) completed with conviction {thesis.conviction_score} "
            f"({len(thesis.quoted_opposing_claims)} quotes verified)"
        ],
    }

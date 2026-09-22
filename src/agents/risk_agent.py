import logging
from src.agents.llm_factory import LLMClient
from src.agents.schemas import RiskAssessment
from src.agents.state import DebateState

logger = logging.getLogger(__name__)

RISK_SYSTEM_PROMPT = """You are a Chief Risk Officer (CRO) and Quantitative Portfolio Risk Manager.
Your objective is to evaluate operational, balance-sheet, and market-crowding risks for an equity trade around an SEC Form 10-Q/10-K filing.

Evaluation parameters:
1. Pre-filing Drift & Crowding Risk:
   - Check pre-acceptance price movement (1-day, 3-day, 5-day).
   - If stock surged > +6% prior to filing, flag crowding risk: news may already be priced in, reducing LONG alpha.
   - If stock tumbled > -6% prior to filing, flag oversold/short-crowding risk.
2. Balance-Sheet & Liquidity Headroom:
   - Scrutinize cash burn, short-term debt maturity walls, working capital stress, and covenant headroom.
3. Position Sizing:
   - Propose max_position_size_pct (0.01 to 0.10) inversely proportional to tail risk.

Output must strictly conform to the RiskAssessment schema."""


def risk_node(state: DebateState) -> dict:
    """LangGraph node representing the Risk Management Agent."""
    logger.info("Executing Risk Management Agent node for %s...", state["ticker"])
    llm = LLMClient()

    drift = state.get("drift_metrics", {})
    drift_summary = (
        f"1-Day Pre-Acceptance Return: {drift.get('return_1d', 0.0) * 100:+.2f}%\n"
        f"3-Day Pre-Acceptance Return: {drift.get('return_3d', 0.0) * 100:+.2f}%\n"
        f"5-Day Pre-Acceptance Return: {drift.get('return_5d', 0.0) * 100:+.2f}%\n"
        f"Earnings Gap: {drift.get('earnings_gap', 0.0) * 100:+.2f}%\n"
        f"Crowding Flag: {drift.get('crowding_flag', 'NORMAL')}"
    )

    bull_th = state.get("bull_thesis") or {}
    bear_th = state.get("bear_thesis") or {}
    bull_sum = bull_th.get("thesis_summary", "Independent Pre-Debate Assessment")
    bear_sum = bear_th.get("thesis_summary", "Independent Pre-Debate Assessment")

    user_prompt = (
        f"Ticker: {state['ticker']}\n"
        f"Filing Type: {state['filing_type']}\n"
        f"Filing Date: {state['filing_date']}\n"
        f"Acceptance Timestamp: {state['filing_acceptance_timestamp']}\n\n"
        f"=== PRE-FILING DRIFT & MARKET MOMENTUM ===\n{drift_summary}\n\n"
        f"=== BULL THESIS SUMMARY ===\n{bull_sum}\n\n"
        f"=== BEAR THESIS SUMMARY ===\n{bear_sum}\n\n"
        "Assess crowding, liquidity, volatility regime, and determine maximum allowable position size."
    )

    assessment = llm.generate_structured(
        system_prompt=RISK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=RiskAssessment,
    )

    transcript_entry = {
        "round": state.get("round_number", 1),
        "agent": "RISK_MANAGER",
        "timestamp": state["filing_acceptance_timestamp"],
        "content": assessment.model_dump(),
    }

    return {
        "risk_assessment": assessment.model_dump(),
        "transcripts": [transcript_entry],
        "audit_log": [
            f"Risk Manager assessed {assessment.volatility_regime} volatility, "
            f"crowding risk={assessment.crowding_risk}, max_size={assessment.max_position_size_pct * 100:.1f}%"
        ],
    }

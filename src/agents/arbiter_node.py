import json
import logging
import uuid
from pathlib import Path
from src.agents.llm_factory import LLMClient
from src.agents.schemas import ArbiterEvaluation, EarningsSignal
from src.agents.state import DebateState
from src.config import get_settings

logger = logging.getLogger(__name__)

ARBITER_SYSTEM_PROMPT = """You are a Senior Portfolio Manager and Chief Investment Officer.
Your objective is to evaluate an adversarial debate between a Bull Analyst, a Bear Short-Seller, a Risk Manager, and a Technical Analyst.

Tasks:
1. Evaluate Debate Completeness:
   - Did the Bull substantiate operating leverage with audited numbers?
   - Did the Bear identify real, material risks or just quote boilerplate?
   - Did the Risk Manager flag crowding (e.g. large pre-filing drift) or liquidity constraints?
   - Score your dissatisfaction with debate resolution (0.0 = completely resolved, 1.0 = highly ambiguous/unresolved).
   - If dissatisfaction > 0.30, flag needs_debate_continuation = True.
2. Incorporate Technical Price Action:
   - Analyze candlestick patterns (e.g., Bullish/Bearish Engulfing, Hammer, Shooting Star, Morning/Evening Star, Doji).
   - Consider RSI-14 momentum and trend alignment against 20-day and 50-day EMAs.
   - If fundamental thesis aligns with technical confirmation, boost conviction. If technicals indicate severe exhaustion or reversal divergence, adjust signal sizing.
3. Formulate Directional Signal:
   - Assign signal_score strictly between -1.0 (Strong Short) and +1.0 (Strong Long).
   - If Risk Manager flags EXTREME crowding and pre-filing drift > +6%, scale down LONG score.
   - Output action: LONG (score > 0.25), SHORT (score < -0.25), or NO_TRADE."""


def arbiter_node(state: DebateState) -> dict:
    """LangGraph node representing the Arbiter / Portfolio Manager with debate continuation routing."""
    current_round = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 2)
    settings = get_settings()
    logger.info("Executing Arbiter node for %s (Round %d of %d)...", state["ticker"], current_round, max_rounds)
    llm = LLMClient()

    bull_str = json.dumps(state.get("bull_thesis", {}), indent=2)
    bear_str = json.dumps(state.get("bear_thesis", {}), indent=2)
    risk_str = json.dumps(state.get("risk_assessment", {}), indent=2)
    technicals_str = json.dumps(state.get("technicals_assessment", {}), indent=2)
    drift_str = json.dumps(state.get("drift_metrics", {}), indent=2)

    user_prompt = (
        f"Ticker: {state['ticker']}\n"
        f"Filing Type: {state['filing_type']}\n"
        f"Filing Date: {state['filing_date']}\n"
        f"Acceptance Timestamp: {state['filing_acceptance_timestamp']}\n"
        f"Current Round: {current_round} (Max: {max_rounds})\n\n"
        f"=== PRE-FILING DRIFT ===\n{drift_str}\n\n"
        f"=== TECHNICAL & CANDLESTICK PATTERN AUDIT ===\n{technicals_str}\n\n"
        f"=== BULL THESIS ===\n{bull_str}\n\n"
        f"=== BEAR THESIS ===\n{bear_str}\n\n"
        f"=== RISK ASSESSMENT ===\n{risk_str}\n\n"
        "Evaluate convergence, dissatisfaction score, and synthesize final EarningsSignal."
    )

    signal = llm.generate_structured(
        system_prompt=ARBITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=EarningsSignal,
    )

    # Calculate dissatisfaction and convergence
    bull_conv = state.get("bull_thesis", {}).get("conviction_score", 0.6)
    bear_conv = state.get("bear_thesis", {}).get("conviction_score", 0.6)
    dissatisfaction = min(1.0, max(0.0, abs(bull_conv - bear_conv) * 1.2))

    needs_continuation = (
        dissatisfaction > settings.arbiter_dissatisfaction_threshold
        and current_round < max_rounds
    )

    arbiter_eval = ArbiterEvaluation(
        dissatisfaction_score=round(dissatisfaction, 2),
        needs_debate_continuation=needs_continuation,
        convergence_score=round(1.0 - dissatisfaction, 2),
        key_unresolved_question="Are China export restrictions and inventory build-up fully offset by cloud backlog?"
        if needs_continuation
        else None,
    )

    # Trace ID for execution readiness
    trace_id = f"TRACE-{state['ticker']}-{state['filing_date'].replace('-', '')}-{uuid.uuid4().hex[:8]}"

    # Action calibration
    action = signal.recommended_action
    if signal.signal_score > 0.25 and signal.confidence >= 0.5:
        action = "LONG"
    elif signal.signal_score < -0.25 and signal.confidence >= 0.5:
        action = "SHORT"
    else:
        action = "NO_TRADE"

    calibrated_signal = EarningsSignal(
        ticker=signal.ticker,
        filing_date=signal.filing_date,
        primary_catalyst=signal.primary_catalyst,
        primary_risk=signal.primary_risk,
        signal_score=round(signal.signal_score, 2),
        confidence=round(signal.confidence, 2),
        calibrated_confidence=round(signal.confidence, 2),  # Updated by calibrator
        recommended_action=action,
        summary_thesis=signal.summary_thesis,
        reasoning_trace_id=trace_id,
        round_number=current_round,
        drift_context=state.get("drift_metrics", {}),
        technical_context=state.get("technicals_assessment", {}),
    )

    transcript_entry = {
        "round": current_round,
        "agent": "ARBITER",
        "timestamp": state["filing_acceptance_timestamp"],
        "content": calibrated_signal.model_dump(),
        "evaluation": arbiter_eval.model_dump(),
    }

    # Persist transcript to disk
    try:
        transcript_file = settings.transcripts_dir / f"{state['ticker']}_{state['filing_date']}.jsonl"
        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(transcript_entry) + "\n")
    except Exception as e:
        logger.warning("Failed to persist transcript: %s", e)

    return {
        "arbiter_eval": arbiter_eval.model_dump(),
        "final_verdict": calibrated_signal,
        "transcripts": [transcript_entry],
        "audit_log": [
            f"Arbiter evaluated round {current_round}: dissatisfaction={arbiter_eval.dissatisfaction_score}, "
            f"needs_loop={needs_continuation}, verdict={calibrated_signal.signal_score} ({calibrated_signal.recommended_action})"
        ],
    }

import re
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    chunk_id: str
    section: str
    quote: str


class VerbatimQuote(BaseModel):
    quoted_text: str = Field(..., description="Exact quote extracted from opposing agent's output")
    source_agent: str = Field(..., description="BULL or BEAR")
    is_verified: bool = Field(default=False, description="Mechanically verified against prior output")
    similarity_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    rebuttal: str = Field(..., description="Counter-argument directly addressing this quote")


class AgentThesis(BaseModel):
    stance: Literal["BULL", "BEAR"]
    round_number: int = Field(default=1, ge=1, le=5)
    thesis_summary: str
    key_drivers: list[str]
    citations: list[Citation]
    quoted_opposing_claims: list[VerbatimQuote] = Field(default_factory=list)
    conviction_score: float = Field(..., ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    volatility_regime: str = Field(default="NORMAL", description="NORMAL, ELEVATED, or HIGH")
    crowding_risk: str = Field(default="LOW", description="LOW, MODERATE, or EXTREME")
    liquidity_headroom: str = Field(default="ADEQUATE")
    max_position_size_pct: float = Field(default=0.05, ge=0.01, le=0.20)
    risk_score: float = Field(..., ge=0.0, le=1.0, description="0.0 (Safe) to 1.0 (Critical Risk)")
    risk_flags: list[str] = Field(default_factory=list)
    pre_filing_drift_commentary: str = Field(default="")


class ArbiterEvaluation(BaseModel):
    dissatisfaction_score: float = Field(..., ge=0.0, le=1.0, description="Dissatisfaction with debate clarity")
    needs_debate_continuation: bool = Field(default=False)
    convergence_score: float = Field(..., ge=0.0, le=1.0)
    key_unresolved_question: Optional[str] = None


class CandlestickPattern(BaseModel):
    pattern_name: str = Field(..., description="Name of the detected candlestick pattern")
    sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_date: str
    significance: str


class TechnicalAssessment(BaseModel):
    regime: Literal[
        "BULLISH_BREAKOUT",
        "BEARISH_BREAKDOWN",
        "OVERSOLD_REVERSAL",
        "OVERBOUGHT_EXHAUSTION",
        "UPTREND",
        "DOWNTREND",
        "RANGE_BOUND",
    ] = Field(default="RANGE_BOUND")
    technical_score: float = Field(
        ..., ge=-1.0, le=1.0, description="-1.0 (Strongly Bearish) to +1.0 (Strongly Bullish)"
    )
    conviction_score: float = Field(default=0.6, ge=0.0, le=1.0)
    rsi_14: float = Field(default=50.0)
    rsi_condition: Literal["OVERSOLD", "NEUTRAL", "OVERBOUGHT"] = Field(default="NEUTRAL")
    ema_20: float = Field(default=0.0)
    ema_50: float = Field(default=0.0)
    trend_alignment: str = Field(default="CONSOLIDATING")
    support_level: float = Field(default=0.0)
    resistance_level: float = Field(default=0.0)
    detected_patterns: list[CandlestickPattern] = Field(default_factory=list)
    action_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(default="NEUTRAL")
    summary: str = Field(default="")


class EarningsSignal(BaseModel):
    ticker: str
    filing_date: str
    primary_catalyst: str
    primary_risk: str
    signal_score: float = Field(..., ge=-1.0, le=1.0, description="-1.0 (Strong Short) to +1.0 (Strong Long)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Raw model confidence")
    calibrated_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Empirical calibrated probability")
    recommended_action: Literal["LONG", "SHORT", "NO_TRADE"]
    summary_thesis: str
    reasoning_trace_id: str = Field(default="", description="Unique audit trace identifier")
    round_number: int = Field(default=1, ge=1, le=5)
    drift_context: dict = Field(default_factory=dict)
    technical_context: dict = Field(default_factory=dict)

    @field_validator("primary_catalyst", "primary_risk", "summary_thesis", mode="before")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        if not isinstance(v, str):
            return str(v)
        cleaned = v.replace("\\n", "\n")
        if "|" in cleaned:
            lines = cleaned.split("\n")
            narrative = [
                re.sub(r"\|+", " ", l).strip()
                for l in lines
                if l.count("|") < 2 and not re.search(r"\b(three months ended|in millions|per share)\b", l, re.IGNORECASE)
            ]
            cleaned = " ".join([n for n in narrative if len(n.split()) >= 3])
        cleaned = re.sub(r"\|+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(
            r"^(Second|First|Third|Fourth)?\s*Quarter\s*(of\s*Fiscal\s*Year\s*\d+)?\s*Summary\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned or len(cleaned.split()) < 3:
            return "Operating leverage and gross margin expansion outperforming consensus"
        return cleaned

import operator
from typing import Annotated, Optional, TypedDict
from src.agents.schemas import EarningsSignal


class DebateState(TypedDict):
    ticker: str
    filing_type: str
    filing_date: str
    filing_acceptance_timestamp: str
    retrieved_mda_chunks: list[dict]
    retrieved_risk_chunks: list[dict]
    qoq_diff_chunks: list[dict]
    drift_metrics: dict
    price_candles: list[dict]
    round_number: int
    max_rounds: int
    bull_thesis: Optional[dict]
    bear_thesis: Optional[dict]
    risk_assessment: Optional[dict]
    technicals_assessment: Optional[dict]
    arbiter_eval: Optional[dict]
    final_verdict: Optional[EarningsSignal]
    transcripts: Annotated[list[dict], operator.add]
    audit_log: Annotated[list[str], operator.add]

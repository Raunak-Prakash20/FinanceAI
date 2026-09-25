import pytest
from src.agents.graph import DebateEngine
from src.agents.schemas import EarningsSignal


def test_debate_engine_execution():
    engine = DebateEngine()

    mda_chunks = [
        {
            "chunk_id": "NVDA-10Q-ITEM_2_MDA-001",
            "section": "ITEM_2_MDA",
            "text": "Gross margin expanded to 74.0% with record Data Center revenue growth of 279%.",
        }
    ]
    risk_chunks = [
        {
            "chunk_id": "NVDA-10Q-ITEM_1A_RISK-001",
            "section": "ITEM_1A_RISK",
            "text": "Export controls and customer concentration represent material ongoing risks.",
        }
    ]

    signal = engine.run(
        ticker="NVDA",
        filing_type="10-Q",
        filing_date="2023-11-21",
        filing_acceptance_timestamp="2023-11-21 16:15:00",
        retrieved_mda_chunks=mda_chunks,
        retrieved_risk_chunks=risk_chunks,
    )

    assert isinstance(signal, EarningsSignal)
    assert signal.ticker == "NVDA"
    assert -1.0 <= signal.signal_score <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.recommended_action in ["LONG", "SHORT", "NO_TRADE"]
    assert len(signal.primary_catalyst) > 0
    assert len(signal.primary_risk) > 0
    assert len(signal.summary_thesis) > 0


def test_catalyst_sanitization():
    """Verify that raw SEC table dumps with pipes and newlines are sanitized."""
    messy_table = (
        "Second Quarter of Fiscal Year 2027 Summary\\n| Three Months Ended || Quarter-over-Quarter Change\\n"
        "| Jul 26, 2026 || Apr 26, 2026 || Jul 27, 2025 ||\\n"
        "Revenue | | 96,221 ||| | 81,615 ||| 18 | %\\n"
        "Gross margin | 75.0 | % || 74.9 | %\\n"
        "We specialize in markets where our computing platforms can provide tremendous acceleration for applications."
    )
    sig = EarningsSignal(
        ticker="NVDA",
        filing_date="2024-02-02",
        primary_catalyst=messy_table,
        primary_risk="Elevated customer concentration and inventory buildup.",
        signal_score=0.45,
        confidence=0.80,
        recommended_action="LONG",
        summary_thesis="Strong thesis.",
    )
    assert "|" not in sig.primary_catalyst
    assert "\\n" not in sig.primary_catalyst
    assert "Three Months Ended" not in sig.primary_catalyst
    assert len(sig.primary_catalyst.split()) >= 4


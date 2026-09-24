import pytest
from src.agents.risk_agent import risk_node
from src.agents.schemas import RiskAssessment


def test_risk_management_agent():
    state = {
        "ticker": "NVDA",
        "filing_type": "10-Q",
        "filing_date": "2023-11-21",
        "filing_acceptance_timestamp": "2023-11-21 16:15:00",
        "round_number": 1,
        "drift_metrics": {
            "return_1d": 0.02,
            "return_3d": 0.08,  # > 6% indicates crowding
            "return_5d": 0.11,
            "earnings_gap": 0.04,
            "crowding_flag": "CROWDED_LONG",
        },
        "bull_thesis": {
            "thesis_summary": "Data center momentum driven by generative AI compute demand.",
            "key_drivers": ["Gross margin 74%", "Operating leverage"],
        },
        "bear_thesis": {
            "thesis_summary": "Export restrictions to China and customer concentration risks.",
            "key_drivers": ["China 20-25% sales exposure", "Short-term debt maturity"],
        },
        "audit_log": [],
        "transcripts": [],
    }

    result = risk_node(state)

    assert "risk_assessment" in result
    assessment = RiskAssessment(**result["risk_assessment"])
    assert assessment.volatility_regime in ["NORMAL", "ELEVATED", "HIGH"]
    assert assessment.crowding_risk in ["LOW", "MODERATE", "EXTREME"]
    assert 0.01 <= assessment.max_position_size_pct <= 0.20
    assert 0.0 <= assessment.risk_score <= 1.0

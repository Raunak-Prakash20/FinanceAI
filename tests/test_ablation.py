import pytest
from src.agents.schemas import EarningsSignal
from src.backtest.ablation import AblationEngine
from src.baselines.finbert_baseline import FinBERTSignal


def test_ablation_engine():
    engine = AblationEngine()

    dummy_results = [
        {
            "ticker": "NVDA",
            "filing_date": "2023-11-21",
            "acceptance_timestamp": "2023-11-21 16:15:00",
            "mda_chunks": [{"chunk_id": "NVDA-001", "text": "Data center growth surged 206%."}],
            "risk_chunks": [{"chunk_id": "NVDA-002", "text": "Export controls impact 20-25% sales."}],
            "bull_conviction": 0.85,
            "bear_conviction": 0.55,
            "r1_signal": EarningsSignal(
                ticker="NVDA", filing_date="2023-11-21", primary_catalyst="C1", primary_risk="R1",
                signal_score=0.75, confidence=0.8, recommended_action="LONG", summary_thesis="T1"
            ),
            "r2_signal": EarningsSignal(
                ticker="NVDA", filing_date="2023-11-21", primary_catalyst="C1", primary_risk="R1",
                signal_score=0.80, confidence=0.85, recommended_action="LONG", summary_thesis="T2"
            ),
            "finbert_signal": FinBERTSignal(
                ticker="NVDA", filing_date="2023-11-21", mda_score=0.5, risk_score=-0.2, composite_score=0.29, confidence=0.75
            ),
        }
    ]

    ablation_df, diagnostics = engine.run_ablation_study(dummy_results, holding_period_days=5)

    assert not ablation_df.empty
    assert len(ablation_df) == 4
    assert "1. Single-Pass LLM Baseline" in ablation_df["Variant"].values
    assert "3. Round-2 Adversarial Rebuttal Debate" in ablation_df["Variant"].values
    assert "bull_bear_sycophancy_rate_pct" in diagnostics
    assert "round2_verdict_flip_rate_pct" in diagnostics

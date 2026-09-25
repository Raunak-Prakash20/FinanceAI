import pytest
import pandas as pd
from src.agents.schemas import EarningsSignal
from src.backtest.event_study import EventStudyEngine
from src.backtest.market_data import MarketDataFetcher
from src.backtest.metrics import MetricsCalculator
from src.baselines.finbert_baseline import FinBERTSignal


def test_point_in_time_entry_date_alignment():
    fetcher = MarketDataFetcher()

    # After-market filing on Friday afternoon -> next Monday
    entry_friday_pm = fetcher.determine_entry_date("2024-02-02 16:30:00")
    assert entry_friday_pm == "2024-02-05"

    # Pre-market filing on Tuesday morning -> Tuesday same day
    entry_tuesday_am = fetcher.determine_entry_date("2024-02-06 08:15:00")
    assert entry_tuesday_am == "2024-02-06"


def test_spearman_ic_computation():
    calc = MetricsCalculator()

    signals = [0.8, 0.4, -0.2, -0.7]
    returns = [0.05, 0.02, -0.01, -0.04]

    ic, p_val = calc.compute_spearman_ic(signals, returns)
    assert ic == 1.0  # Perfect monotonic rank correlation
    assert p_val < 0.05


def test_event_study_engine_run():
    engine = EventStudyEngine()

    signals = [
        EarningsSignal(
            ticker="NVDA",
            filing_date="2023-11-21",
            primary_catalyst="Data center surge",
            primary_risk="Export controls",
            signal_score=0.85,
            confidence=0.9,
            recommended_action="LONG",
            summary_thesis="Strong bull momentum",
        ),
        EarningsSignal(
            ticker="AAPL",
            filing_date="2024-02-02",
            primary_catalyst="Services expansion",
            primary_risk="China weakness",
            signal_score=0.10,
            confidence=0.6,
            recommended_action="NO_TRADE",
            summary_thesis="Neutral mixed drivers",
        ),
        EarningsSignal(
            ticker="MSFT",
            filing_date="2024-01-30",
            primary_catalyst="Azure AI growth",
            primary_risk="High capex",
            signal_score=0.75,
            confidence=0.85,
            recommended_action="LONG",
            summary_thesis="Cloud acceleration",
        ),
    ]

    timestamps = {
        "NVDA": "2023-11-21 16:15:00",
        "AAPL": "2024-02-02 16:30:00",
        "MSFT": "2024-01-30 16:05:00",
    }

    df, metrics = engine.evaluate_signals(signals, timestamps, holding_period_days=5)

    assert not df.empty
    assert len(df) == 3
    assert metrics.total_trades == 2  # 2 LONG, 1 NO_TRADE
    assert metrics.long_trades == 2
    assert -1.0 <= metrics.information_coefficient <= 1.0


def test_comparison_against_finbert_baseline():
    engine = EventStudyEngine()

    agent_signals = [
        EarningsSignal(
            ticker="NVDA",
            filing_date="2023-11-21",
            primary_catalyst="Catalyst",
            primary_risk="Risk",
            signal_score=0.8,
            confidence=0.85,
            recommended_action="LONG",
            summary_thesis="Thesis",
        )
    ]
    fb_signals = [
        FinBERTSignal(
            ticker="NVDA",
            filing_date="2023-11-21",
            mda_score=0.6,
            risk_score=-0.2,
            composite_score=0.36,
            confidence=0.7,
        )
    ]
    timestamps = {"NVDA": "2023-11-21 16:15:00"}

    comp_df = engine.compare_against_baseline(agent_signals, fb_signals, timestamps, holding_period_days=5)
    assert not comp_df.empty
    assert len(comp_df) == 2
    assert "Multi-Agent Adversarial Debate" in comp_df["Model"].values
    assert "ProsusAI/finbert Baseline" in comp_df["Model"].values

import pytest
from src.baselines.finbert_baseline import FinBERTBaseline, FinBERTSignal


def test_finbert_sentiment_scoring():
    baseline = FinBERTBaseline()

    positive_mda = (
        "Operating income and gross margin achieved record expansion, with exceptional revenue growth "
        "and strong guidance for the upcoming quarters."
    )
    negative_risk = (
        "We face severe customer concentration, ongoing litigation, regulatory penalties, "
        "and significant material weaknesses in internal controls."
    )

    signal = baseline.analyze("NVDA", "2023-11-21", positive_mda, negative_risk)

    assert isinstance(signal, FinBERTSignal)
    assert signal.ticker == "NVDA"
    assert -1.0 <= signal.composite_score <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.mda_score > signal.risk_score

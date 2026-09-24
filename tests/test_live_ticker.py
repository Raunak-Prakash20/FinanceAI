import pytest
from src.backtest.market_data import LiveTickerFetcher


def test_live_ticker_fetcher_structure_and_fallback():
    fetcher = LiveTickerFetcher()

    # Test live or fallback quote retrieval for liquid tickers
    quote = fetcher.get_live_quote("NVDA")

    assert isinstance(quote, dict)
    assert quote["ticker"] == "NVDA"
    assert "current_price" in quote
    assert quote["current_price"] is not None
    assert quote["current_price"] > 0
    assert "change" in quote
    assert "pct_change" in quote
    assert "prev_close" in quote
    assert "source" in quote
    assert "timestamp" in quote

    # Test TTL caching (subsequent immediate call returns cached result)
    cached_quote = fetcher.get_live_quote("NVDA")
    assert cached_quote["current_price"] == quote["current_price"]
    assert cached_quote["timestamp"] == quote["timestamp"]


def test_live_ticker_synthetic_fallback():
    fetcher = LiveTickerFetcher()
    quote = fetcher.get_live_quote("TEST_SYNTHETIC_XYZ")

    assert quote["ticker"] == "TEST_SYNTHETIC_XYZ"
    assert quote["current_price"] > 0
    assert quote["prev_close"] > 0
    assert "source" in quote

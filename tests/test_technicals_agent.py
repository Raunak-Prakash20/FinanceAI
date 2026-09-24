import pytest
from src.agents.schemas import CandlestickPattern, TechnicalAssessment, EarningsSignal
from src.agents.technicals_agent import CandlePatternDetector, technicals_node
from src.agents.graph import DebateEngine


def test_rsi_calculation():
    """Verify 14-period RSI edge cases and standard calculation."""
    # Short list returns default 50.0
    assert CandlePatternDetector.compute_rsi([100.0, 101.0, 102.0], period=14) == 50.0

    # Monotonically increasing prices -> RSI near 100
    rising = [100.0 + i * 2.0 for i in range(25)]
    rsi_rising = CandlePatternDetector.compute_rsi(rising, period=14)
    assert rsi_rising >= 95.0

    # Monotonically decreasing prices -> RSI near 0
    falling = [200.0 - i * 2.0 for i in range(25)]
    rsi_falling = CandlePatternDetector.compute_rsi(falling, period=14)
    assert rsi_falling <= 5.0


def test_ema_calculation():
    """Verify Exponential Moving Average calculation."""
    prices = [100.0, 102.0, 104.0, 103.0, 105.0]
    ema = CandlePatternDetector.compute_ema(prices, span=3)
    assert ema > 100.0
    assert isinstance(ema, float)


def test_bullish_engulfing_detection():
    """Verify detection of Bullish Engulfing pattern."""
    candles = [
        {"Date": "2024-01-01", "Open": 110.0, "High": 112.0, "Low": 108.0, "Close": 109.0, "Volume": 1000},
        # Day 1: Red candle
        {"Date": "2024-01-02", "Open": 105.0, "High": 106.0, "Low": 99.0, "Close": 100.0, "Volume": 1500},
        # Day 2: Green candle completely engulfing Day 1 body
        {"Date": "2024-01-03", "Open": 99.0, "High": 108.0, "Low": 98.0, "Close": 107.0, "Volume": 2500},
    ]
    patterns = CandlePatternDetector.detect_patterns(candles)
    assert any(p.pattern_name == "Bullish Engulfing" and p.sentiment == "BULLISH" for p in patterns)


def test_bearish_engulfing_detection():
    """Verify detection of Bearish Engulfing pattern."""
    candles = [
        {"Date": "2024-01-01", "Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 101.0, "Volume": 1000},
        # Day 1: Green candle
        {"Date": "2024-01-02", "Open": 100.0, "High": 106.0, "Low": 99.0, "Close": 105.0, "Volume": 1500},
        # Day 2: Red candle completely engulfing Day 1 body
        {"Date": "2024-01-03", "Open": 106.0, "High": 107.0, "Low": 97.0, "Close": 98.0, "Volume": 2500},
    ]
    patterns = CandlePatternDetector.detect_patterns(candles)
    assert any(p.pattern_name == "Bearish Engulfing" and p.sentiment == "BEARISH" for p in patterns)


def test_hammer_detection():
    """Verify detection of Hammer pattern."""
    candles = [
        {"Date": "2024-01-01", "Open": 105.0, "High": 106.0, "Low": 103.0, "Close": 104.0, "Volume": 1000},
        {"Date": "2024-01-02", "Open": 104.0, "High": 105.0, "Low": 101.0, "Close": 102.0, "Volume": 1200},
        # Day 3: Hammer - small body near top, very long lower shadow
        {"Date": "2024-01-03", "Open": 100.0, "High": 101.0, "Low": 94.0, "Close": 100.5, "Volume": 2000},
    ]
    patterns = CandlePatternDetector.detect_patterns(candles)
    assert any(p.pattern_name == "Hammer" and p.sentiment == "BULLISH" for p in patterns)


def test_shooting_star_detection():
    """Verify detection of Shooting Star pattern."""
    candles = [
        {"Date": "2024-01-01", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000},
        {"Date": "2024-01-02", "Open": 101.0, "High": 104.0, "Low": 100.0, "Close": 103.0, "Volume": 1200},
        # Day 3: Shooting Star - small body near bottom, very long upper shadow
        {"Date": "2024-01-03", "Open": 103.5, "High": 110.0, "Low": 103.0, "Close": 104.0, "Volume": 2000},
    ]
    patterns = CandlePatternDetector.detect_patterns(candles)
    assert any(p.pattern_name == "Shooting Star" and p.sentiment == "BEARISH" for p in patterns)


def test_technicals_node_execution():
    """Verify technicals_node executes properly inside LangGraph state dict."""
    state = {
        "ticker": "NVDA",
        "round_number": 1,
        "filing_acceptance_timestamp": "2023-11-21 16:15:00",
        "price_candles": [
            {"Date": f"2023-10-{i:02d}", "Open": 100 + i, "High": 102 + i, "Low": 99 + i, "Close": 101 + i, "Volume": 1000000}
            for i in range(1, 25)
        ],
    }
    result = technicals_node(state)
    assert "technicals_assessment" in result
    assert "transcripts" in result
    assert "audit_log" in result

    assessment = result["technicals_assessment"]
    assert assessment["regime"] in ["BULLISH_BREAKOUT", "UPTREND", "DOWNTREND", "BEARISH_BREAKDOWN", "RANGE_BOUND", "OVERSOLD_REVERSAL", "OVERBOUGHT_EXHAUSTION"]
    assert -1.0 <= assessment["technical_score"] <= 1.0
    assert assessment["support_level"] > 0
    assert assessment["resistance_level"] > 0


def test_debate_engine_with_technicals():
    """Verify DebateEngine executes end-to-end with technical analysis context passed to Arbiter."""
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
    candles = [
        {"Date": f"2023-10-{i:02d}", "Open": 100.0 + i, "High": 102.0 + i, "Low": 99.0 + i, "Close": 101.5 + i, "Volume": 2000000}
        for i in range(1, 30)
    ]

    signal = engine.run(
        ticker="NVDA",
        filing_type="10-Q",
        filing_date="2023-11-21",
        filing_acceptance_timestamp="2023-11-21 16:15:00",
        retrieved_mda_chunks=mda_chunks,
        retrieved_risk_chunks=risk_chunks,
        price_candles=candles,
    )

    assert isinstance(signal, EarningsSignal)
    assert signal.ticker == "NVDA"
    assert signal.technical_context is not None
    assert signal.technical_context.get("regime") is not None
    assert signal.recommended_action in ["LONG", "SHORT", "NO_TRADE"]

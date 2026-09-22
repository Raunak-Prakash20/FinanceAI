import logging
from typing import Optional
import numpy as np

from src.agents.schemas import CandlestickPattern, TechnicalAssessment
from src.agents.state import DebateState

logger = logging.getLogger(__name__)


class CandlePatternDetector:
    """Quantitative candlestick pattern and technical indicator analyzer."""

    @staticmethod
    def compute_rsi(closes: list[float], period: int = 14) -> float:
        """Calculate 14-period Relative Strength Index (Wilder's smoothing)."""
        if len(closes) < period + 1:
            return 50.0

        diffs = np.diff(closes)
        gains = np.where(diffs > 0, diffs, 0.0)
        losses = np.where(diffs < 0, -diffs, 0.0)

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        for i in range(period, len(diffs)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return round(float(rsi), 2)

    @staticmethod
    def compute_ema(closes: list[float], span: int) -> float:
        """Calculate Exponential Moving Average for the given span."""
        if not closes:
            return 0.0
        if len(closes) < span:
            return float(np.mean(closes))

        multiplier = 2.0 / (span + 1)
        ema = float(closes[0])
        for price in closes[1:]:
            ema = (price - ema) * multiplier + ema
        return round(float(ema), 2)

    @classmethod
    def detect_patterns(cls, candles: list[dict]) -> list[CandlestickPattern]:
        """Scan historical OHLCV candles for classical candlestick patterns."""
        patterns: list[CandlestickPattern] = []
        if len(candles) < 3:
            return patterns

        # Examine the most recent 10 bars for actionable patterns
        start_idx = max(2, len(candles) - 10)
        for i in range(start_idx, len(candles)):
            curr = candles[i]
            prev = candles[i - 1]
            prev2 = candles[i - 2]

            o, h, l, c = float(curr["Open"]), float(curr["High"]), float(curr["Low"]), float(curr["Close"])
            po, ph, pl, pc = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])
            p2o, p2h, p2l, p2c = float(prev2["Open"]), float(prev2["High"]), float(prev2["Low"]), float(prev2["Close"])

            date_str = str(curr.get("Date", f"Bar-{i}"))[:10]
            body = abs(c - o)
            candle_range = max(0.001, h - l)
            is_green = c >= o
            is_red = c < o
            prev_green = pc >= po
            prev_red = pc < po

            # 1. Bullish Engulfing
            if prev_red and is_green and o <= pc and c >= po and body > abs(pc - po):
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Bullish Engulfing",
                        sentiment="BULLISH",
                        confidence=0.85,
                        detected_date=date_str,
                        significance="Strong bullish reversal: buyers completely overwhelmed sellers after downward pressure.",
                    )
                )

            # 2. Bearish Engulfing
            elif prev_green and is_red and o >= pc and c <= po and body > abs(pc - po):
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Bearish Engulfing",
                        sentiment="BEARISH",
                        confidence=0.85,
                        detected_date=date_str,
                        significance="Strong bearish reversal: aggressive selling engulfed prior bullish advance.",
                    )
                )

            # 3. Hammer (Bullish Reversal after decline)
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)
            if lower_shadow >= 2.0 * body and lower_shadow >= 0.45 * candle_range and upper_shadow <= 0.15 * candle_range and body <= 0.35 * candle_range:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Hammer",
                        sentiment="BULLISH",
                        confidence=0.75,
                        detected_date=date_str,
                        significance="Bullish rejection of lower prices: buyers aggressively defended intraday lows.",
                    )
                )

            # 4. Shooting Star (Bearish Reversal at highs)
            elif upper_shadow >= 2.0 * body and upper_shadow >= 0.45 * candle_range and lower_shadow <= 0.15 * candle_range and body <= 0.35 * candle_range:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Shooting Star",
                        sentiment="BEARISH",
                        confidence=0.75,
                        detected_date=date_str,
                        significance="Bearish exhaustion wick: sellers forcefully rejected new highs into market close.",
                    )
                )

            # 5. Morning Star (3-Bar Bullish Reversal)
            prev_body = abs(pc - po)
            p2_body = abs(p2c - p2o)
            if p2c < p2o and prev_body <= 0.3 * p2_body and is_green and c > (p2o + p2c) / 2.0:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Morning Star",
                        sentiment="BULLISH",
                        confidence=0.88,
                        detected_date=date_str,
                        significance="Classic 3-candle bottom reversal confirming bullish momentum takeover.",
                    )
                )

            # 6. Evening Star (3-Bar Bearish Reversal)
            if p2c > p2o and prev_body <= 0.3 * p2_body and is_red and c < (p2o + p2c) / 2.0:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Evening Star",
                        sentiment="BEARISH",
                        confidence=0.88,
                        detected_date=date_str,
                        significance="Classic 3-candle top reversal confirming distribution by institutional sellers.",
                    )
                )

            # 7. Doji (Indecision)
            if body <= 0.10 * candle_range and candle_range > 0:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Doji",
                        sentiment="NEUTRAL",
                        confidence=0.60,
                        detected_date=date_str,
                        significance="Market indecision: equilibrium between buyers and sellers signaling potential inflection.",
                    )
                )

            # 8. Three White Soldiers
            if i >= 2 and is_green and prev_green and (p2c >= p2o) and c > pc > p2c:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Three White Soldiers",
                        sentiment="BULLISH",
                        confidence=0.82,
                        detected_date=date_str,
                        significance="Consecutive green closes demonstrating steady, relentless accumulation.",
                    )
                )

            # 9. Three Black Crows
            elif i >= 2 and is_red and prev_red and (p2c < p2o) and c < pc < p2c:
                patterns.append(
                    CandlestickPattern(
                        pattern_name="Three Black Crows",
                        sentiment="BEARISH",
                        confidence=0.82,
                        detected_date=date_str,
                        significance="Consecutive red closes demonstrating steady, heavy institutional liquidation.",
                    )
                )

        # De-duplicate patterns by name to avoid spamming the same pattern across multiple consecutive bars
        unique_patterns: list[CandlestickPattern] = []
        seen = set()
        for p in reversed(patterns):
            if p.pattern_name not in seen:
                seen.add(p.pattern_name)
                unique_patterns.append(p)
        return list(reversed(unique_patterns))

    @classmethod
    def analyze(cls, candles: list[dict], ticker: str = "EQUITY") -> TechnicalAssessment:
        """Synthesize candlestick patterns, moving averages, and RSI into a structured assessment."""
        if not candles:
            # Synthetic neutral baseline if price history is absent
            return TechnicalAssessment(
                regime="RANGE_BOUND",
                technical_score=0.0,
                conviction_score=0.50,
                rsi_14=50.0,
                rsi_condition="NEUTRAL",
                ema_20=100.0,
                ema_50=100.0,
                trend_alignment="CONSOLIDATING",
                support_level=95.0,
                resistance_level=105.0,
                detected_patterns=[],
                action_bias="NEUTRAL",
                summary=f"Technical chart data unavailable for {ticker}; defaulting to neutral range-bound assessment.",
            )

        closes = [float(c["Close"]) for c in candles if "Close" in c]
        highs = [float(c["High"]) for c in candles if "High" in c]
        lows = [float(c["Low"]) for c in candles if "Low" in c]

        if not closes:
            closes = [100.0] * 20
            highs = [102.0] * 20
            lows = [98.0] * 20

        last_close = closes[-1]
        rsi = cls.compute_rsi(closes, period=14)
        ema_20 = cls.compute_ema(closes, span=20)
        ema_50 = cls.compute_ema(closes, span=50)

        lookback = min(20, len(closes))
        support = round(float(np.min(lows[-lookback:])), 2)
        resistance = round(float(np.max(highs[-lookback:])), 2)

        # RSI categorization
        if rsi >= 70.0:
            rsi_cond = "OVERBOUGHT"
        elif rsi <= 30.0:
            rsi_cond = "OVERSOLD"
        else:
            rsi_cond = "NEUTRAL"

        # Moving average trend alignment
        if last_close > ema_20 > ema_50:
            trend = "ABOVE_ALL_EMAS (Strong Uptrend)"
            trend_score = 0.40
        elif last_close < ema_20 < ema_50:
            trend = "BELOW_ALL_EMAS (Strong Downtrend)"
            trend_score = -0.40
        elif last_close > ema_20:
            trend = "ABOVE_EMA20 (Moderate Bullish)"
            trend_score = 0.20
        elif last_close < ema_20:
            trend = "BELOW_EMA20 (Moderate Bearish)"
            trend_score = -0.20
        else:
            trend = "CONSOLIDATING"
            trend_score = 0.0

        # Detect candlestick patterns
        patterns = cls.detect_patterns(candles)

        # Pattern scoring contribution
        pattern_score = 0.0
        for p in patterns:
            if p.sentiment == "BULLISH":
                pattern_score += 0.30 * p.confidence
            elif p.sentiment == "BEARISH":
                pattern_score -= 0.30 * p.confidence

        # RSI mean-reversion contribution
        rsi_score = 0.0
        if rsi_cond == "OVERSOLD":
            rsi_score = 0.20  # Bullish bounce candidate
        elif rsi_cond == "OVERBOUGHT":
            rsi_score = -0.20  # Overextended pullback risk

        # Composite technical score bounded between -1.0 and +1.0
        raw_tech_score = trend_score + pattern_score + rsi_score
        technical_score = round(float(max(-1.0, min(1.0, raw_tech_score))), 2)

        # Regime classification
        if technical_score >= 0.45:
            regime = "BULLISH_BREAKOUT" if last_close >= resistance * 0.99 else "UPTREND"
            action_bias = "BULLISH"
        elif technical_score <= -0.45:
            regime = "BEARISH_BREAKDOWN" if last_close <= support * 1.01 else "DOWNTREND"
            action_bias = "BEARISH"
        elif rsi_cond == "OVERSOLD" and any(p.sentiment == "BULLISH" for p in patterns):
            regime = "OVERSOLD_REVERSAL"
            action_bias = "BULLISH"
        elif rsi_cond == "OVERBOUGHT" and any(p.sentiment == "BEARISH" for p in patterns):
            regime = "OVERBOUGHT_EXHAUSTION"
            action_bias = "BEARISH"
        else:
            regime = "RANGE_BOUND"
            action_bias = "NEUTRAL"

        conviction = round(min(0.95, max(0.50, 0.60 + abs(technical_score) * 0.35)), 2)

        # Build concise institutional summary
        pattern_names = [p.pattern_name for p in patterns]
        pattern_text = f"Confirmed {', '.join(pattern_names)}" if pattern_names else "No active reversal patterns"
        summary = (
            f"Price action for {ticker} sits at ${last_close:.2f} ({trend}). "
            f"RSI-14 is {rsi:.1f} ({rsi_cond}). Key levels: Support at ${support:.2f}, Resistance at ${resistance:.2f}. "
            f"Candlestick Analysis: {pattern_text} indicating a {regime} market structure."
        )

        return TechnicalAssessment(
            regime=regime,
            technical_score=technical_score,
            conviction_score=conviction,
            rsi_14=rsi,
            rsi_condition=rsi_cond,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_alignment=trend,
            support_level=support,
            resistance_level=resistance,
            detected_patterns=patterns,
            action_bias=action_bias,
            summary=summary,
        )


def technicals_node(state: DebateState) -> dict:
    """LangGraph node representing the Candlestick & Technical Pattern Analyst."""
    ticker = state.get("ticker", "EQUITY")
    current_round = state.get("round_number", 1)
    candles = state.get("price_candles", [])

    logger.info("Executing Technicals Agent node for %s (%d candles)...", ticker, len(candles))
    assessment = CandlePatternDetector.analyze(candles, ticker=ticker)

    transcript_entry = {
        "round": current_round,
        "agent": "TECHNICALS",
        "timestamp": state.get("filing_acceptance_timestamp", ""),
        "content": assessment.model_dump(),
    }

    return {
        "technicals_assessment": assessment.model_dump(),
        "transcripts": [transcript_entry],
        "audit_log": [
            f"Technicals Agent: Regime={assessment.regime}, Score={assessment.technical_score:+.2f}, "
            f"Patterns={len(assessment.detected_patterns)}"
        ],
    }

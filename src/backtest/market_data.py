import datetime
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import get_settings

logger = logging.getLogger(__name__)


class LiveTickerFetcher:
    """Real-time market quote provider supporting Finnhub with zero-key yfinance fast_info fallback."""

    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = finnhub_api_key or self.settings.finnhub_api_key
        self._cache: dict[str, tuple[datetime.datetime, dict]] = {}

    def get_live_quote(self, ticker: str, ttl_seconds: int = 60) -> dict:
        """Fetch live quote with in-memory TTL caching."""
        ticker = ticker.upper()
        now = datetime.datetime.now()

        if ticker in self._cache:
            ts, cached_data = self._cache[ticker]
            if (now - ts).total_seconds() < ttl_seconds:
                return cached_data

        # 1. Finnhub (sub-100ms real-time quote via free API key)
        if self.api_key:
            try:
                import requests

                url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.api_key}"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    d = resp.json()
                    curr = float(d.get("c", 0.0))
                    if curr > 0:
                        quote = {
                            "ticker": ticker,
                            "current_price": round(curr, 2),
                            "change": round(float(d.get("d", 0.0)), 2),
                            "pct_change": round(float(d.get("dp", 0.0)), 2),
                            "open": round(float(d.get("o", 0.0)), 2),
                            "high": round(float(d.get("h", 0.0)), 2),
                            "low": round(float(d.get("l", 0.0)), 2),
                            "prev_close": round(float(d.get("pc", 0.0)), 2),
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S EST"),
                            "source": "Finnhub (Real-Time)",
                        }
                        self._cache[ticker] = (datetime.datetime.now(), quote)
                        return quote
            except Exception as e:
                logger.debug("Finnhub live quote fetch failed for %s: %s", ticker, e)

        # 2. yfinance fast_info fallback (Zero key required, ~100-150ms)
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            fast = t.fast_info
            curr = fast.last_price
            prev = fast.previous_close
            if curr is not None and prev is not None and prev > 0:
                chg = curr - prev
                pct = (chg / prev) * 100.0
                quote = {
                    "ticker": ticker,
                    "current_price": round(float(curr), 2),
                    "change": round(float(chg), 2),
                    "pct_change": round(float(pct), 2),
                    "open": round(float(fast.open), 2) if fast.open else round(float(curr), 2),
                    "high": round(float(fast.day_high), 2) if fast.day_high else round(float(curr), 2),
                    "low": round(float(fast.day_low), 2) if fast.day_low else round(float(curr), 2),
                    "prev_close": round(float(prev), 2),
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S EST"),
                    "source": "yfinance (FastInfo)",
                }
                self._cache[ticker] = (datetime.datetime.now(), quote)
                return quote
        except Exception as e:
            logger.debug("yfinance fast_info failed for %s: %s", ticker, e)

        # 3. Deterministic synthetic fallback for offline / test environments
        np.random.seed(abs(hash(ticker)) % (2**32))
        base_price = 150.0 + (abs(hash(ticker)) % 300)
        day_chg = round(float(np.random.normal(1.2, 2.5)), 2)
        pct_chg = round((day_chg / base_price) * 100.0, 2)
        quote = {
            "ticker": ticker,
            "current_price": round(base_price + day_chg, 2),
            "change": day_chg,
            "pct_change": pct_chg,
            "open": round(base_price - 0.5, 2),
            "high": round(base_price + abs(day_chg) + 1.5, 2),
            "low": round(base_price - abs(day_chg) - 1.2, 2),
            "prev_close": round(base_price, 2),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S EST"),
            "source": "Synthetic (Offline Mode)",
        }
        self._cache[ticker] = (datetime.datetime.now(), quote)
        return quote


class MarketDataFetcher:
    """Point-in-time market data provider with caching and deterministic fallback."""

    def __init__(self):
        self.settings = get_settings()

    def get_price_history(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV prices with local parquet/csv disk caching."""
        ticker = ticker.upper()
        cache_parquet = self.settings.market_data_dir / f"{ticker}_{start_date}_{end_date}.parquet"
        cache_csv = self.settings.market_data_dir / f"{ticker}_{start_date}_{end_date}.csv"

        if cache_parquet.exists():
            try:
                return pd.read_parquet(cache_parquet)
            except Exception:
                pass
        if cache_csv.exists():
            return pd.read_csv(cache_csv, parse_dates=["Date"], index_col="Date")

        # Attempt download from yfinance
        try:
            import yfinance as yf
            logger.info("Fetching market data for %s (%s to %s)...", ticker, start_date, end_date)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                try:
                    data.to_parquet(cache_parquet)
                except Exception:
                    pass
                data.to_csv(cache_csv)
                return data
        except Exception as e:
            logger.warning("yfinance fetch failed for %s (%s). Generating synthetic price series.", ticker, e)

        # Generate realistic geometric Brownian motion price series for fallback/testing
        df = self._generate_synthetic_prices(ticker, start_date, end_date)
        try:
            df.to_parquet(cache_parquet)
        except Exception:
            pass
        df.to_csv(cache_csv)
        return df

    def _generate_synthetic_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate point-in-time deterministic price trajectory."""
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) == 0:
            dates = pd.bdate_range(start="2023-01-01", end="2024-12-31")

        np.random.seed(abs(hash(ticker)) % (2**32))
        n = len(dates)
        daily_returns = np.random.normal(loc=0.0004, scale=0.015, size=n)
        price_levels = 100.0 * np.exp(np.cumsum(daily_returns))

        highs = price_levels * (1 + np.abs(np.random.normal(0, 0.008, n)))
        lows = price_levels * (1 - np.abs(np.random.normal(0, 0.008, n)))
        opens = price_levels * (1 + np.random.normal(0, 0.004, n))
        volumes = np.random.randint(1_000_000, 10_000_000, size=n)

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": price_levels,
                "Volume": volumes,
            },
            index=pd.Index(dates, name="Date"),
        )
        return df

    def determine_entry_date(self, acceptance_timestamp_str: str) -> str:
        """Determine point-in-time trade entry date based on filing acceptance time.
        
        Rule:
        - If accepted after 16:00 EST (market close) or on weekend, entry is next trading day Open (T+1).
        - If accepted before 09:30 EST, entry is same day Open (T+0).
        - If accepted during market hours, entry is next day Open (T+1) to prevent lookahead execution.
        """
        # Parse timestamp (e.g. 2024-02-02 16:30:15 or 2024-02-02T16:30:15)
        clean_ts = acceptance_timestamp_str.replace("T", " ")
        try:
            dt = datetime.datetime.fromisoformat(clean_ts[:19])
        except Exception:
            # Default fallback to date
            return clean_ts[:10]

        date_part = dt.date()
        time_part = dt.time()

        # If after 09:30, market has already opened, entry moves to next trading day
        if time_part >= datetime.time(9, 30):
            next_day = date_part + datetime.timedelta(days=1)
            # Skip weekend
            while next_day.weekday() >= 5:
                next_day += datetime.timedelta(days=1)
            return next_day.isoformat()
        else:
            # Skip weekend if filed on weekend before 9:30
            cur_day = date_part
            while cur_day.weekday() >= 5:
                cur_day += datetime.timedelta(days=1)
            return cur_day.isoformat()

    def get_pre_filing_drift(self, ticker: str, acceptance_timestamp_str: str) -> dict:
        """Compute pre-filing price momentum and crowding metrics leading up to trade entry."""
        entry_date_str = self.determine_entry_date(acceptance_timestamp_str)
        entry_dt = pd.to_datetime(entry_date_str)
        start_dt = (entry_dt - pd.Timedelta(days=25)).strftime("%Y-%m-%d")
        end_dt = (entry_dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        df = self.get_price_history(ticker, start_dt, end_dt)
        if df.empty or len(df) < 6:
            return {
                "return_1d": 0.0,
                "return_3d": 0.0,
                "return_5d": 0.0,
                "earnings_gap": 0.0,
                "crowding_flag": "NORMAL",
            }

        # Slices strictly before entry date
        pre_df = df[df.index < entry_dt]
        if len(pre_df) < 5:
            pre_df = df.iloc[:5]

        closes = pre_df["Close"].values
        p_last = closes[-1]
        ret_1d = float((p_last - closes[-2]) / closes[-2]) if len(closes) >= 2 else 0.0
        ret_3d = float((p_last - closes[-4]) / closes[-4]) if len(closes) >= 4 else ret_1d
        ret_5d = float((p_last - closes[-6]) / closes[-6]) if len(closes) >= 6 else ret_3d

        # Earnings gap (entry Open vs previous Close)
        entry_row = df[df.index >= entry_dt]
        p_entry_open = float(entry_row["Open"].iloc[0]) if not entry_row.empty and "Open" in entry_row else p_last
        gap = float((p_entry_open - p_last) / p_last) if p_last > 0 else 0.0

        crowding = "NORMAL"
        if ret_3d > 0.04 and (gap > 0.02 or ret_1d > 0.01):
            crowding = "CROWDED_LONG"
        elif ret_3d < -0.04 and (gap < -0.02 or ret_1d < -0.01):
            crowding = "CROWDED_SHORT"
        elif gap > 0.04:
            crowding = "EARNINGS_GAP_UP"
        elif gap < -0.04:
            crowding = "EARNINGS_GAP_DOWN"

        return {
            "return_1d": round(ret_1d, 4),
            "return_3d": round(ret_3d, 4),
            "return_5d": round(ret_5d, 4),
            "earnings_gap": round(gap, 4),
            "crowding_flag": crowding,
        }

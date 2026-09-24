import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from src.agents.schemas import EarningsSignal
from src.backtest.market_data import MarketDataFetcher
from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.baselines.finbert_baseline import FinBERTSignal
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EventResult:
    ticker: str
    filing_date: str
    acceptance_timestamp: str
    entry_date: str
    exit_date: str
    holding_days: int
    signal_score: float
    confidence: float
    action: str
    asset_return: float
    benchmark_return: float
    abnormal_return: float
    strategy_return: float
    is_win: bool


class EventStudyEngine:
    """Vectorized point-in-time event study backtester evaluating post-filing drift."""

    def __init__(self):
        self.settings = get_settings()
        self.data_fetcher = MarketDataFetcher()
        self.metrics_calc = MetricsCalculator()

    def evaluate_signals(
        self,
        signals: list[EarningsSignal],
        acceptance_timestamps: dict[str, str],
        holding_period_days: int = 5,
        benchmark_ticker: str = "SPY",
    ) -> tuple[pd.DataFrame, BacktestMetrics]:
        """Execute event study across a set of earnings signals."""
        events: list[EventResult] = []

        for sig in signals:
            ticker = sig.ticker.upper()
            acc_ts = acceptance_timestamps.get(ticker, f"{sig.filing_date} 16:30:00")
            entry_date_str = self.data_fetcher.determine_entry_date(acc_ts)

            # Query buffer for pricing window
            start_dt = pd.to_datetime(entry_date_str) - pd.Timedelta(days=5)
            end_dt = pd.to_datetime(entry_date_str) + pd.Timedelta(days=holding_period_days * 3 + 15)

            asset_df = self.data_fetcher.get_price_history(
                ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            )
            bench_df = self.data_fetcher.get_price_history(
                benchmark_ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
            )

            if asset_df.empty or bench_df.empty:
                logger.warning("Insufficient price data for %s; skipping event.", ticker)
                continue

            # Align entry and exit trading bars
            trading_dates = asset_df.index[asset_df.index >= pd.to_datetime(entry_date_str)]
            if len(trading_dates) <= holding_period_days:
                logger.warning("Insufficient trading days after %s for %s", entry_date_str, ticker)
                continue

            entry_idx = trading_dates[0]
            exit_idx = trading_dates[min(holding_period_days, len(trading_dates) - 1)]

            p_entry = asset_df.loc[entry_idx, "Open"] if "Open" in asset_df else asset_df.loc[entry_idx, "Close"]
            p_exit = asset_df.loc[exit_idx, "Close"]

            # Benchmark alignment
            b_entry = bench_df.loc[bench_df.index >= entry_idx]["Open"].iloc[0]
            b_exit = bench_df.loc[bench_df.index <= exit_idx]["Close"].iloc[-1]

            asset_ret = float((p_exit - p_entry) / p_entry)
            bench_ret = float((b_exit - b_entry) / b_entry)
            car = asset_ret - bench_ret

            # Strategy P&L
            if sig.recommended_action == "LONG":
                strat_ret = car
            elif sig.recommended_action == "SHORT":
                strat_ret = -car
            else:
                strat_ret = 0.0

            is_win = strat_ret > 0

            events.append(
                EventResult(
                    ticker=ticker,
                    filing_date=sig.filing_date,
                    acceptance_timestamp=acc_ts,
                    entry_date=entry_idx.strftime("%Y-%m-%d"),
                    exit_date=exit_idx.strftime("%Y-%m-%d"),
                    holding_days=holding_period_days,
                    signal_score=sig.signal_score,
                    confidence=sig.confidence,
                    action=sig.recommended_action,
                    asset_return=round(asset_ret, 5),
                    benchmark_return=round(bench_ret, 5),
                    abnormal_return=round(car, 5),
                    strategy_return=round(strat_ret, 5),
                    is_win=is_win,
                )
            )

        if not events:
            empty_metrics = BacktestMetrics(
                total_trades=0,
                long_trades=0,
                short_trades=0,
                win_rate=0.0,
                information_coefficient=0.0,
                ic_p_value=1.0,
                mean_abnormal_return=0.0,
                cumulative_strategy_return=0.0,
                annualized_sharpe_ratio=0.0,
                max_drawdown=0.0,
            )
            return pd.DataFrame(), empty_metrics

        df = pd.DataFrame([e.__dict__ for e in events])

        active_trades = df[df["action"] != "NO_TRADE"]
        long_count = int((df["action"] == "LONG").sum())
        short_count = int((df["action"] == "SHORT").sum())
        total_active = len(active_trades)

        win_rate = float((active_trades["is_win"]).mean()) if total_active > 0 else 0.0
        mean_car = float(df["abnormal_return"].mean())

        # Cumulative returns and drawdown
        strat_returns = active_trades["strategy_return"].values if total_active > 0 else np.array([0.0])
        equity = np.cumprod(1 + strat_returns)
        cum_ret = float(equity[-1] - 1.0)
        max_dd = self.metrics_calc.compute_max_drawdown(equity)
        sharpe = self.metrics_calc.compute_sharpe(strat_returns, periods_per_year=int(252 / holding_period_days))

        # Spearman IC
        ic, p_val = self.metrics_calc.compute_spearman_ic(
            df["signal_score"].tolist(), df["abnormal_return"].tolist()
        )

        metrics = BacktestMetrics(
            total_trades=total_active,
            long_trades=long_count,
            short_trades=short_count,
            win_rate=win_rate,
            information_coefficient=ic,
            ic_p_value=p_val,
            mean_abnormal_return=mean_car,
            cumulative_strategy_return=cum_ret,
            annualized_sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )

        return df, metrics

    def compare_against_baseline(
        self,
        agent_signals: list[EarningsSignal],
        finbert_signals: list[FinBERTSignal],
        acceptance_timestamps: dict[str, str],
        holding_period_days: int = 5,
    ) -> pd.DataFrame:
        """Generate comparative performance matrix between Multi-Agent debate and FinBERT baseline."""
        # Convert FinBERT signals to EarningsSignal structure for evaluation
        fb_as_signals = []
        for fb in finbert_signals:
            action = "LONG" if fb.composite_score > 0.1 else ("SHORT" if fb.composite_score < -0.1 else "NO_TRADE")
            fb_as_signals.append(
                EarningsSignal(
                    ticker=fb.ticker,
                    filing_date=fb.filing_date,
                    primary_catalyst="FinBERT sentiment score",
                    primary_risk="FinBERT risk sentiment score",
                    signal_score=fb.composite_score,
                    confidence=fb.confidence,
                    recommended_action=action,
                    summary_thesis="Baseline ProsusAI/finbert sentiment polarity",
                )
            )

        _, agent_metrics = self.evaluate_signals(
            agent_signals, acceptance_timestamps, holding_period_days=holding_period_days
        )
        _, fb_metrics = self.evaluate_signals(
            fb_as_signals, acceptance_timestamps, holding_period_days=holding_period_days
        )

        comparison = pd.DataFrame(
            [
                {"Model": "Multi-Agent Adversarial Debate", **agent_metrics.to_dict()},
                {"Model": "ProsusAI/finbert Baseline", **fb_metrics.to_dict()},
            ]
        )
        return comparison

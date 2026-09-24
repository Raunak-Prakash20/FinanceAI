from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestMetrics:
    total_trades: int
    long_trades: int
    short_trades: int
    win_rate: float
    information_coefficient: float
    ic_p_value: float
    mean_abnormal_return: float
    cumulative_strategy_return: float
    annualized_sharpe_ratio: float
    max_drawdown: float

    def to_dict(self) -> dict:
        return {
            "Total Trades": self.total_trades,
            "Long Trades": self.long_trades,
            "Short Trades": self.short_trades,
            "Win Rate (%)": round(self.win_rate * 100, 2),
            "Information Coefficient (IC)": round(self.information_coefficient, 4),
            "IC p-value": round(self.ic_p_value, 5),
            "Mean Abnormal Return (%)": round(self.mean_abnormal_return * 100, 2),
            "Cumulative Strategy Return (%)": round(self.cumulative_strategy_return * 100, 2),
            "Annualized Sharpe Ratio": round(self.annualized_sharpe_ratio, 2),
            "Max Drawdown (%)": round(self.max_drawdown * 100, 2),
        }


class MetricsCalculator:
    """Calculates quantitative performance, risk, and rank Information Coefficient metrics."""

    @staticmethod
    def compute_spearman_ic(signals: list[float], returns: list[float]) -> tuple[float, float]:
        """Compute Spearman Rank Information Coefficient (IC) and two-sided p-value."""
        if len(signals) < 3 or len(returns) < 3:
            return 0.0, 1.0

        # Guard against zero variance/constant inputs
        sig_arr = np.array(signals, dtype=float)
        ret_arr = np.array(returns, dtype=float)
        if np.all(np.isclose(sig_arr, sig_arr[0])) or np.all(np.isclose(ret_arr, ret_arr[0])):
            return 0.0, 1.0

        res = stats.spearmanr(sig_arr, ret_arr)
        ic = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        return ic, p_val

    @staticmethod
    def compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """Compute maximum peak-to-trough drawdown from equity series."""
        if len(equity_curve) == 0:
            return 0.0
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peaks) / np.where(peaks == 0, 1.0, peaks)
        return float(np.min(drawdowns))

    @staticmethod
    def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute annualized Sharpe ratio assuming 0% risk-free rate."""
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        return float((mean / std) * np.sqrt(periods_per_year))

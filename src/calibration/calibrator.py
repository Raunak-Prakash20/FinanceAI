import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class SignalCalibrator:
    """Calibrates raw LLM confidence scores to empirical win probabilities using Platt or Isotonic scaling."""

    def __init__(self, sample_threshold: int = 150):
        self.sample_threshold = sample_threshold
        self.model = None
        self.method = "platt"

    def fit(self, raw_confidences: list[float] | np.ndarray, outcomes: list[int] | np.ndarray) -> "SignalCalibrator":
        """Fit calibration mapping raw confidence -> realized win rate (0 or 1)."""
        x = np.array(raw_confidences, dtype=float).reshape(-1, 1)
        y = np.array(outcomes, dtype=int)

        n_samples = len(x)
        if n_samples < 5:
            logger.warning("Insufficient samples (%d) for calibration; using identity.", n_samples)
            self.model = None
            return self

        # Use Platt scaling (logistic) for small samples to prevent step-function overfitting
        if n_samples < self.sample_threshold:
            self.method = "platt"
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
            self.model.fit(x, y)
            logger.info("Fitted Platt scaling (logistic) on %d samples.", n_samples)
        else:
            self.method = "isotonic"
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(x.flatten(), y)
            logger.info("Fitted Isotonic Regression on %d samples.", n_samples)

        return self

    def calibrate(self, raw_confidences: list[float] | np.ndarray) -> np.ndarray:
        """Map raw confidences to calibrated empirical probabilities."""
        if self.model is None:
            return np.array(raw_confidences, dtype=float)

        x = np.array(raw_confidences, dtype=float)
        if self.method == "platt":
            # Probability of win class (1)
            probs = self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            probs = self.model.predict(x.flatten())

        return np.clip(probs, 0.05, 0.95)

    def walk_forward_calibrate(
        self,
        events_df: pd.DataFrame,
        time_col: str = "filing_date",
        split_ratio: float = 0.60,
    ) -> pd.DataFrame:
        """Perform out-of-time walk-forward calibration preserving temporal sequence."""
        if events_df.empty or len(events_df) < 5:
            events_df["calibrated_confidence"] = events_df.get("confidence", 0.5)
            return events_df

        df_sorted = events_df.sort_values(by=time_col).copy()
        split_idx = int(len(df_sorted) * split_ratio)

        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:].copy()

        # Fit on prior period
        train_x = train_df["confidence"].values
        train_y = (train_df["strategy_return"] > 0).astype(int).values
        self.fit(train_x, train_y)

        # Calibrate on future period
        test_df["calibrated_confidence"] = self.calibrate(test_df["confidence"].values)
        train_df["calibrated_confidence"] = self.calibrate(train_df["confidence"].values)

        return pd.concat([train_df, test_df])

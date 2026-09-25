import numpy as np
import pandas as pd
import pytest
from src.calibration.calibrator import SignalCalibrator


def test_signal_calibrator_platt_and_isotonic():
    calibrator = SignalCalibrator(sample_threshold=20)

    # Small sample test (Platt scaling / logistic)
    confidences = np.array([0.55, 0.60, 0.70, 0.85, 0.90, 0.65, 0.80, 0.75])
    outcomes = np.array([0, 0, 1, 1, 1, 0, 1, 1])

    calibrator.fit(confidences, outcomes)
    assert calibrator.method == "platt"

    calibrated = calibrator.calibrate(np.array([0.60, 0.85]))
    assert len(calibrated) == 2
    assert 0.05 <= calibrated[0] <= 0.95
    assert calibrated[1] > calibrated[0]  # Monotonic ordering


def test_walk_forward_calibration():
    calibrator = SignalCalibrator()

    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"],
        "filing_date": ["2023-01-15", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        "confidence": [0.65, 0.70, 0.80, 0.60, 0.85, 0.90],
        "strategy_return": [0.02, 0.04, -0.01, -0.02, 0.05, 0.03],
    })

    result_df = calibrator.walk_forward_calibrate(df, split_ratio=0.5)
    assert "calibrated_confidence" in result_df.columns
    assert len(result_df) == 6

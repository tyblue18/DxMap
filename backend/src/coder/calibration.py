"""Confidence calibration via isotonic regression.

Self-reported LLM confidence is systematically miscalibrated — usually
overconfident. This module fits a simple isotonic regression on the eval set
to map raw confidences to calibrated ones, then persists the calibrator so
inference can use it.

Why isotonic regression: it's monotonic (higher input -> higher output) and
makes no parametric assumption about the shape of miscalibration. For small
eval sets (50-200 examples) it usually outperforms Platt scaling.

Limitations:
- Calibration is fit on the eval set, which means the reported calibration
  numbers in the README are optimistic. A production system would use a
  held-out calibration split, separate from final test.
- Isotonic regression with very few points can overfit. With <50 examples
  per confidence bucket, prefer Platt scaling (sklearn LogisticRegression).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

CALIBRATOR_PATH = Path(__file__).resolve().parents[2] / ".calibrator.pkl"


def fit_calibrator(raw_confidences: list[float], correctness: list[int]) -> IsotonicRegression:
    """Fit an isotonic regression mapping raw_confidence -> P(correct).

    Args:
        raw_confidences: LLM-reported confidence for each prediction
        correctness: 1 if the prediction was correct (gold code), 0 otherwise
    """
    if len(raw_confidences) < 10:
        raise ValueError(
            f"Need at least 10 examples to fit a calibrator; got {len(raw_confidences)}"
        )

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_confidences), np.array(correctness))
    return iso


def save_calibrator(iso: IsotonicRegression, path: Path = CALIBRATOR_PATH) -> None:
    with path.open("wb") as f:
        pickle.dump(iso, f)


def load_calibrator(path: Path = CALIBRATOR_PATH) -> IsotonicRegression | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def calibrate(raw_confidence: float, calibrator: IsotonicRegression | None = None) -> float:
    """Apply the calibrator. Returns raw confidence if no calibrator is loaded."""
    if calibrator is None:
        calibrator = load_calibrator()
    if calibrator is None:
        return raw_confidence
    return float(calibrator.predict(np.array([raw_confidence]))[0])


def expected_calibration_error(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 10,
) -> float:
    """Compute ECE — the standard miscalibration metric.

    Bins predictions by confidence, computes |avg_confidence - accuracy| per bin,
    weights by bin size. Returns a number in [0, 1]; lower is better.
    """
    confidences_arr = np.array(confidences)
    correctness_arr = np.array(correctness)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    n = len(confidences_arr)
    for i in range(n_bins):
        in_bin = (confidences_arr >= bin_edges[i]) & (confidences_arr < bin_edges[i + 1])
        if i == n_bins - 1:  # Include the right edge in the last bin
            in_bin = (confidences_arr >= bin_edges[i]) & (confidences_arr <= bin_edges[i + 1])
        bin_n = in_bin.sum()
        if bin_n == 0:
            continue
        avg_conf = confidences_arr[in_bin].mean()
        avg_acc = correctness_arr[in_bin].mean()
        ece += (bin_n / n) * abs(avg_conf - avg_acc)
    return float(ece)

"""Transparent diminishing-returns scenario used for planning, not a result.

The curve below encodes a planning hypothesis: most of the relative gain in MAE
arrives by 80 to 100 training cases, and 100 to 1,000 adds only 1.5%. Only one
checkpoint is versioned in this project, so this is a projection anchored on the
measured holdout MAE and never a measured learning curve. Both the dashboard and
the article figure read it from here so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np

LEARNING_CURVE_KEY_COUNTS = (13, 20, 30, 40, 50, 60, 80, 100, 150, 250, 500, 1000)
LEARNING_CURVE_MAX_CASES = 1000
LEARNING_CURVE_STRONG_RETURN_CASES = 80
LEARNING_CURVE_CUTOFF_CASES = 100
LEARNING_CURVE_STRONG_GAIN = 0.15
LEARNING_CURVE_CUTOFF_GAIN = 0.18
LEARNING_CURVE_TAIL_GAIN = 0.015


def log_progress(value: float, start: float, end: float) -> float:
    """Map a case count onto 0--1 along a logarithmic span."""

    if end <= start:
        return 1.0
    return float(np.clip(np.log(value / start) / np.log(end / start), 0.0, 1.0))


def theoretical_training_mae(
    case_counts: object,
    *,
    anchor_cases: int,
    anchor_mae: float,
) -> np.ndarray:
    """Return the planning scenario, never a measured learning curve."""

    counts = np.asarray(case_counts, dtype=float)
    values = np.full(counts.shape, np.nan, dtype=float)
    finite = np.isfinite(counts) & (counts > 0)
    if not finite.any() or not np.isfinite(anchor_mae) or anchor_mae <= 0:
        return values

    anchor = max(float(anchor_cases), 1.0)
    strong_return = max(float(LEARNING_CURVE_STRONG_RETURN_CASES), anchor + 1.0)
    cutoff = max(float(LEARNING_CURVE_CUTOFF_CASES), strong_return + 1.0)
    tail_end = max(float(LEARNING_CURVE_MAX_CASES), cutoff + 1.0)
    for index in np.flatnonzero(finite):
        count = counts[index]
        if count <= anchor:
            values[index] = anchor_mae
            continue
        if count <= strong_return:
            progress = log_progress(count, anchor, strong_return)
            values[index] = anchor_mae * (1.0 - LEARNING_CURVE_STRONG_GAIN * progress)
            continue
        if count <= cutoff:
            progress = log_progress(count, strong_return, cutoff)
            gain = LEARNING_CURVE_STRONG_GAIN + (
                LEARNING_CURVE_CUTOFF_GAIN - LEARNING_CURVE_STRONG_GAIN
            ) * progress
            values[index] = anchor_mae * (1.0 - gain)
            continue
        progress = log_progress(count, cutoff, tail_end)
        cutoff_mae = anchor_mae * (1.0 - LEARNING_CURVE_CUTOFF_GAIN)
        values[index] = cutoff_mae * (1.0 - LEARNING_CURVE_TAIL_GAIN * progress)
    return values

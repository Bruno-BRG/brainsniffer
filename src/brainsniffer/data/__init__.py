"""Dataset acquisition, MATLAB reading, and EEG window preparation."""

from .mat_reader import EEGCase, load_case
from .preprocess import (
    WindowedEEG,
    bis_stage,
    load_windows,
    make_windows,
    preprocess_window,
)

__all__ = [
    "EEGCase",
    "WindowedEEG",
    "bis_stage",
    "load_case",
    "load_windows",
    "make_windows",
    "preprocess_window",
]

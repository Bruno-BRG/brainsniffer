"""Shared configuration and domain definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIGSHARE_ARTICLE_ID = 5589841
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
FIGSHARE_DOI = "10.6084/m9.figshare.5589841.v1"
DEFAULT_SAMPLING_RATE = 128
DEFAULT_LABEL_INTERVAL_SECONDS = 5.0
DEFAULT_MIN_SIGNAL_QUALITY = 0.2


@dataclass(frozen=True)
class PreprocessConfig:
    """Signal-processing settings shared by training and replay inference."""

    sampling_rate: int = DEFAULT_SAMPLING_RATE
    window_seconds: float = DEFAULT_LABEL_INTERVAL_SECONDS
    lowcut_hz: float = 0.5
    highcut_hz: float = 45.0
    notch_hz: float | None = 50.0
    notch_quality: float = 30.0
    filter_order: int = 4
    amplitude_scale_uv: float = 50.0
    clip_uv: float = 100.0
    causal: bool = True
    label_offset_seconds: float = 0.0

    @property
    def window_samples(self) -> int:
        return int(round(self.sampling_rate * self.window_seconds))


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible defaults for the baseline CNN experiment."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` without importing torch at module import time."""

    if requested != "auto":
        return requested

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def default_data_dir() -> Path:
    return Path("data/raw")


def default_model_path() -> Path:
    return Path("models/brainsniffer_cnn.pt")

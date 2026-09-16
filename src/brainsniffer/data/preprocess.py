"""EEG cleaning, quality scoring, BIS staging, and window generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from ..config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig
from .mat_reader import EEGCase, load_case

STAGES = ("deep", "general", "light", "awake")


@dataclass(frozen=True)
class WindowedEEG:
    """Preprocessed windows and their aligned reference values."""

    signals: np.ndarray
    bis: np.ndarray
    case_ids: np.ndarray
    start_seconds: np.ndarray
    quality: np.ndarray
    group_ids: np.ndarray | None = None
    source_datasets: np.ndarray | None = None


def bis_stage(value: float) -> str:
    """Map a valid BIS reference to the four-stage research taxonomy."""

    if not np.isfinite(value) or value < 0 or value > 100:
        return "invalid"
    if value < 40:
        return "deep"
    if value < 60:
        return "general"
    if value < 80:
        return "light"
    return "awake"


def signal_quality(signal: np.ndarray, config: PreprocessConfig) -> float:
    """Return a conservative 0--1 quality heuristic for display and filtering.

    This is not a clinical SQI and is intentionally exposed as a diagnostic flag.
    """

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    finite = np.isfinite(signal)
    if not finite.any():
        return 0.0
    safe = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    finite_fraction = float(finite.mean())
    saturation_fraction = float((np.abs(safe) >= config.clip_uv * 0.98).mean())
    differences = np.abs(np.diff(safe)) if safe.size > 1 else np.array([0.0])
    flatline_fraction = float((differences < 1e-5).mean())
    flatline_penalty = min(flatline_fraction * 3.0, 1.0)
    saturation_penalty = min(saturation_fraction * 2.0, 1.0)
    return float(
        np.clip(finite_fraction * (1.0 - saturation_penalty) * (1.0 - flatline_penalty), 0.0, 1.0)
    )


def signal_diagnostics(
    signal: np.ndarray,
    config: PreprocessConfig | None = None,
) -> dict[str, float | int | bool | None]:
    """Summarize raw-signal integrity without retaining the signal itself."""

    config = config or PreprocessConfig()
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "sample_count": int(array.size),
        "nonfinite_count": int((~finite).sum()),
        "finite_fraction": float(finite.mean()) if array.size else 0.0,
        "raw_min": float(finite_values.min()) if finite_values.size else None,
        "raw_max": float(finite_values.max()) if finite_values.size else None,
        "quality": signal_quality(array, config),
        "imputation_applied_offline": bool((~finite).any()),
    }


def data_handling_policy() -> dict[str, str | bool]:
    """Return the explicit boundary between offline analysis and live input."""

    return {
        "mode": "offline_evaluation",
        "nonfinite_samples": "linear_interpolation_during_window_construction",
        "online_nonfinite_policy": "reject_before_filter_or_resampler",
        "raw_eeg_in_report": False,
    }


def _fill_nonfinite(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return np.zeros_like(signal)
    indices = np.arange(signal.size)
    return np.interp(indices, indices[finite], signal[finite])


def preprocess_window(
    signal: np.ndarray,
    config: PreprocessConfig | None = None,
    *,
    realtime: bool = False,
) -> np.ndarray:
    """Filter and scale a single EEG window.

    Offline analysis uses zero-phase filtering unless ``realtime=True``. The
    production-style replay/acquisition path uses ``StreamingPreprocessor``
    instead, which carries causal filter state across chunks.
    """

    config = config or PreprocessConfig()
    cleaned = _fill_nonfinite(signal)
    cleaned = cleaned - np.median(cleaned)
    cleaned = np.clip(cleaned, -config.clip_uv, config.clip_uv)

    nyquist = config.sampling_rate / 2.0
    highcut = min(config.highcut_hz, nyquist * 0.95)
    if 0 < config.lowcut_hz < highcut:
        sos = butter(
            config.filter_order,
            [config.lowcut_hz, highcut],
            btype="bandpass",
            fs=config.sampling_rate,
            output="sos",
        )
        filter_fn = sosfilt if realtime else sosfiltfilt
        try:
            cleaned = filter_fn(sos, cleaned)
        except ValueError:
            cleaned = sosfilt(sos, cleaned)

    if config.notch_hz is not None and 0 < config.notch_hz < nyquist:
        b, a = iirnotch(config.notch_hz, config.notch_quality, fs=config.sampling_rate)
        notch_sos = tf2sos(b, a)
        filter_fn = sosfilt if realtime else sosfiltfilt
        try:
            cleaned = filter_fn(notch_sos, cleaned)
        except ValueError:
            cleaned = sosfilt(notch_sos, cleaned)

    normalized = np.clip(cleaned / config.amplitude_scale_uv, -5.0, 5.0)
    return np.asarray(normalized, dtype=np.float32)


class StreamingPreprocessor:
    """Causal EEG preprocessing with filter state carried across chunks."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        nyquist = self.config.sampling_rate / 2.0
        highcut = min(self.config.highcut_hz, nyquist * 0.95)
        self._band_sos = butter(
            self.config.filter_order,
            [self.config.lowcut_hz, highcut],
            btype="bandpass",
            fs=self.config.sampling_rate,
            output="sos",
        )
        self._band_state = sosfilt_zi(self._band_sos) * 0.0
        self._notch_sos = None
        self._notch_state = None
        if self.config.notch_hz is not None and 0 < self.config.notch_hz < nyquist:
            b, a = iirnotch(
                self.config.notch_hz,
                self.config.notch_quality,
                fs=self.config.sampling_rate,
            )
            self._notch_sos = tf2sos(b, a)
            self._notch_state = sosfilt_zi(self._notch_sos) * 0.0

    def process(self, samples: np.ndarray | list[float]) -> np.ndarray:
        """Process a chunk without looking ahead or resetting filter state."""

        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)
        cleaned = np.clip(_fill_nonfinite(samples), -self.config.clip_uv, self.config.clip_uv)
        filtered, self._band_state = sosfilt(self._band_sos, cleaned, zi=self._band_state)
        if self._notch_sos is not None and self._notch_state is not None:
            filtered, self._notch_state = sosfilt(
                self._notch_sos,
                filtered,
                zi=self._notch_state,
            )
        return np.asarray(
            np.clip(filtered / self.config.amplitude_scale_uv, -5.0, 5.0),
            dtype=np.float32,
        )


def make_windows(
    case: EEGCase,
    config: PreprocessConfig | None = None,
    *,
    stride_seconds: float | None = None,
    min_quality: float = DEFAULT_MIN_SIGNAL_QUALITY,
) -> WindowedEEG:
    """Create aligned EEG/BIS windows without crossing case boundaries."""

    config = config or PreprocessConfig(sampling_rate=case.sampling_rate)
    if config.sampling_rate != case.sampling_rate:
        raise ValueError("A taxa de amostragem da configuração deve coincidir com a do caso")
    if not np.isfinite(config.label_offset_seconds):
        raise ValueError("label_offset_seconds deve ser finito")
    stride_seconds = stride_seconds if stride_seconds is not None else config.window_seconds
    window_samples = config.window_samples
    stride_samples = int(round(stride_seconds * config.sampling_rate))
    if stride_samples <= 0:
        raise ValueError("stride_seconds deve ser positivo")

    causal_preprocessor = StreamingPreprocessor(config) if config.causal else None
    causal_signal = causal_preprocessor.process(case.eeg) if causal_preprocessor else None
    starts = range(0, max(case.eeg.size - window_samples + 1, 0), stride_samples)
    signals: list[np.ndarray] = []
    labels: list[float] = []
    case_ids: list[str] = []
    group_ids: list[str] = []
    source_datasets: list[str] = []
    start_seconds: list[float] = []
    qualities: list[float] = []
    for start in starts:
        label_time = start / case.sampling_rate + config.label_offset_seconds
        label_index = int(round(label_time / case.label_interval_seconds))
        if label_index < 0:
            continue
        if label_index >= case.bis.size:
            break
        label = float(case.bis[label_index])
        if bis_stage(label) == "invalid":
            continue
        raw_window = case.eeg[start : start + window_samples]
        quality = signal_quality(raw_window, config)
        if quality < min_quality:
            continue
        processed_window = (
            causal_signal[start : start + window_samples]
            if causal_signal is not None
            else preprocess_window(raw_window, config)
        )
        signals.append(processed_window)
        labels.append(label)
        case_ids.append(case.case_id)
        group_ids.append(case.group_id or case.case_id)
        source_datasets.append(case.source_dataset or "unknown")
        start_seconds.append(start / case.sampling_rate)
        qualities.append(quality)

    shape = (0, 1, window_samples)
    stacked = (
        np.stack(signals).astype(np.float32)[:, None, :]
        if signals
        else np.empty(shape, dtype=np.float32)
    )
    return WindowedEEG(
        signals=stacked,
        bis=np.asarray(labels, dtype=np.float32),
        case_ids=np.asarray(case_ids),
        start_seconds=np.asarray(start_seconds, dtype=np.float32),
        quality=np.asarray(qualities, dtype=np.float32),
        group_ids=np.asarray(group_ids),
        source_datasets=np.asarray(source_datasets),
    )


def _concat_windows(items: list[WindowedEEG], window_samples: int) -> WindowedEEG:
    if not items:
        return WindowedEEG(
            signals=np.empty((0, 1, window_samples), dtype=np.float32),
            bis=np.empty(0, dtype=np.float32),
            case_ids=np.empty(0, dtype=str),
            start_seconds=np.empty(0, dtype=np.float32),
            quality=np.empty(0, dtype=np.float32),
            group_ids=np.empty(0, dtype=str),
            source_datasets=np.empty(0, dtype=str),
        )
    group_ids = [
        item.group_ids if item.group_ids is not None else item.case_ids for item in items
    ]
    source_datasets = [
        (
            item.source_datasets
            if item.source_datasets is not None
            else np.full(item.case_ids.shape, "unknown", dtype=str)
        )
        for item in items
    ]
    return WindowedEEG(
        signals=np.concatenate([item.signals for item in items], axis=0),
        bis=np.concatenate([item.bis for item in items]),
        case_ids=np.concatenate([item.case_ids for item in items]),
        start_seconds=np.concatenate([item.start_seconds for item in items]),
        quality=np.concatenate([item.quality for item in items]),
        group_ids=np.concatenate(group_ids),
        source_datasets=np.concatenate(source_datasets),
    )


def load_windows(
    paths: list[str | Path],
    config: PreprocessConfig | None = None,
    *,
    min_quality: float = DEFAULT_MIN_SIGNAL_QUALITY,
) -> WindowedEEG:
    """Load and concatenate independent windows from a list of case files."""

    config = config or PreprocessConfig()
    windows = [
        make_windows(
            load_case(path, sampling_rate=config.sampling_rate), config, min_quality=min_quality
        )
        for path in paths
    ]
    return _concat_windows(windows, config.window_samples)


def subset_windows(
    windows: WindowedEEG,
    max_windows: int | None,
    *,
    seed: int = 42,
) -> WindowedEEG:
    """Limit a dataset while retaining at least one window per case.

    This is intended for smoke tests. It deliberately preserves case groups so
    the subsequent train/validation/test split remains meaningful.
    """

    if max_windows is None or max_windows >= windows.signals.shape[0]:
        return windows
    unique_cases = np.asarray(sorted({str(item) for item in windows.case_ids}))
    if max_windows < unique_cases.size:
        raise ValueError("max_windows deve ser pelo menos o número de casos")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for case_id in unique_cases:
        candidates = np.flatnonzero(windows.case_ids.astype(str) == case_id)
        selected.append(int(rng.choice(candidates)))
    remaining = np.setdiff1d(np.arange(windows.signals.shape[0]), np.asarray(selected))
    extra_count = max_windows - len(selected)
    if extra_count:
        selected.extend(
            rng.choice(remaining, size=min(extra_count, remaining.size), replace=False).tolist()
        )
    indices = np.asarray(sorted(selected[:max_windows]))
    return WindowedEEG(
        signals=windows.signals[indices],
        bis=windows.bis[indices],
        case_ids=windows.case_ids[indices],
        start_seconds=windows.start_seconds[indices],
        quality=windows.quality[indices],
        group_ids=(
            windows.group_ids[indices]
            if windows.group_ids is not None
            else None
        ),
        source_datasets=(
            windows.source_datasets[indices]
            if windows.source_datasets is not None
            else None
        ),
    )

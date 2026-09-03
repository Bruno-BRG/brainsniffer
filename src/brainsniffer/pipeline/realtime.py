"""Low-latency replay/inference primitives for a streaming EEG adapter."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from ..config import PreprocessConfig
from ..data.mat_reader import EEGCase
from ..data.preprocess import StreamingPreprocessor, bis_stage, signal_quality


@dataclass(frozen=True)
class RealtimePrediction:
    sample_index: int
    elapsed_seconds: float
    raw_bis: float | None
    smoothed_bis: float | None
    stage: str
    quality: float
    source_timestamp: float | None = None


class RealtimeEstimator:
    """Buffer one channel of EEG and emit a prediction every stride."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: PreprocessConfig | None = None,
        *,
        stride_seconds: float = 1.0,
        smoothing_alpha: float = 0.25,
        min_quality: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.config = config or PreprocessConfig()
        self.device = device
        self.window_samples = self.config.window_samples
        self.stride_samples = max(1, int(round(stride_seconds * self.config.sampling_rate)))
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha deve estar em (0, 1]")
        if not 0 <= min_quality <= 1:
            raise ValueError("min_quality deve estar entre 0 e 1")
        self.smoothing_alpha = smoothing_alpha
        self.min_quality = min_quality
        self._raw_buffer: deque[float] = deque(maxlen=self.window_samples)
        self._processed_buffer: deque[float] = deque(maxlen=self.window_samples)
        self._stream_preprocessor = StreamingPreprocessor(self.config)
        self._samples_seen = 0
        self._samples_since_prediction = 0
        self._smoothed: float | None = None
        self._last_source_timestamp: float | None = None

    @torch.inference_mode()
    def _predict_buffer(self) -> RealtimePrediction:
        raw = np.asarray(self._raw_buffer, dtype=np.float32)
        quality = signal_quality(raw, self.config)
        if quality < self.min_quality:
            # Do not carry a stale BIS value across a poor-signal interval.
            # The next valid window starts a fresh smoother state.
            self._smoothed = None
            return RealtimePrediction(
                sample_index=self._samples_seen,
                elapsed_seconds=self._samples_seen / self.config.sampling_rate,
                raw_bis=None,
                smoothed_bis=None,
                stage="abstain",
                quality=quality,
                source_timestamp=self._last_source_timestamp,
            )
        processed = np.asarray(self._processed_buffer, dtype=np.float32)
        tensor = torch.from_numpy(processed[None, None, :]).float().to(self.device)
        raw_bis = float(torch.clamp(self.model(tensor), 0.0, 100.0).detach().cpu().item())
        if self._smoothed is None:
            self._smoothed = raw_bis
        else:
            self._smoothed = (
                self.smoothing_alpha * raw_bis + (1.0 - self.smoothing_alpha) * self._smoothed
            )
        return RealtimePrediction(
            sample_index=self._samples_seen,
            elapsed_seconds=self._samples_seen / self.config.sampling_rate,
            raw_bis=raw_bis,
            smoothed_bis=float(self._smoothed),
            stage=bis_stage(float(self._smoothed)),
            quality=quality,
            source_timestamp=self._last_source_timestamp,
        )

    def push(
        self,
        samples: np.ndarray | list[float],
        timestamps: np.ndarray | list[float] | None = None,
    ) -> list[RealtimePrediction]:
        """Push samples and return zero or more predictions."""

        samples_array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples_array.size and not np.isfinite(samples_array).all():
            # Offline preprocessing may impute missing values for dataset
            # inspection, but a live stream must fail closed before touching
            # the causal filter state or emitting a misleading prediction.
            raise ValueError("samples devem ser finitas no modo streaming")
        timestamp_array = (
            None if timestamps is None else np.asarray(timestamps, dtype=np.float64).reshape(-1)
        )
        if timestamp_array is not None and timestamp_array.size == 0 and samples_array.size:
            timestamp_array = None
        if timestamp_array is not None and timestamp_array.size != samples_array.size:
            raise ValueError("timestamps deve ter o mesmo número de elementos que samples")
        if timestamp_array is not None and timestamp_array.size:
            if not np.isfinite(timestamp_array).all():
                raise ValueError("timestamps devem ser finitos")
            if np.any(np.diff(timestamp_array) <= 0):
                raise ValueError("timestamps devem ser estritamente crescentes")
            if (
                self._last_source_timestamp is not None
                and timestamp_array[0] <= self._last_source_timestamp
            ):
                raise ValueError("timestamps devem ser estritamente crescentes")
        processed_array = self._stream_preprocessor.process(samples_array)
        predictions: list[RealtimePrediction] = []
        for index, sample in enumerate(samples_array):
            if timestamp_array is not None and np.isfinite(timestamp_array[index]):
                self._last_source_timestamp = float(timestamp_array[index])
            self._raw_buffer.append(float(sample))
            self._processed_buffer.append(float(processed_array[index]))
            self._samples_seen += 1
            self._samples_since_prediction += 1
            if len(self._raw_buffer) == self.window_samples and (
                self._samples_seen == self.window_samples
                or self._samples_since_prediction >= self.stride_samples
            ):
                predictions.append(self._predict_buffer())
                self._samples_since_prediction = 0
        return predictions

    def mark_stale(self) -> RealtimePrediction:
        """Invalidate the last estimate after a source-silence interval.

        A resumed stream must fill a fresh window. Keeping samples and filter
        state across an unobserved interval could blend two disconnected
        segments into a plausible but physically invalid prediction.
        """

        self._raw_buffer.clear()
        self._processed_buffer.clear()
        self._stream_preprocessor = StreamingPreprocessor(self.config)
        self._samples_since_prediction = 0
        self._smoothed = None
        return RealtimePrediction(
            sample_index=self._samples_seen,
            elapsed_seconds=self._samples_seen / self.config.sampling_rate,
            raw_bis=None,
            smoothed_bis=None,
            stage="abstain",
            quality=0.0,
            source_timestamp=self._last_source_timestamp,
        )


def replay_case(
    model: torch.nn.Module,
    case: EEGCase,
    config: PreprocessConfig | None = None,
    *,
    stride_seconds: float = 1.0,
    smoothing_alpha: float = 0.25,
    min_quality: float = 0.2,
    device: str = "cpu",
) -> list[RealtimePrediction]:
    """Replay a recorded case as if it arrived in streaming chunks."""

    estimator = RealtimeEstimator(
        model,
        config or PreprocessConfig(sampling_rate=case.sampling_rate),
        stride_seconds=stride_seconds,
        smoothing_alpha=smoothing_alpha,
        min_quality=min_quality,
        device=device,
    )
    chunk_size = max(1, int(round(stride_seconds * case.sampling_rate)))
    predictions: list[RealtimePrediction] = []
    for start in range(0, case.eeg.size, chunk_size):
        predictions.extend(estimator.push(case.eeg[start : start + chunk_size]))
    return predictions

"""Small runtime benchmarks for the streaming inference path."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..config import PreprocessConfig
from .realtime import RealtimeEstimator


@dataclass(frozen=True)
class LatencyResult:
    iterations: int
    prediction_p50_ms: float
    prediction_p95_ms: float
    chunk_p50_ms: float
    chunk_p95_ms: float
    realtime_budget_fraction_p95: float


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark_latency(
    model: torch.nn.Module,
    config: PreprocessConfig | None = None,
    *,
    iterations: int = 30,
    stride_seconds: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> LatencyResult:
    """Measure CPU/GPU wall time for chunk processing and emitted predictions.

    The budget fraction compares p95 prediction time with the configured
    stride interval. It is a software benchmark, not a clinical performance
    guarantee.
    """

    if iterations <= 0:
        raise ValueError("iterations deve ser positivo")
    config = config or PreprocessConfig()
    rng = np.random.default_rng(seed)
    prediction_times: list[float] = []
    chunk_times: list[float] = []
    stride_samples = max(1, int(round(stride_seconds * config.sampling_rate)))
    for _ in range(iterations):
        estimator = RealtimeEstimator(
            model,
            config,
            stride_seconds=stride_seconds,
            min_quality=0.0,
            device=device,
        )
        samples = rng.normal(0.0, 8.0, config.window_samples + stride_samples).astype(np.float32)
        for start in range(0, samples.size, stride_samples):
            chunk = samples[start : start + stride_samples]
            started = time.perf_counter()
            outputs = estimator.push(chunk)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            chunk_times.append(elapsed_ms)
            if outputs:
                prediction_times.append(elapsed_ms)
    prediction_p95 = _percentile(prediction_times, 95)
    return LatencyResult(
        iterations=iterations,
        prediction_p50_ms=_percentile(prediction_times, 50),
        prediction_p95_ms=prediction_p95,
        chunk_p50_ms=_percentile(chunk_times, 50),
        chunk_p95_ms=_percentile(chunk_times, 95),
        realtime_budget_fraction_p95=prediction_p95 / (stride_seconds * 1000.0),
    )

"""Synthetic JSONL integration checks; no checkpoint files or physical device."""
import io
import json

import numpy as np
import pytest

from brainsniffer import cli
from brainsniffer.config import PreprocessConfig
from brainsniffer.pipeline.streaming import StreamingResampler


class Recorder:
    def __init__(self, *args, **kwargs):
        self.segments = [[]]

    def push(self, samples, timestamps=None):
        self.segments[-1].extend(np.asarray(samples).tolist())
        return []

    def mark_stale(self):
        self.segments.append([])
        from brainsniffer.pipeline.realtime import RealtimePrediction
        return RealtimePrediction(
            sample_index=0, elapsed_seconds=0, raw_bis=None, smoothed_bis=None,
            stage="abstain", quality=0,
        )


def test_gap_discards_pending_resampler_context():
    resampler = StreamingResampler(256, 128)
    estimator = Recorder()
    old = np.ones(40) * 100
    list(cli._push_stream_segments(estimator, resampler, old, np.arange(40)/256,
                                  previous_timestamp=None, max_gap_factor=1.5))
    new = np.sin(np.arange(128))
    times = 10 + np.arange(128)/256
    list(cli._push_stream_segments(estimator, resampler, new, times,
                                  previous_timestamp=39/256, max_gap_factor=1.5))
    reference = StreamingResampler(256, 128).process(new, timestamps=times)
    np.testing.assert_allclose(estimator.segments[-1], reference.samples)
    assert len(estimator.segments) == 2


def test_internal_gap_splits_before_conversion():
    estimator = Recorder()
    list(cli._push_stream_segments(estimator, StreamingResampler(128, 128),
                                  [1, 2, 3, 4], [0, 1/128, 10, 10+1/128],
                                  previous_timestamp=None, max_gap_factor=1.5))
    assert estimator.segments == [[1, 2], [3, 4]]


@pytest.mark.parametrize("unit,rejected", [("mV", True), ("V", True), ("µV", False), (None, False)])
def test_json_unit_gate_without_optional_flags(monkeypatch, tmp_path, unit, rejected):
    estimator = Recorder()
    monkeypatch.setattr(
        "brainsniffer.pipeline.realtime.RealtimeEstimator", lambda *a, **k: estimator
    )
    monkeypatch.setattr(cli, "load_checkpoint", lambda p: (None, PreprocessConfig(), {}))
    monkeypatch.setattr(cli, "_checkpoint_sha256", lambda p: None)
    payload = {"samples": [1, 2, 3], "sampling_rate": 128}
    if unit is not None:
        payload["metadata"] = {"unit": unit}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    report = tmp_path / "report.json"
    args = ["stream-json", "--report", str(report)]
    if rejected:
        with pytest.raises(ValueError, match="unidade"):
            cli.main(args)
        assert estimator.segments == [[]]
    else:
        assert cli.main(args) == 0
        assert estimator.segments == [[1, 2, 3]]
    assert json.loads(report.read_text())["status"] == ("error" if rejected else "completed")


def test_real_estimator_restarts_window_after_gap():
    import torch

    from brainsniffer.pipeline.realtime import RealtimeEstimator

    class Model(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0)

    config = PreprocessConfig()
    estimator = RealtimeEstimator(Model(), config, min_quality=0)
    resampler = StreamingResampler(128, 128)
    n = config.window_samples
    samples = np.sin(np.arange(n) * 0.3) * 10
    first = list(cli._push_stream_segments(estimator, resampler, samples, np.arange(n)/128,
                                         previous_timestamp=None, max_gap_factor=1.5))
    assert first
    origin = n/128 + 10
    partial = list(cli._push_stream_segments(estimator, resampler, samples[:-1],
                                           origin + np.arange(n-1)/128,
                                           previous_timestamp=(n-1)/128, max_gap_factor=1.5))
    assert partial == []
    outputs = list(cli._push_stream_segments(estimator, resampler, samples[-1:],
                                           [origin+(n-1)/128],
                                           previous_timestamp=origin+(n-2)/128, max_gap_factor=1.5))
    assert len(outputs) == 1
    assert outputs[0].source_timestamp == origin+(n-1)/128
    assert outputs[0].sample_index == 2*n


def test_online_quality_constant_storage():
    stats = cli._OnlineQuality()
    assert stats.mean is stats.minimum is None
    for i in range(10000):
        stats.append(i % 2)
    assert stats.count == 10000
    assert stats.minimum == 0
    assert stats.mean == pytest.approx(0.5)
    assert set(vars(stats)) == {"count", "minimum", "mean"}


def test_json_gap_resets_both_components(monkeypatch):
    estimator = Recorder()
    monkeypatch.setattr(
        "brainsniffer.pipeline.realtime.RealtimeEstimator", lambda *a, **k: estimator
    )
    monkeypatch.setattr(cli, "load_checkpoint", lambda p: (None, PreprocessConfig(), {}))
    monkeypatch.setattr(cli, "_checkpoint_sha256", lambda p: None)
    chunks = [
        {"samples": [100.0] * 40, "timestamps": (np.arange(40) / 256).tolist(),
         "sampling_rate": 256},
        {"samples": [0.0] * 128, "timestamps": (10 + np.arange(128) / 256).tolist(),
         "sampling_rate": 256},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(map(json.dumps, chunks))))
    assert cli.main(["stream-json"]) == 0
    assert len(estimator.segments) == 2
    assert estimator.segments[-1]
    np.testing.assert_allclose(estimator.segments[-1], 0)

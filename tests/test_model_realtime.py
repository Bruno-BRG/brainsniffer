import numpy as np
import pytest
import torch

from brainsniffer.config import PreprocessConfig
from brainsniffer.models.cnn import Conv1DDepthEstimator
from brainsniffer.pipeline.realtime import RealtimeEstimator


def test_cnn_output_is_bounded():
    model = Conv1DDepthEstimator()
    output = model(torch.zeros(4, 1, PreprocessConfig().window_samples))
    assert output.shape == (4,)
    assert torch.all(output >= 0)
    assert torch.all(output <= 100)


def test_realtime_emits_after_window_and_smooths():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    config = PreprocessConfig()
    estimator = RealtimeEstimator(ConstantModel(), config, stride_seconds=1.0)
    samples = np.sin(np.arange(config.window_samples + config.sampling_rate, dtype=np.float32))
    predictions = estimator.push(samples)
    assert len(predictions) == 2
    assert predictions[0].stage == "general"
    assert predictions[-1].smoothed_bis == 55.0


def test_realtime_abstains_on_flat_signal():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    config = PreprocessConfig()
    estimator = RealtimeEstimator(ConstantModel(), config, min_quality=0.2)
    predictions = estimator.push(np.zeros(config.window_samples, dtype=np.float32))
    assert predictions[0].stage == "abstain"
    assert predictions[0].raw_bis is None
    assert predictions[0].smoothed_bis is None


def test_realtime_marks_source_silence_stale_and_resets_window():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    config = PreprocessConfig()
    estimator = RealtimeEstimator(ConstantModel(), config)
    valid = estimator.push(np.sin(np.arange(config.window_samples, dtype=np.float32)))
    assert valid[0].stage == "general"

    stale = estimator.mark_stale()

    assert stale.stage == "abstain"
    assert stale.raw_bis is None
    assert stale.smoothed_bis is None
    assert stale.quality == 0.0
    assert len(estimator._raw_buffer) == 0
    assert len(estimator._processed_buffer) == 0


def test_realtime_rejects_bad_timestamps_before_advancing_state():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    config = PreprocessConfig()
    estimator = RealtimeEstimator(ConstantModel(), config)
    with pytest.raises(ValueError, match="timestamps"):
        estimator.push(np.ones(4, dtype=np.float32), timestamps=[1.0])
    assert estimator._samples_seen == 0


def test_realtime_rejects_nonmonotonic_timestamps_before_filtering():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    estimator = RealtimeEstimator(ConstantModel(), PreprocessConfig())
    with pytest.raises(ValueError, match="estritamente crescentes"):
        estimator.push([1.0, 2.0], timestamps=[1.0, 1.0])
    assert estimator._samples_seen == 0


def test_realtime_rejects_nonfinite_samples_before_filtering():
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    estimator = RealtimeEstimator(ConstantModel(), PreprocessConfig())
    with pytest.raises(ValueError, match="samples devem ser finitas"):
        estimator.push([1.0, np.nan, 2.0])
    assert estimator._samples_seen == 0

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


@pytest.mark.parametrize("min_quality", [0.0, 0.2])
def test_realtime_abstains_on_flat_signal(min_quality):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    config = PreprocessConfig()
    # min_quality=0 disables a threshold, not the unusable-signal safeguard.
    estimator = RealtimeEstimator(ConstantModel(), config, min_quality=min_quality)
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


class SequenceModel(torch.nn.Module):
    """Safe fake: no checkpoint or training, one configured value per call."""

    def __init__(self, values):
        super().__init__()
        self.values = iter(values)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return torch.full((inputs.shape[0],), next(self.values), device=inputs.device)


def test_realtime_rejects_noncausal_config_before_preparing_model():
    class UntouchedModel(torch.nn.Module):
        def to(self, *args, **kwargs):
            pytest.fail("invalid config must be rejected before model preparation")

    with pytest.raises(ValueError, match="causal=False"):
        RealtimeEstimator(UntouchedModel(), PreprocessConfig(causal=False))


@pytest.mark.parametrize("bad_output", [float("nan"), float("inf"), -float("inf")])
def test_realtime_nonfinite_output_abstains_and_resets_smoothing(bad_output):
    config = PreprocessConfig()
    model = SequenceModel([20.0, bad_output, 80.0, 40.0])
    estimator = RealtimeEstimator(model, config, smoothing_alpha=0.25)
    samples = np.sin(np.arange(config.window_samples, dtype=np.float32))
    first = estimator.push(samples)[0]
    assert first.smoothed_bis == 20.0

    stride = samples[:config.sampling_rate]
    failed = estimator.push(stride)[0]
    assert failed.stage == "abstain"
    assert failed.raw_bis is None
    assert failed.smoothed_bis is None
    assert failed.quality > 0.0
    assert estimator._smoothed is None
    assert estimator.push(stride)[0].smoothed_bis == 80.0
    assert estimator.push(stride)[0].smoothed_bis == 70.0
    assert model.calls == 4


def test_realtime_zero_quality_skips_model_and_resets_smoothing():
    config = PreprocessConfig()
    model = SequenceModel([20.0, 80.0])
    estimator = RealtimeEstimator(
        model, config, stride_seconds=config.window_seconds, min_quality=0.0
    )
    samples = np.sin(np.arange(config.window_samples, dtype=np.float32))
    assert estimator.push(samples)[0].smoothed_bis == 20.0
    failed = estimator.push(np.zeros(config.window_samples, dtype=np.float32))[0]
    assert failed.quality == 0.0
    assert failed.stage == "abstain"
    assert failed.raw_bis is None
    assert failed.smoothed_bis is None
    assert estimator._smoothed is None
    assert model.calls == 1
    assert estimator.push(samples)[0].smoothed_bis == 80.0
    assert model.calls == 2


@pytest.mark.parametrize("offset", [-2.0, 0.0, 2.0])
def test_realtime_emission_timestamps_are_not_shifted_to_label_target(offset):
    config = PreprocessConfig(label_offset_seconds=offset)
    estimator = RealtimeEstimator(SequenceModel([55.0]), config)
    samples = np.sin(np.arange(config.window_samples, dtype=np.float32))
    timestamps = 1000.0 + np.arange(config.window_samples) / config.sampling_rate
    prediction = estimator.push(samples, timestamps)[0]
    assert prediction.sample_index == config.window_samples
    assert prediction.elapsed_seconds == config.window_samples / config.sampling_rate
    assert prediction.source_timestamp == timestamps[-1]
    assert estimator.config.label_offset_seconds == offset

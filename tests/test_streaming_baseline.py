import numpy as np
import pytest
from scipy.signal import resample_poly

from brainsniffer.data.preprocess import WindowedEEG
from brainsniffer.pipeline.baseline import cross_validate_spectral_baseline, spectral_features
from brainsniffer.pipeline.streaming import StreamingResampler, resample_chunk


def test_resample_chunk_reaches_model_rate():
    source = np.sin(np.linspace(0, 4 * np.pi, 256, dtype=np.float32))
    chunk = resample_chunk(source, 256, 128)
    assert chunk.sampling_rate == 128
    assert abs(chunk.samples.size - 128) <= 1


def test_streaming_resampler_preserves_state_and_timestamps():
    source_rate = 256.0
    target_rate = 128.0
    source = np.sin(np.linspace(0, 12 * np.pi, 997, dtype=np.float32))
    source_timestamps = 100.0 + np.arange(source.size, dtype=np.float64) / source_rate
    resampler = StreamingResampler(source_rate, target_rate)
    outputs = []
    timestamp_outputs = []
    for start in range(0, source.size, 37):
        chunk = resampler.process(
            source[start : start + 37],
            timestamps=source_timestamps[start : start + 37],
        )
        outputs.append(chunk.samples)
        timestamp_outputs.append(chunk.timestamps)
    tail = resampler.flush()
    outputs.append(tail.samples)
    timestamp_outputs.append(tail.timestamps)

    converted = np.concatenate(outputs)
    converted_timestamps = np.concatenate(timestamp_outputs)
    expected = resample_poly(source, 1, 2)
    assert converted.size == expected.size
    assert np.isfinite(converted).all()
    assert np.sqrt(np.mean((converted - expected) ** 2)) < 0.1
    assert converted_timestamps.size == converted.size
    assert np.all(np.diff(converted_timestamps) > 0)


def test_streaming_resampler_upsampling_flush_keeps_timestamps_strictly_increasing():
    source_rate = 64.0
    target_rate = 128.0
    source = np.sin(np.linspace(0, 8 * np.pi, 101, dtype=np.float32))
    source_timestamps = 100.0 + np.arange(source.size, dtype=np.float64) / source_rate
    resampler = StreamingResampler(source_rate, target_rate)
    outputs = []
    timestamp_outputs = []
    for start in range(0, source.size, 11):
        chunk = resampler.process(
            source[start : start + 11],
            timestamps=source_timestamps[start : start + 11],
        )
        outputs.append(chunk.samples)
        timestamp_outputs.append(chunk.timestamps)
    tail = resampler.flush()
    outputs.append(tail.samples)
    timestamp_outputs.append(tail.timestamps)

    converted = np.concatenate(outputs)
    converted_timestamps = np.concatenate(timestamp_outputs)
    assert converted.size == int(np.ceil(source.size * target_rate / source_rate))
    assert converted_timestamps.size == converted.size
    assert np.all(np.diff(converted_timestamps) > 0)
    assert converted_timestamps[-1] > source_timestamps[-1]


def test_streaming_resampler_rejects_nonmonotonic_timestamps():
    resampler = StreamingResampler(256, 128)
    with pytest.raises(ValueError, match="estritamente crescentes"):
        resampler.process([1.0, 2.0], timestamps=[1.0, 1.0])

    resampler.process([1.0, 2.0], timestamps=[1.0, 1.01])
    with pytest.raises(ValueError, match="estritamente crescentes"):
        resampler.process([3.0], timestamps=[1.005])


def test_streaming_resampler_rejects_nonfinite_samples_without_state_change():
    resampler = StreamingResampler(256, 128)
    with pytest.raises(ValueError, match="samples devem ser finitas"):
        resampler.process([1.0, np.inf])
    assert resampler._source_seen == 0
    assert resampler._input_buffer.size == 0


@pytest.mark.parametrize("source_rate,target_rate,up,down", [
    (64., 128., 2, 1), (256., 128., 1, 2), (200., 128., 16, 25),
])
@pytest.mark.parametrize("count", [0, 1, 2, 37])
def test_resample_chunk_uses_target_grid(source_rate, target_rate, up, down, count):
    samples = np.arange(count, dtype=np.float32)
    timestamps = 10. + np.arange(count) / source_rate
    chunk = resample_chunk(samples, source_rate, target_rate, timestamps=timestamps)
    expected = resample_poly(samples, up, down) if count else samples
    np.testing.assert_allclose(chunk.samples, expected)
    np.testing.assert_allclose(chunk.timestamps, 10. + np.arange(expected.size) / target_rate)
    assert chunk.timestamps.size == chunk.samples.size
    assert np.all(np.diff(chunk.timestamps) > 0)
    assert chunk.sampling_rate == target_rate


@pytest.mark.parametrize(
    "source_rate,target_rate", [(0, 128), (128, -1), (np.nan, 128), (128, np.inf)]
)
@pytest.mark.parametrize("samples", [[], [1.]])
def test_resample_chunk_rejects_invalid_rates_even_for_empty_input(
    source_rate, target_rate, samples
):
    with pytest.raises(ValueError, match="source_rate e target_rate"):
        resample_chunk(samples, source_rate, target_rate)


@pytest.mark.parametrize("source_rate,target_rate", [(256, 128), (64, 128), (128, 128)])
@pytest.mark.parametrize("with_timestamps", [False, True])
@pytest.mark.parametrize("flush_before_reset", [False, True])
def test_streaming_resampler_reset_matches_fresh_instance(
    source_rate, target_rate, with_timestamps, flush_before_reset
):
    resampler = StreamingResampler(source_rate, target_rate, max_denominator=123)
    old = np.arange(137, dtype=np.float32)
    resampler.process(old, 100. + np.arange(old.size) / source_rate)
    if flush_before_reset:
        resampler.flush()
    configuration = (resampler.source_rate, resampler.target_rate, resampler.up, resampler.down,
                     resampler._history_samples, resampler._holdback_outputs)
    assert resampler.reset() is None
    assert resampler.flush().samples.size == 0
    assert resampler.flush().timestamps.size == 0
    resampler.reset()  # Idempotent; pending output stays discarded.
    assert configuration == (
        resampler.source_rate, resampler.target_rate, resampler.up, resampler.down,
        resampler._history_samples, resampler._holdback_outputs,
    )
    fresh = StreamingResampler(source_rate, target_rate, max_denominator=123)
    new = np.sin(np.arange(89, dtype=np.float32))
    for start in range(0, new.size, 7):
        samples = new[start:start + 7]
        timestamps = (np.arange(start, start + samples.size) / source_rate
                      if with_timestamps else None)
        actual = resampler.process(samples, timestamps)
        expected = fresh.process(samples, timestamps)
        np.testing.assert_array_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
    actual, expected = resampler.flush(), fresh.flush()
    np.testing.assert_array_equal(actual.samples, expected.samples)
    np.testing.assert_array_equal(actual.timestamps, expected.timestamps)


def test_spectral_features_are_finite_and_named():
    signals = np.zeros((2, 1, 640), dtype=np.float32)
    signals[0, 0] = np.sin(np.linspace(0, 10 * np.pi, 640, dtype=np.float32))
    features, names = spectral_features(signals)
    assert features.shape == (2, len(names))
    assert np.isfinite(features).all()


def test_grouped_baseline_exposes_effective_fold_provenance():
    samples = np.arange(640, dtype=np.float32) / 128
    signals = []
    labels = []
    case_ids = []
    for case_index in range(4):
        for window_index in range(3):
            frequency = 2.0 + case_index * 3.0 + window_index
            signals.append(np.sin(2 * np.pi * frequency * samples))
            labels.append(30.0 + case_index * 12.0 + window_index * 3.0)
            case_ids.append(f"case{case_index + 1}")
    windows = WindowedEEG(
        signals=np.asarray(signals, dtype=np.float32)[:, None, :],
        bis=np.asarray(labels, dtype=np.float32),
        case_ids=np.asarray(case_ids),
        start_seconds=np.tile(np.asarray([0.0, 5.0, 10.0], dtype=np.float32), 4),
        quality=np.ones(12, dtype=np.float32),
    )

    result = cross_validate_spectral_baseline(windows, n_splits=2, seed=7)

    assert result.seed == 7
    assert result.sampling_rate == 128
    assert len(result.case_folds) == 2
    assert result.feature_names[0:5] == ("delta", "theta", "alpha", "beta", "gamma")
    assert result.estimator_parameters[0]["random_state"] == 7
    assert result.estimator_parameters[1]["random_state"] == 8
    all_test_cases = [case for fold in result.case_folds for case in fold.test_cases]
    assert sorted(all_test_cases) == ["case1", "case2", "case3", "case4"]
    assert all(set(fold.train_cases).isdisjoint(fold.test_cases) for fold in result.case_folds)

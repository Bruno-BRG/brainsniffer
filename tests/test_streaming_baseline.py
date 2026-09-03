import numpy as np
import pytest
from scipy.signal import resample_poly

from brainsniffer.pipeline.baseline import spectral_features
from brainsniffer.pipeline.streaming import LSLSource, StreamingResampler, resample_chunk


class _FakeInfo:
    def name(self):
        return "Fake EEG"

    def type(self):
        return "EEG"

    def channel_count(self):
        return 2

    def nominal_srate(self):
        return 128


class _FakeInlet:
    def info(self):
        return _FakeInfo()

    def pull_chunk(self, timeout, max_samples):
        return [[1.0, 10.0], [2.0, 20.0]], [1.0, 1.01]


class _FakeXml:
    def __init__(self, *, children=None, values=None, empty=False):
        self._children = children or {}
        self._values = values or {}
        self._empty = empty

    def child(self, name):
        return self._children.get(name, _FakeXml(empty=True))

    def child_value(self, name):
        return self._values.get(name, "")

    def next_sibling(self):
        return _FakeXml(empty=True)

    def empty(self):
        return self._empty


class _MetadataInfo(_FakeInfo):
    def source_id(self):
        return "device-123"

    def desc(self):
        channel = _FakeXml(
            values={
                "label": "Fpz",
                "unit": "uV",
                "reference": "linked ears",
                "montage": "frontal referenced",
            }
        )
        return _FakeXml(children={"channels": _FakeXml(children={"channel": channel})})


class _MetadataInlet(_FakeInlet):
    def info(self):
        return _MetadataInfo()


def test_lsl_source_selects_a_channel_and_preserves_timestamps():
    source = LSLSource(_FakeInlet(), channel_index=1)
    chunk = source.read_chunk()
    assert source.stream_name == "Fake EEG"
    assert source.metadata == {
        "source_name": "Fake EEG",
        "stream_type": "EEG",
        "channel_index": 1,
        "channel_count": 2,
        "sampling_rate": 128.0,
    }
    assert chunk.samples.tolist() == [10.0, 20.0]
    assert chunk.timestamps.tolist() == [1.0, 1.01]


def test_lsl_source_reads_channel_metadata_from_descriptor():
    source = LSLSource(_MetadataInlet(), channel_index=0)

    assert source.metadata["source_id"] == "device-123"
    assert source.metadata["channel_name"] == "Fpz"
    assert source.metadata["unit"] == "uV"
    assert source.metadata["reference"] == "linked ears"
    assert source.metadata["montage"] == "frontal referenced"


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


def test_spectral_features_are_finite_and_named():
    signals = np.zeros((2, 1, 640), dtype=np.float32)
    signals[0, 0] = np.sin(np.linspace(0, 10 * np.pi, 640, dtype=np.float32))
    features, names = spectral_features(signals)
    assert features.shape == (2, len(names))
    assert np.isfinite(features).all()

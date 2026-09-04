"""Vendor-neutral streaming adapters and rate conversion for research replay."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class EEGChunk:
    """A single-channel chunk plus source timestamps."""

    samples: np.ndarray
    timestamps: np.ndarray
    sampling_rate: float
    stream_name: str = ""


def _lsl_channel_metadata(info: object, channel_index: int) -> dict[str, object]:
    """Read common channel fields from an optional LSL XML descriptor."""

    metadata: dict[str, object] = {}
    for method_name, key in (("source_id", "source_id"),):
        method = getattr(info, method_name, None)
        if callable(method):
            try:
                value = str(method()).strip()
            except Exception:
                value = ""
            if value:
                metadata[key] = value

    desc_method = getattr(info, "desc", None)
    if not callable(desc_method):
        return metadata
    try:
        channels = desc_method().child("channels")
        if channels.empty():
            return metadata
        channel = channels.child("channel")
        for _ in range(channel_index):
            channel = channel.next_sibling()
            if channel.empty():
                return metadata
        fields = {
            "label": "channel_name",
            "name": "channel_name",
            "unit": "unit",
            "reference": "reference",
            "montage": "montage",
        }
        for xml_name, key in fields.items():
            value = str(channel.child_value(xml_name)).strip()
            if value and key not in metadata:
                metadata[key] = value
    except Exception:
        # Metadata is helpful but optional; malformed vendor XML must not make
        # the transport parser crash. ``--require-metadata`` still fails later
        # if the required fields remain absent.
        return metadata
    return metadata


class LSLSource:
    """Pull one channel from an LSL EEG stream.

    The optional ``pylsl`` dependency is imported only when a connection is
    requested. An acquisition vendor must expose a compatible LSL outlet or
    provide a separate bridge to it.
    """

    def __init__(self, inlet: object, *, channel_index: int = 0) -> None:
        self.inlet = inlet
        self.channel_index = channel_index
        info = inlet.info()
        self.stream_name = str(info.name())
        self.stream_type = str(info.type())
        self.channel_count = int(info.channel_count())
        self.sampling_rate = float(info.nominal_srate())
        self._descriptor_metadata = _lsl_channel_metadata(info, self.channel_index)
        if self.channel_index < 0 or self.channel_index >= self.channel_count:
            raise ValueError("channel_index está fora da quantidade de canais do stream")
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("O stream LSL precisa declarar uma taxa regular positiva")

    @classmethod
    def connect(
        cls,
        *,
        stream_name: str | None = None,
        stream_type: str = "EEG",
        channel_index: int = 0,
        timeout_seconds: float = 5.0,
    ) -> LSLSource:
        """Discover one stream by name/type and connect to it."""

        try:
            import pylsl
        except ImportError as error:
            raise RuntimeError(
                "Modo LSL requer a dependência opcional: uv sync --extra live"
            ) from error

        streams = pylsl.resolve_streams(wait_time=timeout_seconds)
        candidates = [
            info
            for info in streams
            if (not stream_name or str(info.name()) == stream_name)
            and (not stream_type or str(info.type()) == stream_type)
        ]
        if not candidates:
            filters = []
            if stream_name:
                filters.append(f"name={stream_name!r}")
            if stream_type:
                filters.append(f"type={stream_type!r}")
            query = ", ".join(filters) or "qualquer stream"
            raise RuntimeError(f"Nenhum stream LSL encontrado para {query}")
        inlet = pylsl.StreamInlet(candidates[0], max_buflen=5, recover=True)
        return cls(inlet, channel_index=channel_index)

    def read_chunk(self, *, timeout_seconds: float = 0.2, max_samples: int = 256) -> EEGChunk:
        """Pull at most ``max_samples`` and return an empty chunk on timeout."""

        samples, timestamps = self.inlet.pull_chunk(
            timeout=timeout_seconds,
            max_samples=max_samples,
        )
        if len(samples) == 0:
            return EEGChunk(
                samples=np.empty(0, dtype=np.float32),
                timestamps=np.empty(0, dtype=np.float64),
                sampling_rate=self.sampling_rate,
                stream_name=self.stream_name,
            )
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] <= self.channel_index:
            raise ValueError("O chunk LSL não tem a forma esperada (amostras, canais)")
        timestamp_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if timestamp_array.size != array.shape[0]:
            raise ValueError("O chunk LSL precisa de um timestamp por amostra")
        return EEGChunk(
            samples=array[:, self.channel_index],
            timestamps=timestamp_array,
            sampling_rate=self.sampling_rate,
            stream_name=self.stream_name,
        )

    @property
    def metadata(self) -> dict[str, object]:
        """Return source facts known from the LSL stream descriptor."""

        return {
            "source_name": self.stream_name,
            "stream_type": self.stream_type,
            "channel_index": self.channel_index,
            "channel_count": self.channel_count,
            "sampling_rate": self.sampling_rate,
            **self._descriptor_metadata,
        }


def _ceil_resampled_length(sample_count: int, up: int, down: int) -> int:
    return (sample_count * up + down - 1) // down


class StreamingResampler:
    """Stateful polyphase resampler for chunked acquisition.

    ``scipy.signal.resample_poly`` is a finite-window operation. Re-running it
    independently on every network chunk introduces a boundary transient, so
    this adapter keeps a bounded overlap and emits only samples with enough
    future context. The resulting delay is small for common EEG rate pairs and
    is part of the stream's measured latency; it is not a clinical guarantee.
    """

    def __init__(
        self,
        source_rate: float,
        target_rate: float,
        *,
        max_denominator: int = 1000,
    ) -> None:
        if not np.isfinite(source_rate) or not np.isfinite(target_rate):
            raise ValueError("source_rate e target_rate devem ser finitos")
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("source_rate e target_rate devem ser positivos")
        if max_denominator < 1:
            raise ValueError("max_denominator deve ser positivo")

        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        ratio = Fraction(target_rate / source_rate).limit_denominator(max_denominator)
        self.up = ratio.numerator
        self.down = ratio.denominator
        self._passthrough = np.isclose(source_rate, target_rate)
        self._source_seen = 0
        self._emitted_output = 0
        self._buffer_start = 0
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._timestamp_buffer = np.empty(0, dtype=np.float64)
        self._timestamps_enabled: bool | None = None
        self._last_timestamp: float | None = None

        if not self._passthrough:
            # These values mirror scipy's default Kaiser-window FIR length.
            half_len = 10 * max(self.up, self.down)
            self._holdback_outputs = int(np.ceil((half_len + self.down) / self.down))
            history = int(np.ceil((half_len + self.down) / self.up)) + 2
            history = max(self.down, history)
            self._history_samples = ((history + self.down - 1) // self.down) * self.down
        else:
            self._holdback_outputs = 0
            self._history_samples = 0

    def _validate_timestamps(
        self,
        timestamps: np.ndarray | list[float] | None,
        sample_count: int,
    ) -> np.ndarray | None:
        array = (
            None
            if timestamps is None
            else np.asarray(timestamps, dtype=np.float64).reshape(-1)
        )
        if array is not None and array.size != sample_count:
            raise ValueError("timestamps deve ter o mesmo número de elementos que samples")
        if array is not None and array.size:
            if not np.isfinite(array).all():
                raise ValueError("timestamps devem ser finitos")
            if np.any(np.diff(array) <= 0):
                raise ValueError("timestamps devem ser estritamente crescentes")
            if self._last_timestamp is not None and array[0] <= self._last_timestamp:
                raise ValueError("timestamps devem ser estritamente crescentes")
            self._last_timestamp = float(array[-1])
        if sample_count:
            enabled = array is not None
            if self._timestamps_enabled is None:
                self._timestamps_enabled = enabled
            elif self._timestamps_enabled != enabled:
                raise ValueError("A presença de timestamps não pode mudar durante o stream")
        return array

    def _timestamps_for_output(self, first_output: int, count: int) -> np.ndarray:
        if not self._timestamps_enabled or count == 0:
            return np.empty(0, dtype=np.float64)
        source_positions = np.arange(
            self._buffer_start,
            self._buffer_start + self._timestamp_buffer.size,
            dtype=np.float64,
        )
        output_positions = (
            np.arange(first_output, first_output + count, dtype=np.float64) * self.down / self.up
        )
        if source_positions.size == 0:
            return np.empty(0, dtype=np.float64)
        if source_positions.size == 1:
            # A one-sample finite stream has no observed interval from which
            # to interpolate. Use the declared source rate for the tail.
            return self._timestamp_buffer[0] + (
                output_positions - source_positions[0]
            ) / self.source_rate

        # ``np.interp`` clamps outside its domain. That is fine for most
        # downsampling paths, but upsampling can produce one or more output
        # positions after the final source sample during ``flush``. Clamping
        # would duplicate the final timestamp and make the downstream realtime
        # estimator reject an otherwise valid finite replay. Extrapolate with
        # the observed edge slope instead, preserving strictly increasing
        # timestamps even when source timestamps are slightly irregular.
        timestamps = np.interp(output_positions, source_positions, self._timestamp_buffer)
        left = output_positions < source_positions[0]
        right = output_positions > source_positions[-1]
        if left.any():
            slope = self._timestamp_buffer[1] - self._timestamp_buffer[0]
            timestamps[left] = self._timestamp_buffer[0] + (
                output_positions[left] - source_positions[0]
            ) * slope
        if right.any():
            slope = self._timestamp_buffer[-1] - self._timestamp_buffer[-2]
            timestamps[right] = self._timestamp_buffer[-1] + (
                output_positions[right] - source_positions[-1]
            ) * slope
        return timestamps

    def _trim_buffer(self) -> None:
        if self._source_seen <= self._history_samples:
            return
        candidate = ((self._source_seen - self._history_samples) // self.down) * self.down
        # Never discard the source context needed to locate the next output.
        max_start = (self._emitted_output * self.down) // self.up
        new_start = min(candidate, max_start)
        new_start = (new_start // self.down) * self.down
        if new_start <= self._buffer_start:
            return
        offset = new_start - self._buffer_start
        self._input_buffer = self._input_buffer[offset:]
        if self._timestamps_enabled:
            self._timestamp_buffer = self._timestamp_buffer[offset:]
        self._buffer_start = new_start

    def process(
        self,
        samples: np.ndarray | list[float],
        timestamps: np.ndarray | list[float] | None = None,
    ) -> EEGChunk:
        """Convert one chunk and preserve resampling state for the next one."""

        samples_array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples_array.size and not np.isfinite(samples_array).all():
            # Do not let a bad acquisition chunk enter the resampler's
            # overlap/state. The caller can retain the audit diagnostics and
            # terminate or reconnect the stream explicitly.
            raise ValueError("samples devem ser finitas no resampler streaming")
        timestamp_array = self._validate_timestamps(timestamps, samples_array.size)
        if samples_array.size == 0:
            return EEGChunk(
                samples_array,
                np.empty(0, dtype=np.float64),
                self.target_rate,
            )
        if self._passthrough:
            return EEGChunk(
                samples_array,
                timestamp_array
                if timestamp_array is not None
                else np.empty(0, dtype=np.float64),
                self.target_rate,
            )

        self._input_buffer = np.concatenate((self._input_buffer, samples_array))
        if timestamp_array is not None:
            self._timestamp_buffer = np.concatenate((self._timestamp_buffer, timestamp_array))
        self._source_seen += samples_array.size

        converted = resample_poly(self._input_buffer, self.up, self.down).astype(np.float32)
        global_start_output = (self._buffer_start * self.up) // self.down
        safe_end_output = max(
            0,
            _ceil_resampled_length(self._source_seen, self.up, self.down)
            - self._holdback_outputs,
        )
        local_start = max(0, self._emitted_output - global_start_output)
        local_end = min(converted.size, safe_end_output - global_start_output)
        if local_end <= local_start:
            output = np.empty(0, dtype=np.float32)
            output_timestamps = np.empty(0, dtype=np.float64)
        else:
            output = converted[local_start:local_end]
            output_timestamps = self._timestamps_for_output(
                self._emitted_output,
                local_end - local_start,
            )
            self._emitted_output += output.size
        self._trim_buffer()
        return EEGChunk(output, output_timestamps, self.target_rate)

    def flush(self) -> EEGChunk:
        """Emit the delayed tail after a finite replay; live streams do not flush."""

        if self._passthrough or self._input_buffer.size == 0:
            return EEGChunk(
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float64),
                self.target_rate,
            )
        converted = resample_poly(self._input_buffer, self.up, self.down).astype(np.float32)
        global_start_output = (self._buffer_start * self.up) // self.down
        local_start = max(0, self._emitted_output - global_start_output)
        if local_start >= converted.size:
            return EEGChunk(
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float64),
                self.target_rate,
            )
        output = converted[local_start:]
        output_timestamps = self._timestamps_for_output(self._emitted_output, output.size)
        self._emitted_output += output.size
        return EEGChunk(output, output_timestamps, self.target_rate)


def resample_chunk(
    samples: np.ndarray,
    source_rate: float,
    target_rate: float,
    *,
    timestamps: np.ndarray | None = None,
) -> EEGChunk:
    """Convert a chunk to the model rate using a rational polyphase filter.

    For a production acquisition path, resampling state should be carried
    across chunks; this stateless helper is explicit so that the limitation is
    visible during replay and pilot experiments.
    """

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    timestamp_array = (
        None
        if timestamps is None
        else np.asarray(timestamps, dtype=np.float64).reshape(-1)
    )
    if timestamp_array is not None and timestamp_array.size != samples.size:
        raise ValueError("timestamps deve ter o mesmo número de elementos que samples")
    if samples.size == 0:
        return EEGChunk(
            samples,
            timestamp_array if timestamp_array is not None else np.empty(0, dtype=np.float64),
            float(target_rate),
        )
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate e target_rate devem ser positivos")
    if np.isclose(source_rate, target_rate):
        return EEGChunk(
            samples,
            timestamp_array if timestamp_array is not None else np.empty(0, dtype=np.float64),
            float(target_rate),
        )
    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    converted = resample_poly(samples, ratio.numerator, ratio.denominator).astype(np.float32)
    converted_timestamps = (
        timestamp_array if timestamp_array is not None else np.empty(0, dtype=np.float64)
    )
    if timestamp_array is not None and timestamp_array.size > 1 and converted.size > 1:
        converted_timestamps = np.linspace(
            timestamp_array[0], timestamp_array[-1], converted.size, dtype=np.float64
        )
    elif timestamp_array is not None and timestamp_array.size == 1:
        converted_timestamps = np.full(converted.size, timestamp_array[0], dtype=np.float64)
    return EEGChunk(converted, converted_timestamps, float(target_rate))

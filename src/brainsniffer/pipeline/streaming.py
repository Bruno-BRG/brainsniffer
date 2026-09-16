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
        self.reset()

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

    def reset(self) -> None:
        """Discard overlap, pending output and timestamp history; retain rates/filter.

        No delayed tail is emitted. The next ``process`` starts an independent
        stream and may choose a different timestamp origin or presence policy.
        """

        self._source_seen = 0
        self._emitted_output = 0
        self._buffer_start = 0
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._timestamp_buffer = np.empty(0, dtype=np.float64)
        self._timestamps_enabled: bool | None = None
        self._last_timestamp: float | None = None

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
    visible during replay and pilot experiments. Output timestamps use the
    declared regular grid ``t0 + k / target_rate``, not the input endpoints.
    """

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    timestamp_array = (
        None
        if timestamps is None
        else np.asarray(timestamps, dtype=np.float64).reshape(-1)
    )
    if timestamp_array is not None and timestamp_array.size != samples.size:
        raise ValueError("timestamps deve ter o mesmo número de elementos que samples")
    if not np.isfinite(source_rate) or not np.isfinite(target_rate):
        raise ValueError("source_rate e target_rate devem ser finitos")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate e target_rate devem ser positivos")
    if samples.size == 0:
        return EEGChunk(
            samples,
            timestamp_array if timestamp_array is not None else np.empty(0, dtype=np.float64),
            float(target_rate),
        )
    if source_rate == target_rate:
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
    if timestamp_array is not None:
        converted_timestamps = (
            timestamp_array[0] + np.arange(converted.size, dtype=np.float64) / target_rate
        )
    return EEGChunk(converted, converted_timestamps, float(target_rate))

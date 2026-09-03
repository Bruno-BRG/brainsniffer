"""Streaming input preflight checks for research acquisition bridges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from ..config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig

REQUIRED_METADATA_FIELDS = ("unit", "channel_name", "reference", "montage")


def _normalize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Keep the stream manifest small, scalar, and JSON-serializable."""

    normalized: dict[str, object] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("as chaves de metadata não podem ser vazias")
        if raw_value is None:
            continue
        if isinstance(raw_value, (np.integer,)):
            value: object = int(raw_value)
        elif isinstance(raw_value, (np.floating,)):
            value = float(raw_value)
        elif isinstance(raw_value, (str, int, float, bool)):
            value = raw_value
        else:
            raise ValueError("metadata deve conter somente valores escalares")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("valores numéricos de metadata devem ser finitos")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        if key == "sampling_rate":
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("sampling_rate do metadata deve ser numérico") from error
            if not np.isfinite(value) or value <= 0:
                raise ValueError("sampling_rate do metadata deve ser finito e positivo")
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class StreamAuditReport:
    """Serializable diagnostics for one finite or partial input stream."""

    ok: bool
    sample_count: int
    chunk_count: int
    source_rate: float | None
    duration_seconds: float | None
    finite_fraction: float
    saturation_fraction: float
    flatline_fraction: float
    quality: float
    raw_min: float | None
    raw_max: float | None
    raw_mean: float | None
    raw_rms: float | None
    timestamps_present: bool | None
    timestamp_first: float | None
    timestamp_last: float | None
    timestamp_nonfinite_count: int
    timestamp_nonmonotonic_count: int
    timestamp_gap_count: int
    warnings: tuple[str, ...]
    metadata: dict[str, object] | None = None
    metadata_complete: bool = False
    metadata_missing: tuple[str, ...] = ()
    timestamps_required: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class StreamAudit:
    """Accumulate bounded diagnostics without retaining the complete signal."""

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        *,
        min_quality: float = DEFAULT_MIN_SIGNAL_QUALITY,
        max_gap_factor: float = 1.5,
        require_metadata: bool = False,
        require_timestamps: bool = False,
    ) -> None:
        self.config = config or PreprocessConfig()
        if not 0 <= min_quality <= 1:
            raise ValueError("min_quality deve estar entre 0 e 1")
        if max_gap_factor < 1:
            raise ValueError("max_gap_factor deve ser pelo menos 1")
        self.min_quality = float(min_quality)
        self.max_gap_factor = float(max_gap_factor)
        self.require_metadata = bool(require_metadata)
        self.require_timestamps = bool(require_timestamps)
        self._sample_count = 0
        self._chunk_count = 0
        self._source_rate: float | None = None
        self._finite_count = 0
        self._saturation_count = 0
        self._flatline_count = 0
        self._difference_count = 0
        self._sum = 0.0
        self._sum_squares = 0.0
        self._raw_min: float | None = None
        self._raw_max: float | None = None
        self._last_sample: float | None = None
        self._timestamps_present: bool | None = None
        self._timestamp_first: float | None = None
        self._timestamp_last: float | None = None
        self._timestamp_nonfinite_count = 0
        self._timestamp_nonmonotonic_count = 0
        self._timestamp_gap_count = 0
        self._warnings: list[str] = []
        self._metadata: dict[str, object] | None = None

    def set_metadata(self, metadata: Mapping[str, object] | None) -> None:
        """Register one stable source manifest, normally on the first chunk."""

        if metadata is None:
            return
        normalized = _normalize_metadata(metadata)
        if self._metadata is None:
            self._metadata = {}
        for key, value in normalized.items():
            if key in self._metadata and self._metadata[key] != value:
                raise ValueError("metadata não pode mudar durante o stream")
            self._metadata[key] = value
        self._validate_metadata_rate()

    def _validate_metadata_rate(self) -> None:
        if self._metadata is None or "sampling_rate" not in self._metadata:
            return
        if self._source_rate is not None and not np.isclose(
            float(self._metadata["sampling_rate"]),
            self._source_rate,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("sampling_rate do metadata diverge de source_rate")

    def _metadata_missing(self) -> tuple[str, ...]:
        metadata = self._metadata or {}
        return tuple(
            field
            for field in REQUIRED_METADATA_FIELDS
            if not str(metadata.get(field, "")).strip()
        )

    def metadata_complete(self) -> bool:
        """Whether unit, channel identity, reference, and montage are known."""

        return not self._metadata_missing()

    def push(
        self,
        samples: np.ndarray | list[float],
        *,
        source_rate: float,
        timestamps: np.ndarray | list[float] | None = None,
    ) -> None:
        """Add one chunk and raise on structural stream changes."""

        samples_array = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.isfinite(source_rate) or source_rate <= 0:
            raise ValueError("source_rate deve ser finito e positivo")
        if self._source_rate is None:
            self._source_rate = float(source_rate)
        elif not np.isclose(self._source_rate, source_rate, rtol=1e-6, atol=1e-6):
            raise ValueError("source_rate não pode mudar durante o stream")
        self._validate_metadata_rate()

        timestamp_array = (
            None
            if timestamps is None
            else np.asarray(timestamps, dtype=np.float64).reshape(-1)
        )
        if timestamp_array is not None and timestamp_array.size != samples_array.size:
            raise ValueError("timestamps deve ter o mesmo número de elementos que samples")
        if samples_array.size == 0:
            return
        timestamps_present = timestamp_array is not None
        if self._timestamps_present is None:
            self._timestamps_present = timestamps_present
        elif self._timestamps_present != timestamps_present:
            raise ValueError("A presença de timestamps não pode mudar durante o stream")

        finite = np.isfinite(samples_array)
        safe = np.nan_to_num(samples_array, nan=0.0, posinf=0.0, neginf=0.0)
        self._sample_count += samples_array.size
        self._chunk_count += 1
        self._finite_count += int(finite.sum())
        self._saturation_count += int(
            (finite & (np.abs(samples_array) >= self.config.clip_uv * 0.98)).sum()
        )
        finite_values = samples_array[finite]
        if finite_values.size:
            chunk_min = float(finite_values.min())
            chunk_max = float(finite_values.max())
            self._raw_min = chunk_min if self._raw_min is None else min(self._raw_min, chunk_min)
            self._raw_max = chunk_max if self._raw_max is None else max(self._raw_max, chunk_max)
            self._sum += float(finite_values.sum())
            self._sum_squares += float(np.square(finite_values).sum())

        previous = self._last_sample
        if previous is not None and finite[0] and abs(previous - samples_array[0]) < 1e-5:
            self._flatline_count += 1
            self._difference_count += 1
        if samples_array.size > 1:
            differences = np.abs(np.diff(safe))
            self._flatline_count += int((differences < 1e-5).sum())
            self._difference_count += differences.size
        self._last_sample = float(samples_array[-1]) if np.isfinite(samples_array[-1]) else None

        if timestamp_array is not None:
            finite_timestamps = np.isfinite(timestamp_array)
            self._timestamp_nonfinite_count += int((~finite_timestamps).sum())
            valid_timestamps = timestamp_array[finite_timestamps]
            if valid_timestamps.size:
                if self._timestamp_first is None:
                    self._timestamp_first = float(valid_timestamps[0])
                timestamp_differences = np.diff(valid_timestamps)
                if self._timestamp_last is not None:
                    timestamp_differences = np.concatenate(
                        ([valid_timestamps[0] - self._timestamp_last], timestamp_differences)
                    )
                if timestamp_differences.size:
                    self._timestamp_nonmonotonic_count += int((timestamp_differences <= 0).sum())
                    expected = 1.0 / float(source_rate)
                    self._timestamp_gap_count += int(
                        (timestamp_differences > self.max_gap_factor * expected).sum()
                    )
                self._timestamp_last = float(valid_timestamps[-1])

    def report(self) -> StreamAuditReport:
        """Return diagnostics and conservative warnings accumulated so far."""

        if self._sample_count:
            finite_fraction = self._finite_count / self._sample_count
            saturation_fraction = self._saturation_count / self._sample_count
        else:
            finite_fraction = 0.0
            saturation_fraction = 0.0
        flatline_fraction = (
            self._flatline_count / self._difference_count if self._difference_count else 0.0
        )
        quality = float(
            np.clip(
                finite_fraction
                * (1.0 - min(saturation_fraction * 2.0, 1.0))
                * (1.0 - min(flatline_fraction * 3.0, 1.0)),
                0.0,
                1.0,
            )
        )
        warnings = list(self._warnings)
        if not self._sample_count:
            warnings.append("nenhuma amostra recebida")
        if self._sample_count and finite_fraction < 1.0:
            warnings.append("há amostras não finitas")
        if saturation_fraction > 0:
            warnings.append("há amostras próximas do clipping")
        if flatline_fraction > 0.1:
            warnings.append("há uma fração relevante de linha plana")
        if self._timestamps_present is False:
            warnings.append("timestamps ausentes")
        if self.require_timestamps and self._timestamps_present is not True:
            warnings.append("timestamps obrigatórios ausentes")
        if self._timestamp_nonfinite_count:
            warnings.append("há timestamps não finitos")
        if self._timestamp_nonmonotonic_count:
            warnings.append("há timestamps não monotônicos")
        if self._timestamp_gap_count:
            warnings.append("há lacunas de timestamp acima do limite")
        if quality < self.min_quality:
            warnings.append(f"qualidade heurística abaixo de {self.min_quality:g}")
        metadata_missing = self._metadata_missing()
        metadata_complete = not metadata_missing
        if not metadata_complete:
            warnings.append(
                "metadata obrigatório incompleto: " + ", ".join(metadata_missing)
            )
        return StreamAuditReport(
            ok=bool(
                self._sample_count > 0
                and finite_fraction == 1.0
                and self._timestamp_nonfinite_count == 0
                and self._timestamp_nonmonotonic_count == 0
                and self._timestamp_gap_count == 0
                and (self._timestamps_present is True or not self.require_timestamps)
                and quality >= self.min_quality
                and (metadata_complete or not self.require_metadata)
            ),
            sample_count=self._sample_count,
            chunk_count=self._chunk_count,
            source_rate=self._source_rate,
            duration_seconds=(
                self._sample_count / self._source_rate if self._source_rate else None
            ),
            finite_fraction=float(finite_fraction),
            saturation_fraction=float(saturation_fraction),
            flatline_fraction=float(flatline_fraction),
            quality=quality,
            raw_min=self._raw_min,
            raw_max=self._raw_max,
            raw_mean=(self._sum / self._finite_count if self._finite_count else None),
            raw_rms=(
                float(np.sqrt(self._sum_squares / self._finite_count))
                if self._finite_count
                else None
            ),
            timestamps_present=self._timestamps_present,
            timestamp_first=self._timestamp_first,
            timestamp_last=self._timestamp_last,
            timestamp_nonfinite_count=self._timestamp_nonfinite_count,
            timestamp_nonmonotonic_count=self._timestamp_nonmonotonic_count,
            timestamp_gap_count=self._timestamp_gap_count,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata=self._metadata,
            metadata_complete=metadata_complete,
            metadata_missing=metadata_missing,
            timestamps_required=self.require_timestamps,
        )

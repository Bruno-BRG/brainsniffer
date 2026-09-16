"""Selective VitalDB Open Dataset client for external research validation.

The full VitalDB distribution is intentionally not downloaded by this project.
This module retrieves only the requested case's BIS EEG waveform and BIS track
through the official HTTP API, then writes a normalized local ``.npz`` file.
"""

from __future__ import annotations

import csv
import gzip
import io
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

VITALDB_API_BASE_URL = "https://api.vitaldb.net"
VITALDB_TRACK_INDEX_URL = f"{VITALDB_API_BASE_URL}/trks"
VITALDB_CLINICAL_DATA_URL = (
    "https://physionet.org/files/vitaldb/1.0.0/clinical_data.csv"
)
DEFAULT_EEG_TRACK = "BIS/EEG1_WAV"
DEFAULT_BIS_TRACK = "BIS/BIS"
VITALDB_LABEL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class VitalTrack:
    """A VitalDB track identifier from the official track index."""

    case_id: int
    name: str
    track_id: str
    subject_id: str | None = None


@dataclass(frozen=True)
class VitalTrackData:
    """Decoded track values and their source time representation."""

    name: str
    times: np.ndarray
    values: np.ndarray
    sampling_rate: float | None


@lru_cache(maxsize=1)
def fetch_vitaldb_subject_map() -> dict[str, str]:
    """Fetch the public case-to-subject map used to prevent reoperation leakage."""

    request = Request(VITALDB_CLINICAL_DATA_URL, headers={"User-Agent": "brainsniffer/0.1"})
    with urlopen(request, timeout=120) as response:
        text = io.TextIOWrapper(response, encoding="utf-8-sig")
        reader = csv.DictReader(text)
        mapping: dict[str, str] = {}
        for row in reader:
            case_id = (row.get("caseid") or "").strip()
            subject_id = (row.get("subjectid") or "").strip()
            if case_id and subject_id:
                mapping[case_id] = subject_id
    if not mapping:
        raise ValueError("clinical_data.csv do VitalDB não contém o mapa caseid/subjectid")
    return mapping


@contextmanager
def _open_gzip_text(url: str):
    request = Request(url, headers={"User-Agent": "brainsniffer/0.1"})
    with urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8-sig") as text:
                yield text


def list_case_tracks(
    case_id: int,
    *,
    track_names: tuple[str, ...] = (DEFAULT_EEG_TRACK, DEFAULT_BIS_TRACK),
) -> dict[str, VitalTrack]:
    """Find selected tracks for one case in the official compressed index."""

    wanted = set(track_names)
    found: dict[str, VitalTrack] = {}
    with _open_gzip_text(VITALDB_TRACK_INDEX_URL) as text:
        for row in csv.DictReader(text):
            if int(row["caseid"]) != int(case_id) or row["tname"] not in wanted:
                continue
            found[row["tname"]] = VitalTrack(
                case_id=int(row["caseid"]),
                name=row["tname"],
                track_id=row["tid"],
                subject_id=(row.get("subjectid") or row.get("subject_id") or None),
            )
    missing = wanted - found.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"VitalDB case {case_id} não contém os tracks: {names}")
    return found


def _as_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if np.isfinite(parsed) else None


def read_track(track: VitalTrack, *, waveform: bool) -> VitalTrackData:
    """Read one compressed VitalDB track using its documented CSV encoding."""

    url = f"{VITALDB_API_BASE_URL}/{track.track_id}"
    values: list[float] = []
    time_markers: list[float] = []
    numeric_times: list[float] = []
    numeric_values: list[float] = []
    with _open_gzip_text(url) as text:
        reader = csv.reader(text)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"Track VitalDB inválido para {track.name}")
        for row in reader:
            if len(row) < 2:
                continue
            raw_time, raw_value = row[0], row[1]
            value = _as_float(raw_value)
            if waveform:
                values.append(np.nan if value is None else value)
                marker = _as_float(raw_time)
                if marker is not None:
                    time_markers.append(marker)
            else:
                timestamp = _as_float(raw_time)
                if timestamp is not None and value is not None:
                    numeric_times.append(timestamp)
                    numeric_values.append(value)

    if waveform:
        if len(time_markers) < 2 or not values:
            raise ValueError(f"Track waveform VitalDB incompleto para {track.name}")
        start_time = time_markers[0]
        interval = time_markers[1]
        if interval <= 0 or not np.isfinite(interval):
            raise ValueError(f"Intervalo inválido no waveform VitalDB {track.name}")
        times = start_time + np.arange(len(values), dtype=np.float64) * interval
        return VitalTrackData(
            name=track.name,
            times=times,
            values=np.asarray(values, dtype=np.float32),
            sampling_rate=1.0 / interval,
        )

    if not numeric_values:
        raise ValueError(f"Track numérico VitalDB vazio para {track.name}")
    return VitalTrackData(
        name=track.name,
        times=np.asarray(numeric_times, dtype=np.float64),
        values=np.asarray(numeric_values, dtype=np.float32),
        sampling_rate=None,
    )


def _align_numeric_to_eeg(
    eeg: VitalTrackData,
    bis: VitalTrackData,
    *,
    label_interval_seconds: float = VITALDB_LABEL_INTERVAL_SECONDS,
) -> np.ndarray:
    """Interpolate numeric BIS onto a one-second grid relative to EEG start."""

    if eeg.times.size == 0 or eeg.sampling_rate is None:
        raise ValueError("Waveform EEG VitalDB sem tempo ou taxa")
    if label_interval_seconds <= 0:
        raise ValueError("label_interval_seconds deve ser positivo")
    eeg_start = float(eeg.times[0])
    duration = float(eeg.times[-1] - eeg_start)
    grid = np.arange(
        0.0,
        duration + label_interval_seconds * 0.5,
        label_interval_seconds,
        dtype=np.float64,
    )
    relative_times = bis.times - eeg_start
    valid = np.isfinite(relative_times) & np.isfinite(bis.values)
    if valid.sum() < 2:
        raise ValueError("VitalDB BIS não tem pontos suficientes para alinhamento")
    relative_times = relative_times[valid]
    values = bis.values.astype(np.float64)[valid]
    order = np.argsort(relative_times)
    relative_times = relative_times[order]
    values = values[order]
    relative_times, unique_indices = np.unique(relative_times, return_index=True)
    values = values[unique_indices]
    aligned = np.full(grid.shape, np.nan, dtype=np.float32)
    inside = (grid >= relative_times[0]) & (grid <= relative_times[-1])
    aligned[inside] = np.interp(grid[inside], relative_times, values).astype(np.float32)
    return aligned


def download_vitaldb_case(
    case_id: int,
    destination: str | Path = "data/vitaldb",
    *,
    eeg_track_name: str = DEFAULT_EEG_TRACK,
    bis_track_name: str = DEFAULT_BIS_TRACK,
    overwrite: bool = False,
    subject_id: str | None = None,
) -> Path:
    """Download and normalize one VitalDB case without fetching the full corpus."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"vitaldb_case{int(case_id)}.npz"
    if target.exists() and not overwrite:
        return target

    tracks = list_case_tracks(case_id, track_names=(eeg_track_name, bis_track_name))
    eeg = read_track(tracks[eeg_track_name], waveform=True)
    bis = read_track(tracks[bis_track_name], waveform=False)
    aligned_bis = _align_numeric_to_eeg(eeg, bis)
    resolved_subject_id = (
        subject_id
        or tracks[eeg_track_name].subject_id
        or tracks[bis_track_name].subject_id
    )
    group_id = (
        f"vitaldb:subject:{resolved_subject_id}"
        if resolved_subject_id
        else f"vitaldb:case:vitaldb_case{int(case_id)}"
    )
    partial = destination / f".{target.name}.part"
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            case_id=np.asarray(f"vitaldb_case{int(case_id)}"),
            group_id=np.asarray(group_id),
            subject_id=np.asarray(resolved_subject_id or ""),
            source_dataset=np.asarray("VitalDB Open Dataset"),
            eeg=eeg.values,
            bis=aligned_bis,
            sampling_rate=np.asarray(eeg.sampling_rate, dtype=np.float64),
            label_interval_seconds=np.asarray(VITALDB_LABEL_INTERVAL_SECONDS),
            eeg_unit=np.asarray("uV"),
            eeg_track_name=np.asarray(eeg_track_name),
            bis_track_name=np.asarray(bis_track_name),
            eeg_track_id=np.asarray(tracks[eeg_track_name].track_id),
            bis_track_id=np.asarray(tracks[bis_track_name].track_id),
        )
    partial.replace(target)
    return target

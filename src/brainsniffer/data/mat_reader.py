"""Reader for the MATLAB v7.3 files published with the EEG/BIS dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DEFAULT_LABEL_INTERVAL_SECONDS, DEFAULT_SAMPLING_RATE


@dataclass(frozen=True)
class EEGCase:
    """One anonymized surgical case."""

    case_id: str
    eeg: np.ndarray
    bis: np.ndarray
    sampling_rate: int = DEFAULT_SAMPLING_RATE
    label_interval_seconds: float = DEFAULT_LABEL_INTERVAL_SECONDS
    source_dataset: str | None = None
    eeg_unit: str | None = None
    eeg_track_name: str | None = None
    bis_track_name: str | None = None

    @property
    def duration_seconds(self) -> float:
        return float(self.eeg.size / self.sampling_rate)


def _flatten_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).squeeze()
    return array.reshape(-1)


def _scalar_string(value: object | None) -> str | None:
    """Read an optional scalar string from normalized NPZ metadata."""

    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    return str(array.reshape(-1)[0])


def _read_hdf5_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        datasets: dict[str, object] = {}

        def collect(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name.rsplit("/", 1)[-1].lower()] = obj[()]

        handle.visititems(collect)
    eeg = next((value for key, value in datasets.items() if key in {"eeg", "eeg1"}), None)
    bis = next((value for key, value in datasets.items() if key in {"bis", "bis_index"}), None)
    if eeg is None or bis is None:
        raise ValueError(f"{path} não contém datasets EEG e bis reconhecíveis")
    return _flatten_array(eeg), _flatten_array(bis)


def _read_classic_mat_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    normalized = {key.lower(): value for key, value in data.items() if not key.startswith("__")}
    eeg = normalized.get("eeg")
    if eeg is None:
        eeg = normalized.get("eeg1")
    bis = normalized.get("bis")
    if bis is None:
        bis = normalized.get("bis_index")
    if eeg is None or bis is None:
        raise ValueError(f"{path} não contém variáveis EEG e bis reconhecíveis")
    return _flatten_array(eeg), _flatten_array(bis)


def _read_normalized_npz_case(path: Path) -> EEGCase:
    with np.load(path, allow_pickle=False) as data:
        if "eeg" not in data or "bis" not in data:
            raise ValueError(f"{path} não contém arrays eeg e bis reconhecíveis")
        eeg = _flatten_array(data["eeg"])
        bis = _flatten_array(data["bis"])
        stored_rate = float(data.get("sampling_rate", DEFAULT_SAMPLING_RATE))
        label_interval = float(
            data.get("label_interval_seconds", DEFAULT_LABEL_INTERVAL_SECONDS)
        )
        stored_case_id = data.get("case_id")
        case_id = str(stored_case_id.item()) if stored_case_id is not None else path.stem
        source_dataset = _scalar_string(data.get("source_dataset"))
        eeg_unit = _scalar_string(data.get("eeg_unit"))
        eeg_track_name = _scalar_string(data.get("eeg_track_name"))
        bis_track_name = _scalar_string(data.get("bis_track_name"))
    if not np.isfinite(stored_rate) or stored_rate <= 0:
        raise ValueError(f"{path} contém uma taxa inválida")
    if not np.isclose(stored_rate, DEFAULT_SAMPLING_RATE):
        raise ValueError(
            f"{path} usa {stored_rate:g} Hz; o pipeline atual espera {DEFAULT_SAMPLING_RATE} Hz"
        )
    if eeg.size == 0 or bis.size == 0:
        raise ValueError(f"{path} está vazio")
    return EEGCase(
        case_id=case_id,
        eeg=eeg,
        bis=bis,
        sampling_rate=int(round(stored_rate)),
        label_interval_seconds=label_interval,
        source_dataset=source_dataset,
        eeg_unit=eeg_unit,
        eeg_track_name=eeg_track_name,
        bis_track_name=bis_track_name,
    )


def load_case(
    path: str | Path,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    label_interval_seconds: float = DEFAULT_LABEL_INTERVAL_SECONDS,
) -> EEGCase:
    """Load a case, supporting both MATLAB v7.3 and classic MAT files."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        return _read_normalized_npz_case(path)
    try:
        eeg, bis = _read_hdf5_case(path)
    except OSError:
        eeg, bis = _read_classic_mat_case(path)
    if eeg.size == 0 or bis.size == 0:
        raise ValueError(f"{path} está vazio")
    return EEGCase(
        case_id=path.stem,
        eeg=np.asarray(eeg, dtype=np.float32),
        bis=np.asarray(bis, dtype=np.float32),
        sampling_rate=sampling_rate,
        label_interval_seconds=label_interval_seconds,
    )

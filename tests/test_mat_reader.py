import h5py
import numpy as np

from brainsniffer.data.mat_reader import load_case


def test_load_matlab_v73_case(tmp_path):
    path = tmp_path / "case99.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("EEG", data=np.arange(1280, dtype=np.float64))
        handle.create_dataset("bis", data=np.array([90, 50], dtype=np.float64))
    case = load_case(path)
    assert case.case_id == "case99"
    assert case.eeg.dtype == np.float32
    assert case.eeg.size == 1280
    assert case.bis.tolist() == [90.0, 50.0]
    assert case.duration_seconds == 10.0


def test_load_normalized_npz_case(tmp_path):
    path = tmp_path / "vitaldb_case1.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray("vitaldb_case1"),
        eeg=np.arange(1280, dtype=np.float32),
        bis=np.asarray([50.0, 55.0, 60.0]),
        sampling_rate=np.asarray(128.0),
        label_interval_seconds=np.asarray(1.0),
    )
    case = load_case(path)
    assert case.case_id == "vitaldb_case1"
    assert case.sampling_rate == 128
    assert case.label_interval_seconds == 1.0
    assert case.eeg.size == 1280


def test_load_normalized_npz_preserves_provenance_metadata(tmp_path):
    path = tmp_path / "vitaldb_case1.npz"
    np.savez(
        path,
        case_id=np.asarray("vitaldb_case1"),
        source_dataset=np.asarray("VitalDB Open Dataset"),
        eeg=np.ones(128, dtype=np.float32),
        bis=np.asarray([50.0], dtype=np.float32),
        sampling_rate=np.asarray(128.0),
        label_interval_seconds=np.asarray(1.0),
        eeg_unit=np.asarray("uV"),
        eeg_track_name=np.asarray("BIS/EEG1_WAV"),
        bis_track_name=np.asarray("BIS/BIS"),
    )

    case = load_case(path)

    assert case.source_dataset == "VitalDB Open Dataset"
    assert case.eeg_unit == "uV"
    assert case.eeg_track_name == "BIS/EEG1_WAV"
    assert case.bis_track_name == "BIS/BIS"

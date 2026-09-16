import json

import h5py
import numpy as np

from brainsniffer.config import PreprocessConfig
from brainsniffer.data.corpus import (
    CorpusQualityConfig,
    audit_corpus_case,
    build_corpus_manifest,
    corpus_paths,
    load_corpus_manifest,
    write_corpus_manifest,
)


def _write_mat(path, *, samples=1280, bis=(50.0, 55.0)):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("EEG", data=np.sin(np.arange(samples, dtype=np.float32)))
        handle.create_dataset("bis", data=np.asarray(bis, dtype=np.float32))


def _write_npz(path, *, eeg, case_id="vitaldb_case1", subject_id="subject-1"):
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        group_id=np.asarray(f"vitaldb:subject:{subject_id}"),
        subject_id=np.asarray(subject_id),
        source_dataset=np.asarray("VitalDB Open Dataset"),
        eeg=np.asarray(eeg, dtype=np.float32),
        bis=np.asarray([50.0, 55.0, 60.0], dtype=np.float32),
        sampling_rate=np.asarray(128.0),
        label_interval_seconds=np.asarray(1.0),
    )


def test_audit_quarantines_long_missing_signal_gap(tmp_path):
    path = tmp_path / "vitaldb_case1.npz"
    eeg = np.sin(np.arange(2560, dtype=np.float32))
    eeg[128 : 128 + 512] = np.nan
    _write_npz(path, eeg=eeg)

    record = audit_corpus_case(
        path,
        role="development_pool",
        preprocess_config=PreprocessConfig(),
        quality_config=CorpusQualityConfig(max_gap_seconds=2.0),
    )

    assert record["quality_status"] == "quarantine"
    assert record["eligible_for_training"] is False
    assert "nonfinite_gap_above_gate" in record["exclusion_reasons"]
    assert record["signal"]["nonfinite_count"] == 512
    assert '"eeg":' not in json.dumps(record)


def test_build_manifest_keeps_external_cases_out_of_training(tmp_path):
    figshare_dir = tmp_path / "raw"
    vital_train_dir = tmp_path / "vital-train"
    vital_external_dir = tmp_path / "vital-external"
    figshare_dir.mkdir()
    vital_train_dir.mkdir()
    vital_external_dir.mkdir()
    _write_mat(figshare_dir / "case1.mat")
    _write_mat(figshare_dir / "case2.mat")
    _write_mat(figshare_dir / "case3.mat")
    _write_npz(
        vital_train_dir / "vitaldb_case20.npz",
        eeg=np.sin(np.arange(1280)),
        subject_id="subject-20",
    )
    _write_npz(vital_external_dir / "vitaldb_case1.npz", eeg=np.sin(np.arange(1280)))

    manifest = build_corpus_manifest(
        figshare_dir=figshare_dir,
        vitaldb_train_dir=vital_train_dir,
        vitaldb_external_dir=vital_external_dir,
    )

    assert manifest["summary"]["development_cases"] == 4
    assert manifest["summary"]["frozen_external_cases"] == 1
    assert {record["source_key"] for record in manifest["eligible_cases"]} == {
        "figshare",
        "vitaldb",
    }
    external_ids = {record["case_id"] for record in manifest["frozen_external_cases"]}
    assert external_ids == {"vitaldb_case1"}

    path = write_corpus_manifest(manifest, tmp_path / "corpus.json")
    loaded = load_corpus_manifest(path)
    selected = corpus_paths(loaded)
    assert all("vitaldb_case1" not in item.name for item in selected)
    assert len(selected) == 4

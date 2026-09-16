import json

import pytest
import torch

from brainsniffer.models.cnn import Conv1DDepthEstimator
from brainsniffer.pipeline.training import (
    build_file_manifest,
    load_checkpoint,
    runtime_metadata,
    sha256_file,
    verify_file_manifest,
)


def test_runtime_metadata_contains_reproducibility_fields():
    metadata = runtime_metadata()

    assert metadata["project"]
    assert metadata["python"].startswith("3.12")
    assert metadata["torch"]
    assert metadata["numpy"]
    assert metadata["scipy"]
    assert metadata["scikit_learn"]


def test_sha256_file_is_stable_and_validates_chunk_size(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"brain-sniffer")

    assert sha256_file(path) == sha256_file(path, chunk_size=3)
    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(path, chunk_size=0)


def test_file_manifest_records_and_verifies_input_integrity(tmp_path):
    path = tmp_path / "case1.mat"
    path.write_bytes(b"eeg-bis")

    manifest = build_file_manifest([path])

    assert manifest == [
        {
            "path": str(path),
            "size_bytes": 7,
            "sha256": sha256_file(path),
        }
    ]
    verify_file_manifest(manifest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="arquivo de entrada"):
        verify_file_manifest(manifest)


def test_load_checkpoint_rejects_sidecar_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": Conv1DDepthEstimator().state_dict(),
            "preprocess_config": {},
        },
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"checkpoint_sha256": sha256_file(checkpoint)}),
        encoding="utf-8",
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(checkpoint)

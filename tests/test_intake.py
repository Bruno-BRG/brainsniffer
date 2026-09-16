import io
import json
import sys

import numpy as np
import pytest
import torch

from brainsniffer.cli import main
from brainsniffer.config import PreprocessConfig
from brainsniffer.pipeline.intake import validate_intake_metadata

COMPLETE_MANIFEST = {
    "device_manufacturer": "Bench EEG",
    "device_model": "B-100",
    "firmware": "1.2.3",
    "bridge": "vendor-sdk-json",
    "sampling_rate": 256,
    "unit": "uV",
    "channel_name": "Fpz",
    "reference": "linked ears",
    "montage": "frontal referenced",
    "nominal_range": "±100 uV",
    "processing_applied": "raw EEG; bridge sem ganho adicional",
}


def test_validate_intake_accepts_complete_manifest_without_samples():
    report = validate_intake_metadata(COMPLETE_MANIFEST)

    assert report["ready_for_bench"] is True
    assert report["status"] == "ready_for_bench"
    assert report["missing_fields"] == []
    assert report["metadata"]["sampling_rate"] == 256.0
    assert report["clinical_decision_support"] is False


def test_validate_intake_lists_missing_equipment_fields():
    report = validate_intake_metadata({"unit": "uV", "channel_name": "Fpz"})

    assert report["ready_for_bench"] is False
    assert report["status"] == "incomplete"
    assert report["missing_fields"] == [
        "device_manufacturer",
        "device_model",
        "firmware",
        "bridge",
        "sampling_rate",
        "reference",
        "montage",
        "nominal_range",
        "processing_applied",
    ]


def test_validate_intake_rejects_invalid_sampling_rate():
    invalid = {**COMPLETE_MANIFEST, "sampling_rate": 0}

    with pytest.raises(ValueError, match="sampling_rate"):
        validate_intake_metadata(invalid)


def test_validate_intake_rejects_unit_that_would_skip_conversion():
    invalid = {**COMPLETE_MANIFEST, "unit": "mV"}

    report = validate_intake_metadata(invalid)

    assert report["ready_for_bench"] is False
    assert report["status"] == "incompatible"
    assert "microvolt" in report["compatibility_issues"][0]


def test_validate_intake_cli_returns_nonzero_for_incomplete_manifest(tmp_path, capsys):
    manifest = tmp_path / "incomplete.json"
    manifest.write_text(json.dumps({"unit": "uV"}), encoding="utf-8")

    exit_code = main(["validate-intake", "--metadata-file", str(manifest)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["ready_for_bench"] is False
    assert "device_model" in report["missing_fields"]


def test_validate_intake_cli_accepts_complete_manifest(tmp_path, capsys):
    manifest = tmp_path / "complete.json"
    manifest.write_text(json.dumps(COMPLETE_MANIFEST), encoding="utf-8")

    exit_code = main(["validate-intake", "--metadata-file", str(manifest)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["status"] == "ready_for_bench"


def test_stream_json_can_apply_intake_gate_before_inference(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    manifest = tmp_path / "complete.json"
    manifest.write_text(json.dumps(COMPLETE_MANIFEST), encoding="utf-8")
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                        "samples": np.sin(np.arange(1280) * 2 * np.pi * 10 / 256).tolist(),
                        "sampling_rate": 256,
                        "timestamps": [index / 256 for index in range(1280)],
                }
            )
        ),
    )
    report_path = tmp_path / "session.json"

    exit_code = main(
        [
            "stream-json",
            "--checkpoint",
            "unused.pt",
            "--metadata-file",
            str(manifest),
            "--require-metadata",
            "--require-intake",
            "--require-timestamps",
            "--fail-on-audit",
            "--report",
            str(report_path),
        ]
    )

    json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["intake"]["ready_for_bench"] is True
    assert report["runtime"]["require_intake"] is True

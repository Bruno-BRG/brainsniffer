import io
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainsniffer.cli import _decode_json_chunk, _metadata_from_config, main
from brainsniffer.config import PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase
from brainsniffer.data.preprocess import WindowedEEG
from brainsniffer.data.split import CaseSplit, GroupFold
from brainsniffer.pipeline.baseline import BaselineResult, CrossValidationResult


def test_cli_help_preserves_json_and_replay_without_lsl(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "stream-lsl" not in help_text
    for command in ("stream-json", "audit-json", "validate-intake", "replay"):
        assert command in help_text
    with pytest.raises(SystemExit) as result:
        main(["stream-lsl"])
    assert result.value.code == 2


def test_decode_json_chunk_validates_contract():
    samples, timestamps, rate, metadata = _decode_json_chunk(
        {
            "samples": [0.1, 0.2],
            "timestamps": [10.0, 10.01],
            "sampling_rate": 100,
        }
    )
    assert samples == [0.1, 0.2]
    assert timestamps == [10.0, 10.01]
    assert rate == 100.0
    assert metadata is None


def test_decode_json_chunk_rejects_missing_samples():
    with pytest.raises(ValueError, match="samples"):
        _decode_json_chunk({"sampling_rate": 128})


def test_decode_json_chunk_accepts_raw_sample_list():
    samples, timestamps, rate, metadata = _decode_json_chunk([1.0, 2.0])
    assert samples == [1.0, 2.0]
    assert timestamps is None
    assert rate is None
    assert metadata is None


def test_decode_json_chunk_accepts_source_metadata():
    *_, metadata = _decode_json_chunk(
        {
            "samples": [0.1],
            "metadata": {
                "unit": "uV",
                "channel_name": "Fpz",
                "reference": "linked ears",
                "montage": "frontal referenced",
            },
        }
    )
    assert metadata == {
        "unit": "uV",
        "channel_name": "Fpz",
        "reference": "linked ears",
        "montage": "frontal referenced",
    }


def test_metadata_file_merges_with_flags_and_rejects_conflicts(tmp_path):
    manifest = tmp_path / "stream-metadata.json"
    manifest.write_text(
        json.dumps(
            {
                "unit": "uV",
                "channel_name": "Fpz",
                "reference": "linked ears",
                "montage": "frontal referenced",
                "device_model": "bench-eeg",
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        metadata_file=manifest,
        unit=None,
        channel_name=None,
        reference=None,
        montage=None,
    )
    assert _metadata_from_config(args)["device_model"] == "bench-eeg"

    args.unit = "mV"
    with pytest.raises(ValueError, match="metadata diverge"):
        _metadata_from_config(args)


def test_stream_json_accepts_complete_metadata_file(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    manifest = tmp_path / "stream-metadata.json"
    manifest.write_text(
        json.dumps(
            {
                "unit": "uV",
                "channel_name": "Fpz",
                "reference": "linked ears",
                "montage": "frontal referenced",
            }
        ),
        encoding="utf-8",
    )
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
                    "samples": np.sin(np.arange(640) * 2 * np.pi * 10 / 128).tolist(),
                    "sampling_rate": 128,
                    "timestamps": (np.arange(640) / 128).tolist(),
                    "metadata": {"source_name": "bench"},
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
                "--require-timestamps",
                "--fail-on-audit",
                "--report",
            str(report_path),
        ]
    )

    json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["audit"]["metadata_complete"] is True
    assert report["audit"]["metadata"]["source_name"] == "bench"
    assert report["audit"]["timestamps_present"] is True
    assert report["audit"]["ok"] is True
    assert report["runtime"]["fail_on_audit"] is True


def test_stream_json_writes_partial_report_when_metadata_file_is_invalid(
    monkeypatch, tmp_path
):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    report_path = tmp_path / "invalid-manifest-session.json"

    with pytest.raises(ValueError, match="metadata file"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--metadata-file",
                str(tmp_path / "missing.json"),
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert "metadata file" in report["error"]
    assert report["predictions"]["count"] == 0


def test_stream_json_rejects_rate_change_without_reset(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    lines = [
        json.dumps({"samples": [0.0], "sampling_rate": 256}),
        json.dumps({"samples": [0.0], "sampling_rate": 128}),
    ]
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines)))
    report_path = tmp_path / "failed-session.json"
    with pytest.raises(ValueError, match="não pode mudar"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--report",
                str(report_path),
            ]
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["error"] == "sampling_rate não pode mudar durante o stream"
    assert report["report_version"] == 2
    assert report["scope"] == {
        "intended_use": "research_only",
        "clinical_decision_support": False,
        "controls_anesthetic_delivery": False,
    }
    assert report["runtime"]["preprocess_config"]["sampling_rate"] == 128
    assert report["runtime"]["stride_seconds"] == 1.0
    assert report["runtime"]["max_gap_factor"] == 1.5
    assert report["audit"]["sample_count"] == 1


def test_stream_json_require_metadata_fails_before_inference(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"samples": [0.0]})))
    report_path = tmp_path / "metadata-required.json"
    with pytest.raises(ValueError, match="metadata obrigatório incompleto"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--require-metadata",
                "--report",
                str(report_path),
            ]
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["predictions"]["count"] == 0
    assert report["audit"]["metadata_complete"] is False
    assert report["audit"]["metadata_missing"] == ["unit", "channel_name", "reference", "montage"]


def test_evaluate_recomputes_saved_test_cases(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    (tmp_path / "case1.mat").touch()
    windows = WindowedEEG(
        signals=torch.zeros(2, 1, 640).numpy(),
        bis=torch.tensor([55.0, 55.0]).numpy(),
        case_ids=np.asarray(["case1", "case1"]),
        start_seconds=torch.tensor([0.0, 5.0]).numpy(),
        quality=torch.ones(2).numpy(),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (
            ConstantModel(),
            PreprocessConfig(),
            {
                "min_quality": 0.2,
                "split": {"test_cases": ["case1"]},
                "test_metrics": {"mae": 0.0},
            },
        ),
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)
    report_path = tmp_path / "holdout.json"

    exit_code = main(
        [
            "evaluate",
            "--data-dir",
            str(tmp_path),
            "--checkpoint",
            "unused.pt",
            "--report",
            str(report_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    saved_result = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["n_test_windows"] == 2
    assert result["recomputed_test_metrics"]["mae"] == 0.0
    assert saved_result == result
    assert saved_result["scope"] == "research_only"
    assert saved_result["input_files"][0]["sha256"]
    assert saved_result["raw_eeg_in_report"] is False
    assert "signals" not in saved_result
    assert saved_result["case_bootstrap"] == {}


def test_evaluate_flags_smoke_scope_mismatch(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    (tmp_path / "case1.mat").touch()
    windows = WindowedEEG(
        signals=torch.zeros(2, 1, 640).numpy(),
        bis=torch.tensor([55.0, 55.0]).numpy(),
        case_ids=np.asarray(["case1", "case1"]),
        start_seconds=torch.tensor([0.0, 5.0]).numpy(),
        quality=torch.ones(2).numpy(),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (
            ConstantModel(),
            PreprocessConfig(),
            {
                "min_quality": 0.2,
                "split": {"test_cases": ["case1"]},
                "test_metrics": {"mae": 0.0},
                "dataset_summary": {"n_windows": 1},
            },
        ),
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)

    exit_code = main(["evaluate", "--data-dir", str(tmp_path), "--checkpoint", "unused.pt"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["dataset_summary_match"] is False
    assert "smoke test" in result["warnings"][0]


def test_evaluate_offset_reuses_saved_holdout_without_retraining(
    monkeypatch, tmp_path, capsys
):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    (tmp_path / "case1.mat").touch()
    windows = WindowedEEG(
        signals=np.zeros((2, 1, 640), dtype=np.float32),
        bis=np.asarray([50.0, 60.0], dtype=np.float32),
        case_ids=np.asarray(["case1", "case1"]),
        start_seconds=np.asarray([0.0, 5.0], dtype=np.float32),
        quality=np.ones(2, dtype=np.float32),
    )
    offsets_seen: list[float] = []
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda checkpoint: (
            ConstantModel(),
            PreprocessConfig(),
            {"min_quality": 0.2, "split": {"test_cases": ["case1"]}},
        ),
    )

    def fake_load_windows(paths, config, *, min_quality):
        offsets_seen.append(config.label_offset_seconds)
        assert [path.stem for path in paths] == ["case1"]
        assert min_quality == 0.2
        return windows

    monkeypatch.setattr("brainsniffer.cli.load_windows", fake_load_windows)
    report_path = tmp_path / "offsets.json"

    exit_code = main(
        [
            "evaluate-offset",
            "--data-dir",
            str(tmp_path),
            "--checkpoint",
            "unused.pt",
            "--offset-seconds",
            "-5",
            "0",
            "5",
            "--report",
            str(report_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    saved_result = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert offsets_seen == [-5.0, 0.0, 5.0]
    assert [row["offset_seconds"] for row in result["results"]] == [-5.0, 0.0, 5.0]
    assert all(row["metrics"]["mae"] == 5.0 for row in result["results"])
    assert result["retrained"] is False
    assert result["retained_split_by_case"] is True
    assert saved_result["results"] == result["results"]
    assert saved_result["input_files"][0]["sha256"]


def test_evaluate_external_uses_explicit_files(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    path = tmp_path / "vitaldb_case1.npz"
    path.touch()
    windows = WindowedEEG(
        signals=np.zeros((2, 1, 640), dtype=np.float32),
        bis=np.asarray([55.0, 55.0], dtype=np.float32),
        case_ids=np.asarray(["vitaldb_case1", "vitaldb_case1"]),
        start_seconds=np.asarray([0.0, 5.0], dtype=np.float32),
        quality=np.ones(2, dtype=np.float32),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda checkpoint: (ConstantModel(), PreprocessConfig(), {"min_quality": 0.2}),
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(
        "brainsniffer.cli.load_case",
        lambda path, **kwargs: EEGCase(
            case_id="vitaldb_case1",
            eeg=np.zeros(640, dtype=np.float32),
            bis=np.asarray([55.0], dtype=np.float32),
        ),
    )

    exit_code = main(["evaluate-external", "--case", str(path), "--checkpoint", "unused.pt"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["case_ids"] == ["vitaldb_case1"]
    assert result["metrics"]["mae"] == 0.0
    assert result["per_case"][0]["n_windows"] == 2
    assert result["input_files"][0]["path"] == str(path)
    assert result["input_files"][0]["sha256"]
    assert result["input_diagnostics"][0]["nonfinite_count"] == 0
    assert result["data_handling"]["mode"] == "offline_evaluation"
    assert result["data_handling"]["raw_eeg_in_report"] is False
    assert result["case_bootstrap"] == {}


def test_evaluate_external_discovers_vitaldb_directory(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    first = tmp_path / "vitaldb_case1.npz"
    second = tmp_path / "vitaldb_case2.npz"
    first.write_bytes(b"case-one")
    second.write_bytes(b"case-two")
    windows = WindowedEEG(
        signals=np.zeros((2, 1, 640), dtype=np.float32),
        bis=np.asarray([55.0, 55.0], dtype=np.float32),
        case_ids=np.asarray(["vitaldb_case1", "vitaldb_case2"]),
        start_seconds=np.asarray([0.0, 0.0], dtype=np.float32),
        quality=np.ones(2, dtype=np.float32),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda checkpoint: (ConstantModel(), PreprocessConfig(), {"min_quality": 0.2}),
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(
        "brainsniffer.cli.load_case",
        lambda path, **kwargs: EEGCase(
            case_id=path.stem,
            eeg=np.zeros(640, dtype=np.float32),
            bis=np.asarray([55.0], dtype=np.float32),
        ),
    )

    exit_code = main(
        [
            "evaluate-external",
            "--data-dir",
            str(tmp_path),
            "--checkpoint",
            "unused.pt",
            "--bootstrap-samples",
            "10",
            "--bootstrap-seed",
            "9",
            "--report",
            str(tmp_path / "external-report.json"),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    saved_result = json.loads((tmp_path / "external-report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["files"] == [str(first), str(second)]
    assert result["case_ids"] == ["vitaldb_case1", "vitaldb_case2"]
    assert len(result["input_files"]) == 2
    assert len(result["input_diagnostics"]) == 2
    assert result["bootstrap_samples"] == 10
    assert result["bootstrap_seed"] == 9
    assert saved_result["scope"] == "research_only"
    assert saved_result["data_handling"] == result["data_handling"]


def test_audit_json_outputs_preflight_report(monkeypatch, capsys):
    lines = [
        json.dumps(
            {
                "samples": [0.0, 1.0],
                "sampling_rate": 100,
                "timestamps": [10.0, 10.01],
            }
        ),
        json.dumps(
            {
                "samples": [2.0, 3.0],
                "sampling_rate": 100,
                "timestamps": [10.02, 10.03],
            }
        ),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines)))

    exit_code = main(["audit-json"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["ok"] is True
    assert result["sample_count"] == 4
    assert result["timestamps_present"] is True


def test_audit_json_returns_nonzero_for_rejected_stream(monkeypatch, capsys):
    payload = {
        "samples": [0.0, float("nan")],
        "sampling_rate": 100,
        "timestamps": [10.0, 10.01],
        "metadata": {
            "unit": "uV",
            "channel_name": "Fpz",
            "reference": "linked ears",
            "montage": "frontal referenced",
        },
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    exit_code = main(["audit-json", "--require-metadata"])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["ok"] is False
    assert result["finite_fraction"] == 0.5
    assert result["metadata_complete"] is True


def test_stream_json_writes_privacy_preserving_session_report(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"samples": [0.0] * 640, "sampling_rate": 128})),
    )
    report_path = tmp_path / "session.json"

    exit_code = main(
        ["stream-json", "--checkpoint", "unused.pt", "--report", str(report_path)]
    )

    prediction = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert prediction["checkpoint_sha256"] is None
    assert report["source"] == "jsonl"
    assert report["audit"]["sample_count"] == 640
    assert report["predictions"]["count"] == 1
    assert report["predictions"]["abstentions"] == 1
    assert report["predictions"]["abstention_fraction"] == 1.0
    assert report["status"] == "completed"
    assert report["error"] is None
    assert "samples" not in report


def test_stream_json_fail_on_audit_writes_rejected_session_report(
    monkeypatch, tmp_path, capsys
):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"samples": [0.0] * 640, "sampling_rate": 128})),
    )
    report_path = tmp_path / "rejected-session.json"

    with pytest.raises(RuntimeError, match="auditoria do stream rejeitou"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--fail-on-audit",
                "--report",
                str(report_path),
            ]
        )

    assert capsys.readouterr().out == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["audit"]["ok"] is False
    assert report["predictions"]["count"] == 0
    assert report["runtime"]["fail_on_audit"] is True


def test_stream_json_fail_on_audit_rejects_timestamp_gap_before_inference(
    monkeypatch, tmp_path
):
    class ExplodingModel(torch.nn.Module):
        def forward(self, inputs):
            pytest.fail("o modelo não deveria receber uma sessão com lacuna")

    timestamps = np.arange(640, dtype=np.float64) / 128
    timestamps[320:] += 1.0
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ExplodingModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "samples": np.sin(np.arange(640) * 2 * np.pi * 10 / 128).tolist(),
                    "sampling_rate": 128,
                    "timestamps": timestamps.tolist(),
                }
            )
        ),
    )
    report_path = tmp_path / "gap-session.json"

    with pytest.raises(RuntimeError, match="auditoria do stream rejeitou"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--require-timestamps",
                "--fail-on-audit",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["audit"]["timestamp_gap_count"] == 1
    assert report["predictions"]["count"] == 0


def test_stream_json_requires_timestamps_before_inference(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"samples": [0.0] * 640})))
    report_path = tmp_path / "timestamps-required.json"

    with pytest.raises(ValueError, match="timestamps obrigatórios ausentes"):
        main(
            [
                "stream-json",
                "--checkpoint",
                "unused.pt",
                "--require-timestamps",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["predictions"]["count"] == 0
    assert report["runtime"]["require_timestamps"] is True
    assert report["audit"]["timestamps_required"] is True
    assert report["audit"]["timestamps_present"] is False


def test_stream_json_fails_closed_on_nonfinite_samples_and_writes_partial_report(
    monkeypatch, tmp_path
):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"samples": [0.0, float("nan")], "sampling_rate": 128})),
    )
    report_path = tmp_path / "nonfinite-session.json"

    with pytest.raises(ValueError, match="samples devem ser finitas"):
        main(["stream-json", "--checkpoint", "unused.pt", "--report", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["predictions"]["count"] == 0
    assert report["audit"]["sample_count"] == 2
    assert report["audit"]["finite_fraction"] == 0.5
    assert "samples" not in report


def test_replay_falls_back_to_vitaldb_npz(monkeypatch, tmp_path, capsys):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    vital_path = tmp_path / "vitaldb_case1.npz"
    vital_path.touch()
    prediction = SimpleNamespace(stage="general", smoothed_bis=55.0)
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_case",
        lambda path: (
            pytest.fail("o teste deve usar o fallback VitalDB")
            if path.name != "vitaldb_case1.npz"
            else EEGCase(
                case_id="vitaldb_case1",
                eeg=np.zeros(640, dtype=np.float32),
                bis=np.asarray([55.0], dtype=np.float32),
            )
        ),
    )
    monkeypatch.setattr("brainsniffer.cli.replay_case", lambda *args, **kwargs: [prediction])

    exit_code = main(["replay", "--case", "1", "--data-dir", str(tmp_path)])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["stage"] == "general"


def test_replay_rejects_nonfinite_recording_before_inference(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    case_path = tmp_path / "vitaldb_case1.npz"
    case_path.touch()
    case = EEGCase(
        case_id="vitaldb_case1",
        eeg=np.asarray([0.0, np.nan, 1.0], dtype=np.float32),
        bis=np.asarray([55.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr("brainsniffer.cli.load_case", lambda path: case)
    monkeypatch.setattr(
        "brainsniffer.cli.replay_case",
        lambda *args, **kwargs: pytest.fail("o modelo não deve receber dados não finitos"),
    )

    with pytest.raises(SystemExit, match="não finitas"):
        main(["replay", "--case", "1", "--data-dir", str(tmp_path)])


def _benchmark_windows() -> WindowedEEG:
    return WindowedEEG(
        signals=np.zeros((6, 1, 640), dtype=np.float32),
        bis=np.asarray([35.0, 45.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float32),
        case_ids=np.asarray(["case1", "case1", "case2", "case2", "case3", "case3"]),
        start_seconds=np.asarray([0.0, 5.0, 0.0, 5.0, 0.0, 5.0], dtype=np.float32),
        quality=np.ones(6, dtype=np.float32),
    )


def test_benchmark_baseline_report_is_deterministic_and_preserves_stdout(
    monkeypatch, tmp_path, capsys
):
    paths = [tmp_path / f"case{case_id}.mat" for case_id in range(1, 4)]
    for index, path in enumerate(paths, start=1):
        path.write_bytes(f"input-{index}".encode())
    windows = _benchmark_windows()
    result = BaselineResult(
        split=CaseSplit(
            train_cases=("case3",),
            validation_cases=("case2",),
            test_cases=("case1",),
        ),
        validation_metrics={"mae": 2.0},
        test_metrics={"mae": 3.0},
        feature_names=("delta", "relative_delta"),
        seed=42,
        sampling_rate=128,
        estimator_parameters={
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)

    fit_calls = 0

    def fake_train(loaded, *, sampling_rate, training_config):
        nonlocal fit_calls
        fit_calls += 1
        assert loaded is windows
        assert sampling_rate == 128
        assert training_config.seed == 42
        if fit_calls == 3:
            return replace(
                result,
                validation_metrics={"mae": np.nextafter(2.0, 3.0).item()},
                test_metrics={"mae": np.nextafter(3.0, 4.0).item()},
            )
        return result

    monkeypatch.setattr("brainsniffer.cli.train_spectral_baseline", fake_train)
    command = ["benchmark-baseline", "--data-dir", str(tmp_path)]

    assert main(command) == 0
    legacy_stdout = capsys.readouterr().out
    report_path = tmp_path / "baseline.json"
    assert main([*command, "--report", str(report_path)]) == 0
    report_stdout = capsys.readouterr().out
    first_bytes = report_path.read_bytes()
    assert main([*command, "--report", str(report_path)]) == 0
    capsys.readouterr()

    report = json.loads(first_bytes)
    assert report_stdout == legacy_stdout
    assert report_path.read_bytes() == first_bytes
    assert report["scope"] == "research_only"
    assert report["protocol"] == "holdout"
    assert report["numeric_precision_decimal_places"] == 12
    assert report["seed"] == 42
    assert report["split"] == {
        "seed": 42,
        "test_cases": ["case1"],
        "train_cases": ["case3"],
        "unit": "case",
        "validation_cases": ["case2"],
    }
    assert report["dataset"]["n_cases"] == 3
    assert report["dataset"]["n_windows"] == 6
    assert report["dataset"]["partitions"]["test"] == {
        "n_cases": 1,
        "n_windows": 2,
    }
    assert report["effective_configuration"]["min_quality"] == 0.2
    assert report["effective_configuration"]["preprocess_config"]["causal"] is True
    assert report["effective_configuration"]["estimator"]["parameters"] == (
        result.estimator_parameters
    )
    assert report["feature_names"] == ["delta", "relative_delta"]
    assert report["metrics"] == {"test": {"mae": 3.0}, "validation": {"mae": 2.0}}
    assert [entry["size_bytes"] for entry in report["input_files"]] == [7, 7, 7]
    assert all(len(entry["sha256"]) == 64 for entry in report["input_files"])
    assert report["raw_eeg_in_report"] is False
    assert "signals" not in report


def test_benchmark_baseline_cv_report_records_grouped_case_folds(monkeypatch, tmp_path, capsys):
    for case_id in range(1, 4):
        (tmp_path / f"case{case_id}.mat").write_bytes(bytes([case_id]))
    windows = _benchmark_windows()
    case_folds = (
        GroupFold(fold_index=0, train_cases=("case3",), test_cases=("case1", "case2")),
        GroupFold(fold_index=1, train_cases=("case1", "case2"), test_cases=("case3",)),
    )
    estimator_parameters = (
        {"n_estimators": 100, "random_state": 42},
        {"n_estimators": 100, "random_state": 43},
    )
    result = CrossValidationResult(
        n_splits=2,
        folds=({"fold": 0.0, "mae": 2.0}, {"fold": 1.0, "mae": 4.0}),
        mean={"mae": 3.0},
        std={"mae": 1.0},
        case_folds=case_folds,
        feature_names=("delta",),
        seed=42,
        sampling_rate=128,
        estimator_parameters=estimator_parameters,
    )
    monkeypatch.setattr("brainsniffer.cli.load_windows", lambda *args, **kwargs: windows)

    def fake_cross_validate(loaded, *, sampling_rate, n_splits, seed):
        assert loaded is windows
        assert sampling_rate == 128
        assert n_splits == 2
        assert seed == 42
        return result

    monkeypatch.setattr("brainsniffer.cli.cross_validate_spectral_baseline", fake_cross_validate)
    report_path = tmp_path / "baseline-cv.json"

    assert (
        main(
            [
                "benchmark-baseline",
                "--data-dir",
                str(tmp_path),
                "--folds",
                "2",
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    stdout = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout == {
        "n_splits": 2,
        "folds": [{"fold": 0.0, "mae": 2.0}, {"fold": 1.0, "mae": 4.0}],
        "mean": {"mae": 3.0},
        "std": {"mae": 1.0},
    }
    assert report["protocol"] == "grouped_cross_validation"
    assert report["split"]["unit"] == "case"
    assert report["split"]["n_splits"] == 2
    assert report["split"]["folds"][0] == {
        "fold": 0,
        "test_cases": ["case1", "case2"],
        "test_n_windows": 4,
        "train_cases": ["case3"],
        "train_n_windows": 2,
    }
    assert (
        report["effective_configuration"]["estimator"]["parameters_by_fold"][1]["parameters"][
            "random_state"
        ]
        == 43
    )
    assert report["metrics"] == {
        "folds": [{"fold": 0.0, "mae": 2.0}, {"fold": 1.0, "mae": 4.0}],
        "mean": {"mae": 3.0},
        "std": {"mae": 1.0},
    }
    assert report["raw_eeg_in_report"] is False

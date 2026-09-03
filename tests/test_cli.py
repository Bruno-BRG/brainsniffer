import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brainsniffer.cli import _decode_json_chunk, _metadata_from_config, main
from brainsniffer.config import PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase
from brainsniffer.data.preprocess import WindowedEEG


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


def test_stream_lsl_finite_capture_fails_when_no_sample_arrives(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    class EmptySource:
        stream_name = "empty"
        sampling_rate = 128.0

        def read_chunk(self, **kwargs):
            return SimpleNamespace(
                samples=np.empty(0, dtype=np.float32),
                timestamps=np.empty(0, dtype=np.float64),
                sampling_rate=128.0,
            )

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr("brainsniffer.cli.LSLSource.connect", lambda **kwargs: EmptySource())
    report_path = tmp_path / "empty-lsl-session.json"

    with pytest.raises(RuntimeError, match="Nenhuma amostra EEG recebida"):
        main(
            [
                "stream-lsl",
                "--checkpoint",
                "unused.pt",
                "--duration",
                "0",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source"] == "lsl"
    assert report["status"] == "error"
    assert report["predictions"]["count"] == 0
    assert report["audit"]["sample_count"] == 0
    assert "há amostras não finitas" not in report["audit"]["warnings"]


def test_stream_lsl_rejects_silence_after_data_and_preserves_partial_report(
    monkeypatch, tmp_path
):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    class SourceWithSilence:
        stream_name = "silent-after-data"
        sampling_rate = 128.0

        def __init__(self):
            self.calls = 0

        def read_chunk(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    samples=np.sin(np.arange(640, dtype=np.float32)),
                    timestamps=np.arange(640, dtype=np.float64) / 128.0,
                    sampling_rate=128.0,
                )
            return SimpleNamespace(
                samples=np.empty(0, dtype=np.float32),
                timestamps=np.empty(0, dtype=np.float64),
                sampling_rate=128.0,
            )

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.LSLSource.connect", lambda **kwargs: SourceWithSilence()
    )
    report_path = tmp_path / "stale-lsl-session.json"

    with pytest.raises(RuntimeError, match="sem dados EEG"):
        main(
            [
                "stream-lsl",
                "--checkpoint",
                "unused.pt",
                "--duration",
                "0.1",
                "--stale-timeout",
                "0",
                "--fail-on-audit",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["predictions"]["count"] == 1
    assert report["runtime"]["stale_timeout_seconds"] == 0.0
    assert report["audit"]["sample_count"] == 640


def test_stream_lsl_connection_failure_writes_partial_report(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.LSLSource.connect",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("outlet indisponível")),
    )
    report_path = tmp_path / "connection-failure.json"

    with pytest.raises(RuntimeError, match="outlet indisponível"):
        main(
            [
                "stream-lsl",
                "--checkpoint",
                "unused.pt",
                "--unit",
                "uV",
                "--channel-name",
                "Fpz",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["error"] == "outlet indisponível"
    assert report["audit"]["metadata"]["unit"] == "uV"
    assert report["audit"]["sample_count"] == 0


def test_stream_lsl_rejects_metadata_conflict_with_descriptor(monkeypatch, tmp_path):
    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.full((inputs.shape[0],), 55.0, device=inputs.device)

    class DescriptorSource:
        stream_name = "descriptor"
        sampling_rate = 128.0
        metadata = {
            "unit": "uV",
            "channel_name": "Fpz",
            "reference": "linked ears",
            "montage": "frontal referenced",
        }

        def read_chunk(self, **kwargs):
            return SimpleNamespace(
                samples=np.empty(0, dtype=np.float32),
                timestamps=np.empty(0, dtype=np.float64),
                sampling_rate=128.0,
            )

    monkeypatch.setattr(
        "brainsniffer.cli.load_checkpoint",
        lambda path: (ConstantModel(), PreprocessConfig(), {}),
    )
    monkeypatch.setattr(
        "brainsniffer.cli.LSLSource.connect", lambda **kwargs: DescriptorSource()
    )
    report_path = tmp_path / "metadata-conflict.json"

    with pytest.raises(ValueError, match="metadata"):
        main(
            [
                "stream-lsl",
                "--checkpoint",
                "unused.pt",
                "--unit",
                "mV",
                "--report",
                str(report_path),
            ]
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["error"] == "metadata não pode mudar durante o stream"
    assert report["audit"]["metadata"]["unit"] == "uV"


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

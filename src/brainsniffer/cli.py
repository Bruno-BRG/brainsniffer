"""Command-line entrypoints for acquisition, inspection, training, and replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_MIN_SIGNAL_QUALITY,
    DEFAULT_SAMPLING_RATE,
    PreprocessConfig,
    TrainingConfig,
    default_data_dir,
    default_model_path,
)
from .data.figshare import available_case_ids, download_dataset, fetch_manifest
from .data.mat_reader import load_case
from .data.preprocess import (
    data_handling_policy,
    load_windows,
    signal_diagnostics,
    signal_quality,
    subset_windows,
)
from .data.vitaldb import download_vitaldb_case
from .pipeline.baseline import cross_validate_spectral_baseline, train_spectral_baseline
from .pipeline.benchmark import benchmark_latency
from .pipeline.intake import validate_intake_metadata
from .pipeline.metrics import bootstrap_case_metrics, compute_metrics
from .pipeline.realtime import replay_case
from .pipeline.stream_audit import StreamAudit
from .pipeline.streaming import LSLSource, StreamingResampler
from .pipeline.training import (
    build_file_manifest,
    load_checkpoint,
    predict_model,
    runtime_metadata,
    sha256_file,
    train_model,
    verify_file_manifest,
)


def _checkpoint_sha256(path: Path) -> str | None:
    """Return a checkpoint digest when the path exists (also supports mocked CLI tests)."""

    return sha256_file(path) if path.is_file() else None


def _write_stream_report(
    path: Path,
    *,
    source: str,
    checkpoint: Path,
    checkpoint_sha256: str | None,
    audit: StreamAudit,
    preprocess_config: PreprocessConfig,
    stride_seconds: float,
    prediction_count: int,
    abstention_count: int,
    stale_abstention_count: int,
    prediction_qualities: list[float],
    fail_on_audit: bool,
    require_intake: bool,
    stale_timeout_seconds: float | None,
    intake_report: dict[str, object],
    error: str | None = None,
) -> None:
    """Write privacy-preserving run metadata without retaining raw EEG samples."""

    path.parent.mkdir(parents=True, exist_ok=True)
    quality = np.asarray(prediction_qualities, dtype=np.float64)
    report = {
        "report_version": 2,
        "source": source,
        "status": "error" if error else "completed",
        "error": error,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "scope": {
            "intended_use": "research_only",
            "clinical_decision_support": False,
            "controls_anesthetic_delivery": False,
        },
        "runtime": {
            "environment": runtime_metadata(),
            "preprocess_config": asdict(preprocess_config),
            "stride_seconds": float(stride_seconds),
            "min_quality": audit.min_quality,
            "max_gap_factor": audit.max_gap_factor,
            "require_metadata": audit.require_metadata,
            "require_timestamps": audit.require_timestamps,
            "fail_on_audit": bool(fail_on_audit),
            "require_intake": bool(require_intake),
            "stale_timeout_seconds": stale_timeout_seconds,
        },
        "predictions": {
            "count": prediction_count,
            "abstentions": abstention_count,
            "stale_abstentions": stale_abstention_count,
            "abstention_fraction": (
                abstention_count / prediction_count if prediction_count else None
            ),
            "quality_min": float(quality.min()) if quality.size else None,
            "quality_mean": float(quality.mean()) if quality.size else None,
        },
        "audit": audit.report().as_dict(),
        "intake": intake_report,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _progress(filename: str, downloaded: int, expected: int) -> None:
    if expected:
        percent = downloaded / expected * 100
        print(f"\r{filename}: {percent:6.2f}%", end="", flush=True)
        if downloaded >= expected:
            print()


def _decode_json_chunk(
    payload: object,
) -> tuple[list[float], list[float] | None, float | None, dict[str, object] | None]:
    """Validate the small JSONL contract before touching streaming state."""

    if isinstance(payload, dict):
        if "samples" not in payload:
            raise ValueError("Cada objeto JSON deve conter o campo 'samples'")
        samples = payload["samples"]
        timestamps = payload.get("timestamps")
        payload_rate = payload.get("sampling_rate")
        payload_metadata = payload.get("metadata")
    else:
        samples = payload
        timestamps = None
        payload_rate = None
        payload_metadata = None

    if not isinstance(samples, list):
        raise ValueError("samples deve ser uma lista JSON de números")
    if timestamps is not None and not isinstance(timestamps, list):
        raise ValueError("timestamps deve ser uma lista JSON quando informado")
    if payload_rate is None:
        rate = None
    else:
        try:
            rate = float(payload_rate)
        except (TypeError, ValueError) as error:
            raise ValueError("sampling_rate deve ser numérico") from error
    if payload_metadata is not None and not isinstance(payload_metadata, dict):
        raise ValueError("metadata deve ser um objeto JSON quando informado")
    return samples, timestamps, rate, payload_metadata


def _metadata_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Collect explicit source metadata overrides from a stream command."""

    values = {
        "unit": getattr(args, "unit", None),
        "channel_name": getattr(args, "channel_name", None),
        "reference": getattr(args, "reference", None),
        "montage": getattr(args, "montage", None),
    }
    return {key: value for key, value in values.items() if value not in (None, "")}


def _load_metadata_file(path: Path) -> dict[str, object]:
    """Load a JSON source manifest without silently accepting another shape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"não foi possível ler o metadata file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"metadata file não é JSON válido: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("metadata file deve conter um objeto JSON")
    return {str(key): value for key, value in payload.items()}


def _metadata_from_config(args: argparse.Namespace) -> dict[str, object]:
    """Merge a versioned manifest file with explicit command-line fields."""

    metadata_file = getattr(args, "metadata_file", None)
    metadata = _load_metadata_file(metadata_file) if metadata_file is not None else {}
    return _merge_stream_metadata(metadata, _metadata_from_args(args))


def _merge_stream_metadata(
    base: dict[str, object], chunk: dict[str, object] | None
) -> dict[str, object]:
    """Merge manifests while rejecting a source that changes its identity."""

    merged = dict(base)
    if chunk is None:
        return merged
    for key, value in chunk.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"metadata diverge para o campo {key!r}")
        merged[key] = value
    return merged


def _add_stream_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="manifesto JSON versionado da unidade, canal, referência e montagem",
    )
    parser.add_argument("--unit", default=None, help="unidade do canal, por exemplo uV")
    parser.add_argument("--channel-name", default=None, help="nome/posição do canal")
    parser.add_argument("--reference", default=None, help="referência elétrica do canal")
    parser.add_argument("--montage", default=None, help="montagem do EEG")
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="falhar se unidade, canal, referência ou montagem não estiverem documentados",
    )
    parser.add_argument(
        "--require-timestamps",
        action="store_true",
        help="falhar se o bridge não fornecer um timestamp por amostra",
    )
    parser.add_argument(
        "--require-intake",
        action="store_true",
        help="falhar se a ficha mínima do equipamento não estiver completa",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BrainSniffer: pesquisa EEG e profundidade anestésica"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="baixar casos públicos do Figshare")
    download.add_argument("--out", type=Path, default=default_data_dir())
    download.add_argument("--cases", type=int, nargs="*", help="IDs dos casos; vazio baixa todos")
    download.add_argument("--overwrite", action="store_true")

    vital_download = subparsers.add_parser(
        "download-vitaldb",
        help="baixar seletivamente um caso VitalDB para avaliação externa exploratória",
    )
    vital_download.add_argument("--case", type=int, action="append", required=True)
    vital_download.add_argument("--out", type=Path, default=Path("data/vitaldb"))
    vital_download.add_argument("--overwrite", action="store_true")

    inspect = subparsers.add_parser("inspect-data", help="mostrar metadados e casos locais")
    inspect.add_argument("--data-dir", type=Path, default=default_data_dir())

    train = subparsers.add_parser("train", help="treinar a CNN com divisão por caso")
    train.add_argument("--data-dir", type=Path, default=default_data_dir())
    train.add_argument("--checkpoint", type=Path, default=default_model_path())
    train.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    train.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)
    train.add_argument("--max-windows", type=int, default=None)
    train.add_argument(
        "--label-offset-seconds",
        type=float,
        default=0.0,
        help="deslocamento do rótulo BIS em relação ao início da janela (positivo=futuro)",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="recalcular as métricas do checkpoint em casos de teste salvos"
    )
    evaluate.add_argument("--data-dir", type=Path, default=default_data_dir())
    evaluate.add_argument("--checkpoint", type=Path, default=default_model_path())
    evaluate.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="sobrescrever o limiar salvo no checkpoint",
    )
    evaluate.add_argument(
        "--report",
        type=Path,
        default=None,
        help="salvar o relatório JSON do holdout com manifesto dos casos testados",
    )
    evaluate.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="reamostragens por caso para o intervalo exploratório",
    )
    evaluate.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="seed da reamostragem por caso",
    )

    offset_eval = subparsers.add_parser(
        "evaluate-offset",
        help="medir sensibilidade ao alinhamento temporal do rótulo BIS no holdout",
    )
    offset_eval.add_argument("--data-dir", type=Path, default=default_data_dir())
    offset_eval.add_argument("--checkpoint", type=Path, default=default_model_path())
    offset_eval.add_argument(
        "--offset-seconds",
        type=float,
        nargs="+",
        default=[-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        help="offsets a testar; positivo associa a janela a um BIS posterior",
    )
    offset_eval.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="sobrescrever o limiar salvo no checkpoint",
    )
    offset_eval.add_argument(
        "--report",
        type=Path,
        default=None,
        help="salvar o resultado JSON com manifesto dos arquivos de teste",
    )

    external = subparsers.add_parser(
        "evaluate-external", help="avaliar um ou mais arquivos normalizados sem retreinar"
    )
    external_inputs = external.add_mutually_exclusive_group(required=True)
    external_inputs.add_argument(
        "--case",
        type=Path,
        action="append",
        help="arquivo externo .npz; repita para vários casos",
    )
    external_inputs.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="diretório que contém arquivos vitaldb_case*.npz",
    )
    external.add_argument("--checkpoint", type=Path, default=default_model_path())
    external.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="sobrescrever o limiar salvo no checkpoint",
    )
    external.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="reamostragens por caso para intervalo exploratório",
    )
    external.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="semente determinística do bootstrap agrupado por caso",
    )
    external.add_argument(
        "--report",
        type=Path,
        default=None,
        help="salvar o relatório JSON completo da avaliação externa",
    )

    audit = subparsers.add_parser(
        "audit-json", help="fazer preflight de um stream JSONL sem carregar um modelo"
    )
    audit.add_argument(
        "--source-rate",
        type=float,
        default=None,
        help="taxa dos chunks quando o JSON não informar sampling_rate",
    )
    audit.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)
    audit.add_argument("--max-gap-factor", type=float, default=1.5)
    _add_stream_metadata_arguments(audit)

    intake = subparsers.add_parser(
        "validate-intake",
        help="validar a ficha do equipamento antes da bancada, sem processar EEG",
    )
    intake.add_argument(
        "--metadata-file",
        type=Path,
        required=True,
        help="manifesto JSON da unidade, canal, referência, montagem e equipamento",
    )

    replay = subparsers.add_parser("replay", help="simular inferência causal em fluxo")
    replay.add_argument("--data-dir", type=Path, default=default_data_dir())
    replay.add_argument("--case", type=int, required=True)
    replay.add_argument(
        "--file",
        type=Path,
        default=None,
        help="arquivo explícito .mat ou .npz; se omitido, procura caseN.mat e vitaldb_caseN.npz",
    )
    replay.add_argument("--checkpoint", type=Path, default=default_model_path())
    replay.add_argument("--stride", type=float, default=1.0)
    replay.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)

    baseline = subparsers.add_parser(
        "benchmark-baseline", help="comparar uma baseline espectral com a CNN"
    )
    baseline.add_argument("--data-dir", type=Path, default=default_data_dir())
    baseline.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)
    baseline.add_argument("--max-windows", type=int, default=None)
    baseline.add_argument("--folds", type=int, default=1, help="2–n casos para CV agrupada")

    stream = subparsers.add_parser("stream-json", help="consumir chunks JSON pela entrada padrão")
    stream.add_argument("--checkpoint", type=Path, default=default_model_path())
    stream.add_argument("--stride", type=float, default=1.0)
    stream.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)
    stream.add_argument(
        "--max-gap-factor",
        type=float,
        default=1.5,
        help="maior intervalo de timestamp aceito em múltiplos de 1/taxa",
    )
    stream.add_argument(
        "--source-rate",
        type=float,
        default=None,
        help="taxa dos chunks quando o JSON não informar sampling_rate",
    )
    stream.add_argument(
        "--report",
        type=Path,
        default=None,
        help="salvar relatório da sessão sem armazenar o EEG bruto",
    )
    stream.add_argument(
        "--fail-on-audit",
        action="store_true",
        help="terminar com erro assim que a auditoria rejeitar a sessão",
    )
    _add_stream_metadata_arguments(stream)

    lsl = subparsers.add_parser("stream-lsl", help="consumir um stream EEG via Lab Streaming Layer")
    lsl.add_argument("--checkpoint", type=Path, default=default_model_path())
    lsl.add_argument("--stream-name", default=None)
    lsl.add_argument("--stream-type", default="EEG")
    lsl.add_argument("--channel", type=int, default=0)
    lsl.add_argument("--stride", type=float, default=1.0)
    lsl.add_argument("--min-quality", type=float, default=DEFAULT_MIN_SIGNAL_QUALITY)
    lsl.add_argument(
        "--max-gap-factor",
        type=float,
        default=1.5,
        help="maior intervalo de timestamp aceito em múltiplos de 1/taxa",
    )
    lsl.add_argument("--duration", type=float, default=None)
    lsl.add_argument("--max-samples", type=int, default=256)
    lsl.add_argument(
        "--stale-timeout",
        type=float,
        default=2.0,
        help="segundos sem chunk antes de invalidar ou rejeitar a última estimativa",
    )
    lsl.add_argument(
        "--report",
        type=Path,
        default=None,
        help="salvar relatório da sessão sem armazenar o EEG bruto",
    )
    lsl.add_argument(
        "--fail-on-audit",
        action="store_true",
        help="terminar com erro assim que a auditoria rejeitar a sessão",
    )
    _add_stream_metadata_arguments(lsl)

    latency = subparsers.add_parser(
        "benchmark-latency", help="medir latência do caminho de inferência em streaming"
    )
    latency.add_argument("--checkpoint", type=Path, default=default_model_path())
    latency.add_argument("--iterations", type=int, default=30)
    latency.add_argument("--stride", type=float, default=1.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        manifest = fetch_manifest()
        print(f"Dataset: {manifest.get('title')} | casos: {len(available_case_ids(manifest))}")
        paths = download_dataset(
            args.out, cases=args.cases or None, overwrite=args.overwrite, progress=_progress
        )
        print(f"Arquivos disponíveis em {args.out}: {len(paths)}")
        return 0

    if args.command == "download-vitaldb":
        paths = [
            download_vitaldb_case(case_id, args.out, overwrite=args.overwrite)
            for case_id in args.case
        ]
        print(json.dumps({"dataset": "VitalDB", "files": [str(path) for path in paths]}, indent=2))
        return 0

    if args.command == "inspect-data":
        manifest = fetch_manifest()
        local = sorted(args.data_dir.glob("case*.mat"))
        print(
            json.dumps(
                {"dataset": manifest.get("title"), "local_cases": [p.stem for p in local]}, indent=2
            )
        )
        for path in local:
            case = load_case(path)
            quality = signal_quality(case.eeg, PreprocessConfig(sampling_rate=case.sampling_rate))
            print(
                f"{case.case_id}: {case.duration_seconds / 60:.1f} min, "
                f"EEG={case.eeg.size}, BIS={case.bis.size}, "
                f"quality_global={quality:.3f}"
            )
        return 0

    if args.command == "train":
        paths = sorted(args.data_dir.glob("case*.mat"))
        if not paths:
            raise SystemExit(
                f"Nenhum case*.mat encontrado em {args.data_dir}. Rode download-data primeiro."
            )
        preprocess_config = PreprocessConfig(label_offset_seconds=args.label_offset_seconds)
        windows = load_windows(paths, preprocess_config, min_quality=args.min_quality)
        windows = subset_windows(windows, args.max_windows)
        config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
        result = train_model(
            windows,
            preprocess_config=preprocess_config,
            training_config=config,
            checkpoint_path=args.checkpoint,
            min_quality=args.min_quality,
            input_files=paths,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
                    "validation": result.validation_metrics,
                    "test": result.test_metrics,
                    "device": result.device,
                    "dataset": result.dataset_summary,
                    "input_files": result.input_files,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate":
        model, preprocess, payload = load_checkpoint(args.checkpoint)
        paths = sorted(args.data_dir.glob("case*.mat"))
        if not paths:
            raise SystemExit(f"Nenhum case*.mat encontrado em {args.data_dir}.")
        verify_file_manifest(payload.get("input_files"))
        min_quality = (
            float(args.min_quality)
            if args.min_quality is not None
            else float(payload.get("min_quality", DEFAULT_MIN_SIGNAL_QUALITY))
        )
        windows = load_windows(paths, preprocess, min_quality=min_quality)
        warnings: list[str] = []
        stored_summary = payload.get("dataset_summary", {})
        if args.min_quality is None and isinstance(stored_summary, dict):
            stored_n_windows = stored_summary.get("n_windows")
            if stored_n_windows is not None and int(stored_n_windows) != windows.signals.shape[0]:
                warnings.append(
                    "O número de janelas local difere do checkpoint; "
                    "o resultado armazenado pode ser de um smoke test com --max-windows"
                )
        saved_split = payload.get("split", {})
        test_cases = tuple(str(case) for case in saved_split.get("test_cases", ()))
        if not test_cases:
            raise SystemExit("O checkpoint não contém casos de teste agrupados")
        test_mask = np.isin(windows.case_ids.astype(str), np.asarray(test_cases, dtype=str))
        if not test_mask.any():
            raise SystemExit("Nenhuma janela dos casos de teste está disponível localmente")
        prediction = predict_model(model, windows.signals[test_mask], device="cpu")
        recomputed = compute_metrics(windows.bis[test_mask], prediction)
        test_case_ids = windows.case_ids[test_mask].astype(str)
        case_bootstrap = (
            bootstrap_case_metrics(
                windows.bis[test_mask],
                prediction,
                test_case_ids,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            if np.unique(test_case_ids).size > 1
            else {}
        )
        test_paths = [path for path in paths if path.stem in test_cases]
        report_payload = {
            "scope": "research_only",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
            "data_dir": str(args.data_dir),
            "files": [str(path) for path in test_paths],
            "input_files": build_file_manifest(test_paths),
            "preprocess_config": asdict(preprocess),
            "test_cases": list(test_cases),
            "min_quality": min_quality,
            "n_test_windows": int(test_mask.sum()),
            "stored_test_metrics": payload.get("test_metrics", {}),
            "recomputed_test_metrics": recomputed,
            "case_bootstrap": case_bootstrap,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "dataset_summary_match": not warnings,
            "warnings": warnings,
            "retrained": False,
            "raw_eeg_in_report": False,
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(report_payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-offset":
        model, preprocess, payload = load_checkpoint(args.checkpoint)
        split_payload = payload.get("split", {})
        test_cases = (
            tuple(str(case_id) for case_id in split_payload.get("test_cases", []))
            if isinstance(split_payload, dict)
            else ()
        )
        if not test_cases:
            raise SystemExit("O checkpoint não contém os casos de teste necessários")
        available_paths = {
            path.stem: path for path in sorted(args.data_dir.glob("case*.mat"))
        }
        missing_cases = [case_id for case_id in test_cases if case_id not in available_paths]
        if missing_cases:
            raise SystemExit(
                "Casos de teste ausentes para a análise de offset: "
                + ", ".join(missing_cases)
            )
        paths = [available_paths[case_id] for case_id in test_cases]
        min_quality = (
            float(args.min_quality)
            if args.min_quality is not None
            else float(payload.get("min_quality", DEFAULT_MIN_SIGNAL_QUALITY))
        )
        results: list[dict[str, object]] = []
        for offset in args.offset_seconds:
            offset_preprocess = replace(
                preprocess,
                label_offset_seconds=float(offset),
            )
            windows = load_windows(paths, offset_preprocess, min_quality=min_quality)
            predictions = predict_model(model, windows.signals, device="cpu")
            results.append(
                {
                    "offset_seconds": float(offset),
                    "n_windows": int(windows.signals.shape[0]),
                    "metrics": compute_metrics(windows.bis, predictions),
                }
            )
        report_payload = {
            "scope": "research_only",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
            "data_dir": str(args.data_dir),
            "test_cases": list(test_cases),
            "input_files": build_file_manifest(paths),
            "preprocess_config": asdict(preprocess),
            "min_quality": min_quality,
            "label_offset_semantics": "positive associates each EEG window with a later BIS value",
            "retained_model_and_weights": True,
            "retained_split_by_case": True,
            "retrained": False,
            "results": results,
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(report_payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-external":
        model, preprocess, payload = load_checkpoint(args.checkpoint)
        external_cases = (
            list(args.case)
            if args.case is not None
            else sorted(args.data_dir.glob("vitaldb_case*.npz"))
        )
        if not external_cases:
            source = args.data_dir if args.data_dir is not None else "os arquivos informados"
            raise SystemExit(f"Nenhum arquivo externo encontrado em {source}.")
        missing = [str(path) for path in external_cases if not path.exists()]
        if missing:
            raise SystemExit(f"Arquivos externos ausentes: {', '.join(missing)}")
        min_quality = (
            float(args.min_quality)
            if args.min_quality is not None
            else float(payload.get("min_quality", DEFAULT_MIN_SIGNAL_QUALITY))
        )
        input_diagnostics = []
        for path in external_cases:
            case = load_case(path, sampling_rate=preprocess.sampling_rate)
            input_diagnostics.append(
                {
                    "path": str(path),
                    "case_id": case.case_id,
                    **signal_diagnostics(case.eeg, preprocess),
                }
            )
        windows = load_windows(external_cases, preprocess, min_quality=min_quality)
        if not windows.signals.shape[0]:
            raise SystemExit("Nenhuma janela válida nos arquivos externos")
        input_file_manifest = build_file_manifest(external_cases)
        prediction = predict_model(model, windows.signals, device="cpu")
        metrics = compute_metrics(windows.bis, prediction)
        case_ids = windows.case_ids.astype(str)
        per_case = []
        for case_id in sorted(np.unique(case_ids)):
            case_mask = case_ids == case_id
            per_case.append(
                {
                    "case_id": case_id,
                    "n_windows": int(case_mask.sum()),
                    **compute_metrics(windows.bis[case_mask], prediction[case_mask]),
                }
            )
        case_bootstrap = (
            bootstrap_case_metrics(
                windows.bis,
                prediction,
                case_ids,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            if len(per_case) > 1
            else {}
        )
        report_payload = {
            "scope": "research_only",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
            "files": [str(path) for path in external_cases],
            "input_files": input_file_manifest,
            "input_diagnostics": input_diagnostics,
            "data_handling": data_handling_policy(),
            "preprocess_config": asdict(preprocess),
            "retrained": False,
            "case_ids": sorted(np.unique(case_ids).tolist()),
            "min_quality": min_quality,
            "n_windows": int(windows.signals.shape[0]),
            "metrics": metrics,
            "per_case": per_case,
            "case_bootstrap": case_bootstrap,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(report_payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "audit-json":
        audit_report = StreamAudit(
            min_quality=args.min_quality,
            max_gap_factor=args.max_gap_factor,
            require_metadata=args.require_metadata,
            require_timestamps=args.require_timestamps,
        )
        metadata_base = _metadata_from_config(args)
        audit_report.set_metadata(metadata_base)
        intake_report = validate_intake_metadata(metadata_base)
        source_rate_override = args.source_rate
        configured_source_rate = source_rate_override
        assumed_rate_reported = False
        for line in sys.stdin:
            if not line.strip():
                continue
            samples, timestamps, payload_rate, payload_metadata = _decode_json_chunk(
                json.loads(line)
            )
            audit_report.set_metadata(_merge_stream_metadata(metadata_base, payload_metadata))
            intake_report = validate_intake_metadata(audit_report.report().metadata)
            if payload_rate is not None:
                if source_rate_override is not None and not math.isclose(
                    source_rate_override,
                    payload_rate,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("sampling_rate do JSON diverge de --source-rate")
                if configured_source_rate is not None and not math.isclose(
                    configured_source_rate,
                    payload_rate,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("sampling_rate não pode mudar durante o stream")
                configured_source_rate = payload_rate
            if configured_source_rate is None:
                configured_source_rate = DEFAULT_SAMPLING_RATE
                if not assumed_rate_reported:
                    print(
                        "JSON sem sampling_rate: assumindo "
                        f"{configured_source_rate:g} Hz; informe a taxa real no bridge",
                        file=sys.stderr,
                    )
                    assumed_rate_reported = True
            audit_report.push(
                samples,
                source_rate=configured_source_rate,
                timestamps=timestamps,
            )
        report = audit_report.report()
        report_payload = report.as_dict()
        report_payload["intake"] = intake_report
        if args.require_intake and not intake_report["ready_for_bench"]:
            report_payload["ok"] = False
            report_payload["warnings"] = [
                *report_payload["warnings"],
                "ficha do equipamento incompleta para o gate de bancada",
            ]
        print(json.dumps(report_payload, ensure_ascii=False, indent=2))
        return 0 if report_payload["ok"] else 1

    if args.command == "validate-intake":
        report = validate_intake_metadata(_load_metadata_file(args.metadata_file))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready_for_bench"] else 1

    if args.command == "benchmark-baseline":
        paths = sorted(args.data_dir.glob("case*.mat"))
        if not paths:
            raise SystemExit(f"Nenhum case*.mat encontrado em {args.data_dir}.")
        windows = subset_windows(
            load_windows(paths, min_quality=args.min_quality), args.max_windows
        )
        if args.folds > 1:
            result = cross_validate_spectral_baseline(windows, n_splits=args.folds)
            print(
                json.dumps(
                    {
                        "n_splits": result.n_splits,
                        "folds": result.folds,
                        "mean": result.mean,
                        "std": result.std,
                    },
                    indent=2,
                )
            )
            return 0
        result = train_spectral_baseline(windows)
        print(
            json.dumps(
                {"validation": result.validation_metrics, "test": result.test_metrics},
                indent=2,
            )
        )
        return 0

    if args.command == "replay":
        checkpoint_model, preprocess, payload = load_checkpoint(args.checkpoint)
        if args.file is not None:
            case_path = args.file
        else:
            candidates = (
                args.data_dir / f"case{args.case}.mat",
                args.data_dir / f"vitaldb_case{args.case}.npz",
            )
            case_path = next(
                (candidate for candidate in candidates if candidate.exists()), candidates[0]
            )
        case = load_case(case_path)
        nonfinite_count = int((~np.isfinite(case.eeg)).sum())
        if nonfinite_count:
            raise SystemExit(
                f"Replay rejeitado: {case_path} contém {nonfinite_count} amostras EEG "
                "não finitas. O caminho de streaming falha fechado; use a análise "
                "offline para inspeção/imputação documentada do dataset."
            )
        predictions = replay_case(
            checkpoint_model,
            case,
            preprocess,
            stride_seconds=args.stride,
            min_quality=args.min_quality,
        )
        if not predictions:
            raise SystemExit("O replay não gerou janelas suficientes")
        replay_report = predictions[-1].__dict__.copy()
        replay_report["checkpoint_sha256"] = _checkpoint_sha256(args.checkpoint)
        print(json.dumps(replay_report, indent=2))
        if payload.get("test_metrics"):
            print(json.dumps({"checkpoint_test_metrics": payload["test_metrics"]}, indent=2))
        return 0

    if args.command == "stream-json":
        model, preprocess, _ = load_checkpoint(args.checkpoint)
        checkpoint_sha256 = _checkpoint_sha256(args.checkpoint)
        from .pipeline.realtime import RealtimeEstimator

        estimator = RealtimeEstimator(
            model,
            preprocess,
            stride_seconds=args.stride,
            min_quality=args.min_quality,
        )
        source_rate_override = args.source_rate
        configured_source_rate = source_rate_override
        resampler = None
        assumed_rate_reported = False
        audit = StreamAudit(
            preprocess,
            min_quality=args.min_quality,
            max_gap_factor=args.max_gap_factor,
            require_metadata=args.require_metadata,
            require_timestamps=args.require_timestamps,
        )
        metadata_base: dict[str, object] = {}
        prediction_count = 0
        abstention_count = 0
        prediction_qualities: list[float] = []
        stream_error: str | None = None
        intake_report: dict[str, object] = validate_intake_metadata({})
        try:
            metadata_base = _metadata_from_config(args)
            audit.set_metadata(metadata_base)
            intake_report = validate_intake_metadata(metadata_base)
            for line in sys.stdin:
                if not line.strip():
                    continue
                samples, timestamps, payload_rate, payload_metadata = _decode_json_chunk(
                    json.loads(line)
                )
                audit.set_metadata(_merge_stream_metadata(metadata_base, payload_metadata))
                intake_report = validate_intake_metadata(audit.report().metadata)
                if args.require_metadata and not audit.metadata_complete():
                    missing = ", ".join(audit.report().metadata_missing)
                    raise ValueError(f"metadata obrigatório incompleto: {missing}")
                if args.require_intake and not intake_report["ready_for_bench"]:
                    missing = ", ".join(intake_report["missing_fields"])
                    raise ValueError(f"ficha do equipamento incompleta: {missing}")
                if payload_rate is not None:
                    if source_rate_override is not None and not math.isclose(
                        source_rate_override,
                        payload_rate,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError("sampling_rate do JSON diverge de --source-rate")
                    if resampler is not None and not math.isclose(
                        resampler.source_rate,
                        payload_rate,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        raise ValueError("sampling_rate não pode mudar durante o stream")
                    configured_source_rate = payload_rate
                if resampler is None:
                    if configured_source_rate is None:
                        configured_source_rate = preprocess.sampling_rate
                        if not assumed_rate_reported:
                            print(
                                "JSON sem sampling_rate: assumindo "
                                f"{configured_source_rate:g} Hz; informe a taxa real no bridge",
                                file=sys.stderr,
                            )
                            assumed_rate_reported = True
                    resampler = StreamingResampler(
                        configured_source_rate,
                        preprocess.sampling_rate,
                    )
                audit.push(
                    samples,
                    source_rate=configured_source_rate,
                    timestamps=timestamps,
                )
                if args.require_timestamps and audit.report().timestamps_present is not True:
                    raise ValueError("timestamps obrigatórios ausentes no stream")
                if args.fail_on_audit and not audit.report().ok:
                    raise RuntimeError("auditoria do stream rejeitou a sessão")
                converted = resampler.process(samples, timestamps=timestamps)
                outputs = estimator.push(
                    converted.samples,
                    timestamps=converted.timestamps,
                )
                for output in outputs:
                    prediction_count += 1
                    abstention_count += int(output.stage == "abstain")
                    prediction_qualities.append(output.quality)
                    record = asdict(output)
                    record["checkpoint_sha256"] = checkpoint_sha256
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            if resampler is not None:
                tail = resampler.flush()
                for output in estimator.push(tail.samples, timestamps=tail.timestamps):
                    prediction_count += 1
                    abstention_count += int(output.stage == "abstain")
                    prediction_qualities.append(output.quality)
                    record = asdict(output)
                    record["checkpoint_sha256"] = checkpoint_sha256
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.require_metadata and not audit.metadata_complete():
                missing = ", ".join(audit.report().metadata_missing)
                raise ValueError(f"metadata obrigatório incompleto: {missing}")
            intake_report = validate_intake_metadata(audit.report().metadata)
            if args.require_intake and not intake_report["ready_for_bench"]:
                missing = ", ".join(intake_report["missing_fields"])
                raise ValueError(f"ficha do equipamento incompleta: {missing}")
            if args.fail_on_audit and not audit.report().ok:
                raise RuntimeError("auditoria do stream rejeitou a sessão")
        except KeyboardInterrupt:
            stream_error = "interrompido pelo operador"
            raise
        except Exception as error:
            stream_error = str(error)
            raise
        finally:
            if args.report is not None:
                _write_stream_report(
                    args.report,
                    source="jsonl",
                    checkpoint=args.checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    audit=audit,
                    preprocess_config=preprocess,
                    stride_seconds=args.stride,
                    prediction_count=prediction_count,
                    abstention_count=abstention_count,
                    stale_abstention_count=0,
                    prediction_qualities=prediction_qualities,
                    fail_on_audit=args.fail_on_audit,
                    require_intake=args.require_intake,
                    stale_timeout_seconds=None,
                    intake_report=intake_report,
                    error=stream_error,
                )
        return 0

    if args.command == "stream-lsl":
        model, preprocess, _ = load_checkpoint(args.checkpoint)
        checkpoint_sha256 = _checkpoint_sha256(args.checkpoint)
        from .pipeline.realtime import RealtimeEstimator

        audit = StreamAudit(
            preprocess,
            min_quality=args.min_quality,
            max_gap_factor=args.max_gap_factor,
            require_metadata=args.require_metadata,
            require_timestamps=args.require_timestamps,
        )
        metadata: dict[str, object] = {}
        source: LSLSource | object | None = None
        prediction_count = 0
        abstention_count = 0
        stale_abstention_count = 0
        prediction_qualities: list[float] = []
        stream_error: str | None = None
        intake_report: dict[str, object] = validate_intake_metadata({})
        try:
            if args.stale_timeout < 0:
                raise ValueError("stale-timeout deve ser não negativo")
            metadata = _metadata_from_config(args)
            source = LSLSource.connect(
                stream_name=args.stream_name,
                stream_type=args.stream_type,
                channel_index=args.channel,
            )
            audit.set_metadata(getattr(source, "metadata", {}))
            audit.set_metadata(metadata)
            intake_report = validate_intake_metadata(audit.report().metadata)
            if args.require_metadata and not audit.metadata_complete():
                missing = ", ".join(audit.report().metadata_missing)
                raise RuntimeError(f"metadata obrigatório incompleto: {missing}")
            if args.require_intake and not intake_report["ready_for_bench"]:
                missing = ", ".join(intake_report["missing_fields"])
                raise RuntimeError(f"ficha do equipamento incompleta: {missing}")
            estimator = RealtimeEstimator(
                model,
                preprocess,
                stride_seconds=args.stride,
                min_quality=args.min_quality,
            )
            resampler = StreamingResampler(
                source.sampling_rate,
                preprocess.sampling_rate,
            )
            print(
                f"Conectado a {source.stream_name!r} ({source.sampling_rate:g} Hz, "
                f"canal {args.channel}); alvo do modelo={preprocess.sampling_rate} Hz",
                file=sys.stderr,
            )
            deadline = None if args.duration is None else time.monotonic() + args.duration
            last_data_wall = time.monotonic()
            stale_notified = False
            while deadline is None or time.monotonic() < deadline:
                chunk = source.read_chunk(max_samples=args.max_samples)
                if chunk.samples.size == 0:
                    silence_seconds = time.monotonic() - last_data_wall
                    if silence_seconds >= args.stale_timeout and not stale_notified:
                        if args.fail_on_audit:
                            raise RuntimeError(
                                "sem dados EEG no stream LSL por "
                                f"{silence_seconds:.2f} s; sessão rejeitada"
                            )
                        stale_output = estimator.mark_stale()
                        prediction_count += 1
                        abstention_count += 1
                        stale_abstention_count += 1
                        prediction_qualities.append(stale_output.quality)
                        record = asdict(stale_output)
                        record["checkpoint_sha256"] = checkpoint_sha256
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                        stale_notified = True
                    continue
                last_data_wall = time.monotonic()
                stale_notified = False
                audit.push(
                    chunk.samples,
                    source_rate=chunk.sampling_rate,
                    timestamps=chunk.timestamps,
                )
                if args.require_timestamps and audit.report().timestamps_present is not True:
                    raise RuntimeError("timestamps obrigatórios ausentes no stream")
                if args.fail_on_audit and not audit.report().ok:
                    raise RuntimeError("auditoria do stream rejeitou a sessão")
                converted = resampler.process(
                    chunk.samples,
                    timestamps=chunk.timestamps,
                )
                for output in estimator.push(converted.samples, timestamps=converted.timestamps):
                    prediction_count += 1
                    abstention_count += int(output.stage == "abstain")
                    prediction_qualities.append(output.quality)
                    record = asdict(output)
                    record["checkpoint_sha256"] = checkpoint_sha256
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.duration is not None and audit.report().sample_count == 0:
                raise RuntimeError(
                    "Nenhuma amostra EEG recebida durante a captura LSL; "
                    "verifique o outlet, o nome/tipo do stream e a conexão"
                )
            intake_report = validate_intake_metadata(audit.report().metadata)
            if args.require_intake and not intake_report["ready_for_bench"]:
                missing = ", ".join(intake_report["missing_fields"])
                raise RuntimeError(f"ficha do equipamento incompleta: {missing}")
            if args.fail_on_audit and not audit.report().ok:
                raise RuntimeError("auditoria do stream rejeitou a sessão")
        except KeyboardInterrupt:
            stream_error = "interrompido pelo operador"
            print("\nStream encerrado.", file=sys.stderr)
        except Exception as error:
            if source is None:
                audit.set_metadata(metadata)
            stream_error = str(error)
            raise
        finally:
            if args.report is not None:
                _write_stream_report(
                    args.report,
                    source="lsl",
                    checkpoint=args.checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    audit=audit,
                    preprocess_config=preprocess,
                    stride_seconds=args.stride,
                    prediction_count=prediction_count,
                    abstention_count=abstention_count,
                    stale_abstention_count=stale_abstention_count,
                    prediction_qualities=prediction_qualities,
                    fail_on_audit=args.fail_on_audit,
                    require_intake=args.require_intake,
                    stale_timeout_seconds=args.stale_timeout,
                    intake_report=intake_report,
                    error=stream_error,
                )
        return 0

    if args.command == "benchmark-latency":
        model, preprocess, _ = load_checkpoint(args.checkpoint)
        result = benchmark_latency(
            model,
            preprocess,
            iterations=args.iterations,
            stride_seconds=args.stride,
        )
        benchmark_report = asdict(result)
        benchmark_report["checkpoint"] = str(args.checkpoint)
        benchmark_report["checkpoint_sha256"] = _checkpoint_sha256(args.checkpoint)
        print(json.dumps(benchmark_report, indent=2))
        return 0

    return 1

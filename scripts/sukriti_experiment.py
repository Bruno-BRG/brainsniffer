"""Experimento Sukriti-inspired: bracos fatoriais sobre o Figshare (sem retreino historico).

NADA aqui sobrescreve checkpoints ou relatorios historicos. Cada braco grava
``models/brainsniffer_exp<LETRA>_*.pt`` (+ sidecar .json) e ``reports/exp_*_*.json``,
avaliados nos MESMOS benchmarks (5 Figshare + 15 VitalDB congelados).

Bracos (seed 42 em todos, split 13/5/5 por caso herdado do treino Figshare-only):
- A: baseline reproduzido (SmoothL1, 10 epocas, batch 128, clip 100, sem winsor).
- B: A + winsorizacao causal +/-200 uV com clip elevado a 200 uV.
- C: B + Huber(delta=5), 120 epocas, patience 18, batch 64, scheduler patience 6.
- D: C + z-score por sujeito split-safe (estatistica do treino congelada no
  checkpoint; fallback = media global do treino para sujeito unseen).

Uso:
    .venv/Scripts/python.exe scripts/sukriti_experiment.py --arm A   # treina + avalia
    .venv/Scripts/python.exe scripts/sukriti_experiment.py --arm B --max-windows 2000  # smoke
    .venv/Scripts/python.exe scripts/sukriti_experiment.py --arm C --skip-train --checkpoint models/brainsniffer_expC_huber.pt  # so avalia

Requer torch instalado e dados locais (data/raw + data/vitaldb).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from brainsniffer.cli import _write_deterministic_json  # noqa: E402
from brainsniffer.config import PreprocessConfig, TrainingConfig  # noqa: E402
from brainsniffer.data.preprocess import load_windows, subset_windows  # noqa: E402
from brainsniffer.pipeline.metrics import (  # noqa: E402
    bootstrap_case_metrics,
    bootstrap_case_prediction_probability,
    compute_metrics,
    prediction_probability,
)
from brainsniffer.pipeline.training import (  # noqa: E402
    load_checkpoint,
    predict_model,
    sha256_file,
    train_model,
    verify_file_manifest,
)

FIGSHARE_HOLDOUT_CASES = ("case19", "case3", "case4", "case5", "case9")
MIN_QUALITY = 0.2
N_BOOTSTRAP = 1000
SEED = 42

ARMS: dict[str, dict[str, object]] = {
    "A": {
        "label": "baseline reproduzido",
        "preprocess": PreprocessConfig(),
        "training": TrainingConfig(),
        "checkpoint": "models/brainsniffer_expA_baseline.pt",
    },
    "B": {
        "label": "A + winsor +/-200uV, clip 200uV",
        "preprocess": PreprocessConfig(winsor_uv=200.0, clip_uv=200.0),
        "training": TrainingConfig(),
        "checkpoint": "models/brainsniffer_expB_winsor.pt",
    },
    "C": {
        "label": "B + Huber(d=5), 120ep, patience 18, batch 64",
        "preprocess": PreprocessConfig(winsor_uv=200.0, clip_uv=200.0),
        "training": TrainingConfig(
            epochs=120,
            batch_size=64,
            loss_name="huber",
            loss_huber_delta=5.0,
            early_stopping_patience=18,
            scheduler_patience=6,
        ),
        "checkpoint": "models/brainsniffer_expC_huber.pt",
    },
    "D": {
        "label": "C + z-score por sujeito (estatistica do treino)",
        "preprocess": PreprocessConfig(winsor_uv=200.0, clip_uv=200.0),
        "training": TrainingConfig(
            epochs=120,
            batch_size=64,
            loss_name="huber",
            loss_huber_delta=5.0,
            early_stopping_patience=18,
            scheduler_patience=6,
        ),
        "checkpoint": "models/brainsniffer_expD_znorm.pt",
        "znorm": True,
    },
}

TMP_OUT = ROOT / "tmp" / "reanalysis"
REPORTS_OUT = ROOT / "reports"


def _log(message: str) -> None:
    print(f"[sukriti-exp] {message}", flush=True)


def _figshare_paths() -> list[Path]:
    return sorted((ROOT / "data" / "raw").glob("case*.mat"))


def _holdout_paths() -> list[Path]:
    return [ROOT / "data" / "raw" / f"{case}.mat" for case in FIGSHARE_HOLDOUT_CASES]


def _vitaldb_paths() -> list[Path]:
    return sorted((ROOT / "data" / "vitaldb").glob("vitaldb_case*.npz"))


def _znorm_stats(
    windows_signals: np.ndarray,
    case_ids: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, object]:
    """Media/desvio por caso calculados SOMENTE no treino (split-safe)."""

    per_case: dict[str, dict[str, float]] = {}
    pooled: list[np.ndarray] = []
    for case_id in sorted({str(item) for item in case_ids[train_mask]}):
        mask = (case_ids.astype(str) == case_id) & train_mask
        values = windows_signals[mask].astype(np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        per_case[case_id] = {"mean": mean, "std": std if std > 0 else 1.0}
        pooled.append(values)
    all_values = np.concatenate(pooled) if pooled else np.asarray([0.0])
    global_stats = {
        "mean": float(all_values.mean()),
        "std": float(all_values.std()) if float(all_values.std()) > 0 else 1.0,
    }
    return {"per_case": per_case, "global": global_stats}


def _apply_znorm(
    windows_signals: np.ndarray, case_ids: np.ndarray, stats: dict[str, object]
) -> np.ndarray:
    """Aplica z-score congelado (fallback = global do treino p/ caso unseen)."""

    per_case = stats["per_case"]
    glob = stats["global"]
    out = np.empty_like(windows_signals, dtype=np.float64)
    for index, case_id in enumerate(case_ids.astype(str)):
        entry = per_case.get(str(case_id), glob)
        out[index] = (windows_signals[index].astype(np.float64) - entry["mean"]) / entry["std"]
    return out.astype(np.float32)


def _evaluate(
    model: object,
    preprocess: PreprocessConfig,
    paths: list[Path],
    benchmark: str,
    arm_key: str,
    checkpoint_sha: str,
    znorm_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    from brainsniffer.pipeline.training import predict_model as _predict

    windows = load_windows(paths, preprocess, min_quality=MIN_QUALITY)
    signals = windows.signals
    if znorm_stats is not None:
        signals = _apply_znorm(signals, windows.case_ids, znorm_stats)
    prediction = np.asarray(_predict(model, signals, device="cpu"), dtype=np.float64)
    target = np.asarray(windows.bis, dtype=np.float64)
    case_ids = np.asarray(windows.case_ids, dtype=str)
    metrics = compute_metrics(target, prediction)
    pk = float(prediction_probability(target, prediction))
    case_bootstrap = bootstrap_case_metrics(
        target, prediction, case_ids, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    pk_bootstrap = bootstrap_case_prediction_probability(
        target, prediction, case_ids, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    per_case = []
    for case_id in sorted(np.unique(case_ids)):
        mask = case_ids == case_id
        case_metrics = compute_metrics(target[mask], prediction[mask])
        per_case.append(
            {
                "case_id": str(case_id),
                "n_windows": int(mask.sum()),
                "n": case_metrics.get("n"),
                "mae": case_metrics.get("mae"),
                "rmse": case_metrics.get("rmse"),
                "bias": case_metrics.get("bias"),
                "pearson_r": case_metrics.get("pearson_r"),
                "ccc": case_metrics.get("ccc"),
                "pk": float(prediction_probability(target[mask], prediction[mask])),
                "stage_accuracy": case_metrics.get("stage_accuracy"),
                "stage_macro_f1": case_metrics.get("stage_macro_f1"),
            }
        )
    case_maes = [row["mae"] for row in per_case if row["mae"] is not None]
    report = {
        "scope": "research_only",
        "analysis": "experimento Sukriti-inspired (braco novo, sem retreino historico)",
        "arm": arm_key,
        "benchmark": benchmark,
        "checkpoint": str(ARMS[arm_key]["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha,
        "benchmark_case_ids": [Path(path).stem for path in paths],
        "preprocess_config": asdict(preprocess),
        "min_quality": MIN_QUALITY,
        "n_windows": int(target.size),
        "n_cases": int(np.unique(case_ids).size),
        "metrics": metrics,
        "pk": pk,
        "pk_bootstrap": pk_bootstrap,
        "case_bootstrap": case_bootstrap,
        "per_case": per_case,
        "mean_per_case_mae": float(np.mean(case_maes)) if case_maes else None,
        "median_per_case_mae": float(np.median(case_maes)) if case_maes else None,
        "znorm_stats_summary": (
            {
                "n_cases": len(znorm_stats["per_case"]),
                "global": znorm_stats["global"],
                "note": "estatisticas do split de treino; fallback global p/ unseen",
            }
            if znorm_stats is not None
            else None
        ),
        "bootstrap_samples": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "retrained": True,
        "raw_eeg_in_report": False,
    }
    # NPZ por janela p/ EWMA futuro (mesmo formato do reviewer_reanalysis).
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    npz_path = TMP_OUT / f"exp{arm_key}_{benchmark}.npz"
    np.savez_compressed(
        npz_path, target=target, prediction=prediction, case_ids=case_ids.astype("U64")
    )
    report["prediction_file"] = str(npz_path.relative_to(ROOT)).replace("\\", "/")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--skip-vitaldb", action="store_true", help="avalia so o holdout Figshare"
    )
    args = parser.parse_args()

    spec = ARMS[args.arm]
    preprocess: PreprocessConfig = spec["preprocess"]
    training: TrainingConfig = spec["training"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / spec["checkpoint"]

    if not _figshare_paths():
        raise SystemExit("data/raw sem case*.mat; rode download-data primeiro.")
    if not args.skip_vitaldb and not _vitaldb_paths():
        raise SystemExit("data/vitaldb vazio; use --skip-vitaldb ou baixe os casos.")

    znorm_stats: dict[str, object] | None = None
    if not args.skip_train:
        from brainsniffer.data.split import split_case_ids

        _log(f"treinando braco {args.arm}: {spec['label']}")
        windows = load_windows(_figshare_paths(), preprocess, min_quality=MIN_QUALITY)
        windows = subset_windows(windows, args.max_windows)
        if spec.get("znorm"):
            split_ids = windows.case_ids.astype(str)
            split = split_case_ids(
                split_ids,
                validation_fraction=training.validation_fraction,
                test_fraction=training.test_fraction,
                seed=training.seed,
            )
            train_mask = np.isin(split_ids, np.asarray(split.train_cases, dtype=str))
            znorm_stats = _znorm_stats(windows.signals, windows.case_ids, train_mask)
            windows_signals = _apply_znorm(windows.signals, windows.case_ids, znorm_stats)
            from brainsniffer.data.preprocess import WindowedEEG

            windows = WindowedEEG(
                signals=windows_signals,
                bis=windows.bis,
                case_ids=windows.case_ids,
                start_seconds=windows.start_seconds,
                quality=windows.quality,
                group_ids=windows.group_ids,
                source_datasets=windows.source_datasets,
            )
            _log(f"znorm: {len(znorm_stats['per_case'])} casos de treino amostrados")
        result = train_model(
            windows,
            preprocess_config=preprocess,
            training_config=training,
            checkpoint_path=checkpoint,
            min_quality=MIN_QUALITY,
            input_files=_figshare_paths(),
        )
        _log(
            f"treino ok: best_epoch tested, val_mae={result.validation_metrics.get('mae'):.4f} "
            f"test_mae={result.test_metrics.get('mae'):.4f} device={result.device}"
        )
        if spec.get("znorm") and znorm_stats is not None:
            sidecar = checkpoint.with_suffix(".json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["znorm_stats"] = znorm_stats
            sidecar.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            _log("znorm_stats persistido no sidecar do checkpoint")

    _log(f"avaliando braco {args.arm} em {checkpoint}")
    model, loaded_preprocess, payload = load_checkpoint(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    verify_file_manifest(payload.get("input_files"))
    if spec.get("znorm"):
        znorm_stats = payload.get("znorm_stats")
        if znorm_stats is None:
            raise SystemExit("checkpoint D sem znorm_stats no sidecar; retreine sem --skip-train.")
    else:
        znorm_stats = None

    fig_report = _evaluate(
        model,
        loaded_preprocess,
        _holdout_paths(),
        "figshare_holdout",
        args.arm,
        checkpoint_sha,
        znorm_stats,
    )
    out_fig = REPORTS_OUT / f"exp{args.arm}_figshare_holdout.json"
    _write_deterministic_json(out_fig, fig_report)
    _log(
        f"Figshare: MAE={fig_report['metrics']['mae']:.4f} "
        f"r={fig_report['metrics']['pearson_r']:.4f} "
        f"CCC={fig_report['metrics'].get('ccc'):.4f} Pk={fig_report['pk']:.4f}"
    )

    if not args.skip_vitaldb:
        vit_report = _evaluate(
            model,
            loaded_preprocess,
            _vitaldb_paths(),
            "vitaldb_external",
            args.arm,
            checkpoint_sha,
            znorm_stats,
        )
        out_vit = REPORTS_OUT / f"exp{args.arm}_vitaldb_external.json"
        _write_deterministic_json(out_vit, vit_report)
        _log(
            f"VitalDB: MAE={vit_report['metrics']['mae']:.4f} "
            f"r={vit_report['metrics']['pearson_r']:.4f} "
            f"CCC={vit_report['metrics'].get('ccc'):.4f} Pk={vit_report['pk']:.4f}"
        )
    _log("concluido sem tocar em checkpoints ou reports historicos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

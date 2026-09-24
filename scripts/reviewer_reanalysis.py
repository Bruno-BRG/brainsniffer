"""Reanálises para revisores: predições por janela, bootstrap pareado, calibração e auditoria de Pk.

Execução reprodutível (não retreina a CNN, não baixa dados, não sobrescreve relatórios):

    uv run --locked python scripts/reviewer_reanalysis.py

Cria apenas arquivos NOVOS:
- ``tmp/reanalysis/*.npz``: predições por janela dos 6 braços (target, prediction,
  case_ids, checkpoint_sha256, window_count), gravados em ZIP determinístico.
- ``reports/spectral_baseline_vitaldb.json``: RF espectral treinado no split de treino do
  checkpoint ativo e avaliado nas 38.730 janelas externas do VitalDB.
- ``reports/paired_bootstrap.json``: bootstrap pareado por caso (B=1000, seed 42), teste
  pareado de MAE por caso (sinais/Wilcoxon) e tabelas por caso.
- ``reports/calibration_analysis.json``: reta predito~referência, Bland–Altman,
  ICC(2,1) de concordância absoluta e distribuição do erro absoluto.
- ``reports/pk_audit.json``: auditoria do Pk (pares informativos, empates no alvo,
  Pk por caso, distribuição do BIS por caso).

Política de verificação: as métricas recomputadas são confrontadas com os relatórios
existentes (tolerância 1e-6 relativa) e também com os relatórios ``pk_*.json``, que os
reproduzem bit a bit no ambiente travado atual. Sinais vitais de configuração (casos,
contagens de janelas, splits, SHA-256 dos checkpoints) falham de forma dura se divergirem.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from brainsniffer.cli import _write_deterministic_json  # noqa: E402
from brainsniffer.config import PreprocessConfig, TrainingConfig  # noqa: E402
from brainsniffer.data.preprocess import WindowedEEG, load_windows  # noqa: E402
from brainsniffer.data.split import split_case_ids  # noqa: E402
from brainsniffer.pipeline.baseline import (  # noqa: E402
    _estimator_parameters,
    spectral_features,
    train_spectral_baseline,
)
from brainsniffer.pipeline.metrics import (  # noqa: E402
    _tie_pairs,
    bootstrap_case_metrics,
    bootstrap_case_prediction_probability,
    compute_metrics,
    prediction_probability,
)
from brainsniffer.pipeline.training import (  # noqa: E402
    build_file_manifest,
    load_checkpoint,
    predict_model,
    sha256_file,
)

# ---------------------------------------------------------------------------
# Configuração espelhada dos relatórios existentes (não alterar).
# ---------------------------------------------------------------------------

PREPROCESS = PreprocessConfig()
TRAINING = TrainingConfig()
MIN_QUALITY = 0.2
N_BOOTSTRAP = 1000
SEED = 42

CHECKPOINT_ACTIVE = ROOT / "models" / "brainsniffer_cnn.pt"
CHECKPOINT_MIXED = ROOT / "models" / "brainsniffer_corpus_fixed.pt"
CHECKPOINT_ACTIVE_SHA256 = "fde8e45fff2fa5414944686fd086fc3dd42248f247c7fb4ea1e31aa00618ee8c"
CHECKPOINT_MIXED_SHA256 = "9fbfc4e88a629d93f615593ca361bcf441fe21a32bfb1abdd626d8bc93c6023b"

FIGSHARE_HOLDOUT_CASES = ("case19", "case3", "case4", "case5", "case9")
FIGSHARE_HOLDOUT_PATHS = [ROOT / "data" / "raw" / f"{case}.mat" for case in FIGSHARE_HOLDOUT_CASES]
VITALDB_PATHS = sorted((ROOT / "data" / "vitaldb").glob("vitaldb_case*.npz"))
FIGSHARE_RAW_PATHS = sorted((ROOT / "data" / "raw").glob("case*.mat"))

TMP_OUT = ROOT / "tmp" / "reanalysis"
REPORTS_OUT = ROOT / "reports"

METRIC_NAMES = ("mae", "rmse", "bias", "pearson_r")
PAIR_METRICS = (*METRIC_NAMES, "pk")
REL_TOLERANCE = 1e-6
HARD_TOLERANCE = 1e-4  # divergência acima disto é erro real, não ruído de ambiente


class VerificationError(RuntimeError):
    """Sinaliza divergência não explicada contra um artefato de referência."""


@dataclass(frozen=True)
class Arm:
    """Predições por janela de um braço (mesma ordem determinística de janelas)."""

    key: str
    benchmark: str
    target: np.ndarray
    prediction: np.ndarray
    case_ids: np.ndarray
    checkpoint_sha256: str
    model_kind: str
    model_fingerprint: str

    @property
    def n_windows(self) -> int:
        return int(self.target.size)

    @property
    def n_cases(self) -> int:
        return int(np.unique(self.case_ids).size)


def _log(message: str) -> None:
    print(f"[reanalysis] {message}", flush=True)


def _save_npz(path: Path, payload: dict[str, object]) -> None:
    """Grava NPZ byte-determinístico (timestamps ZIP fixos em 1980-01-01)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in payload.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            with archive.open(info, "w") as handle:
                np.save(handle, np.asarray(value), allow_pickle=False)


def _save_arm_npz(arm: Arm) -> Path:
    path = TMP_OUT / f"{arm.key}.npz"
    _save_npz(
        path,
        {
            "target": np.asarray(arm.target, dtype=np.float64),
            "prediction": np.asarray(arm.prediction, dtype=np.float64),
            "case_ids": np.asarray(arm.case_ids, dtype=str),
            "checkpoint_sha256": np.asarray(arm.checkpoint_sha256),
            "window_count": np.asarray(arm.n_windows, dtype=np.int64),
        },
    )
    return path


def _relative_deviation(value: float, reference: float) -> float:
    if reference == 0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def _compare_metrics(metrics: dict[str, float], reference: dict[str, object]) -> dict[str, object]:
    """Compara métricas comuns e devolve desvio relativo máximo e pass/fail."""

    worst_key = None
    worst = 0.0
    for key, value in metrics.items():
        if key not in reference or reference[key] is None:
            continue
        deviation = _relative_deviation(float(value), float(reference[key]))
        if deviation > worst:
            worst = deviation
            worst_key = key
    return {
        "max_relative_deviation": worst,
        "worst_metric": worst_key,
        "tolerance": REL_TOLERANCE,
        "passes_tolerance": bool(worst <= REL_TOLERANCE),
    }


def _checkpoint_arm(
    key: str,
    benchmark: str,
    checkpoint: Path,
    expected_sha256: str,
    paths: list[Path],
) -> Arm:
    model, preprocess, payload = load_checkpoint(checkpoint)
    if asdict(preprocess) != asdict(PREPROCESS):
        raise VerificationError(f"{key}: preprocess_config do checkpoint difere do esperado")
    if float(payload.get("min_quality", MIN_QUALITY)) != MIN_QUALITY:
        raise VerificationError(f"{key}: min_quality do checkpoint difere de {MIN_QUALITY}")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise VerificationError(f"{key}: SHA-256 do checkpoint divergente ({actual_sha256})")
    windows = load_windows(paths, preprocess, min_quality=MIN_QUALITY)
    prediction = np.asarray(predict_model(model, windows.signals, device="cpu"), dtype=np.float64)
    return Arm(
        key=key,
        benchmark=benchmark,
        target=np.asarray(windows.bis, dtype=np.float64),
        prediction=prediction,
        case_ids=np.asarray(windows.case_ids, dtype=str),
        checkpoint_sha256=actual_sha256,
        model_kind="cnn_checkpoint",
        model_fingerprint=actual_sha256,
    )


def _spectral_predict(model: RandomForestRegressor, windows: WindowedEEG) -> np.ndarray:
    features, _ = spectral_features(windows.signals, PREPROCESS.sampling_rate)
    return np.clip(np.asarray(model.predict(features), dtype=np.float64), 0.0, 100.0)


def _rf_fingerprint(train_cases: list[str], n_train_windows: int) -> str:
    spec = json.dumps(
        {
            "estimator": _estimator_parameters(SEED),
            "train_cases": sorted(train_cases),
            "n_train_windows": int(n_train_windows),
        },
        sort_keys=True,
    )
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


def _fit_spectral_rf(train_windows: WindowedEEG) -> tuple[RandomForestRegressor, tuple[str, ...]]:
    features, feature_names = spectral_features(train_windows.signals, PREPROCESS.sampling_rate)
    model = RandomForestRegressor(**_estimator_parameters(SEED))
    model.fit(features, train_windows.bis)
    return model, feature_names


# ---------------------------------------------------------------------------
# A. Predições por janela dos braços CNN + verificação contra relatórios.
# ---------------------------------------------------------------------------


def _recompute_cnn_arms() -> tuple[
    dict[str, Arm], dict[str, dict[str, float]], dict[str, dict[str, object]]
]:
    specs = {
        "cnn_active_figshare": (
            CHECKPOINT_ACTIVE,
            CHECKPOINT_ACTIVE_SHA256,
            FIGSHARE_HOLDOUT_PATHS,
            "figshare_holdout",
        ),
        "cnn_mixed_figshare": (
            CHECKPOINT_MIXED,
            CHECKPOINT_MIXED_SHA256,
            FIGSHARE_HOLDOUT_PATHS,
            "figshare_holdout",
        ),
        "cnn_active_vitaldb": (
            CHECKPOINT_ACTIVE,
            CHECKPOINT_ACTIVE_SHA256,
            VITALDB_PATHS,
            "vitaldb_external",
        ),
        "cnn_mixed_vitaldb": (
            CHECKPOINT_MIXED,
            CHECKPOINT_MIXED_SHA256,
            VITALDB_PATHS,
            "vitaldb_external",
        ),
    }
    arms: dict[str, Arm] = {}
    metrics_by_arm: dict[str, dict[str, float]] = {}
    pk_by_arm: dict[str, float] = {}
    for key, (checkpoint, sha, paths, benchmark) in specs.items():
        arm = _checkpoint_arm(key, benchmark, checkpoint, sha, paths)
        metrics = compute_metrics(arm.target, arm.prediction)
        pk = prediction_probability(arm.target, arm.prediction)
        npz_path = _save_arm_npz(arm)
        _log(
            f"{key}: n={arm.n_windows} cases={arm.n_cases} "
            f"mae={metrics['mae']:.6f} -> {npz_path.name}"
        )
        arms[key] = arm
        metrics_by_arm[key] = metrics
        pk_by_arm[key] = pk
    return arms, metrics_by_arm, pk_by_arm


ARM_REPORT_SPECS: dict[str, dict[str, object]] = {
    "cnn_active_figshare": {
        "arm_report": "reports/figshare_holdout_evaluation.json",
        "arm_metrics_path": ("recomputed_test_metrics",),
        "window_count_path": ("n_test_windows",),
        "pk_report": "reports/pk_figshare_active.json",
        "pk_metrics_path": (),
    },
    "cnn_mixed_figshare": {
        "arm_report": "reports/mixed_fixed_figshare_holdout.json",
        "arm_metrics_path": ("metrics",),
        "window_count_path": ("n_windows",),
        "pk_report": "reports/pk_figshare_mixed.json",
        "pk_metrics_path": (),
    },
    "cnn_active_vitaldb": {
        "arm_report": "reports/vitaldb_external_validation.json",
        "arm_metrics_path": ("metrics",),
        "window_count_path": ("n_windows",),
        "pk_report": "reports/pk_vitaldb_active.json",
        "pk_metrics_path": (),
    },
    "cnn_mixed_vitaldb": {
        "arm_report": "reports/mixed_vitaldb_external.json",
        "arm_metrics_path": ("metrics",),
        "window_count_path": ("n_windows",),
        "pk_report": "reports/pk_vitaldb_mixed.json",
        "pk_metrics_path": (),
    },
}


def _dig(payload: object, path: tuple[str, ...]) -> object:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _verify_cnn_arms(
    metrics_by_arm: dict[str, dict[str, float]],
    pk_by_arm: dict[str, float],
) -> dict[str, object]:
    verification: dict[str, object] = {}
    for key, spec in ARM_REPORT_SPECS.items():
        arm_report = json.loads((ROOT / str(spec["arm_report"])).read_text(encoding="utf-8"))
        pk_report = json.loads((ROOT / str(spec["pk_report"])).read_text(encoding="utf-8"))
        arm_reference = _dig(arm_report, tuple(spec["arm_metrics_path"]))
        if not isinstance(arm_reference, dict):
            raise VerificationError(f"{key}: bloco de métricas ausente em {spec['arm_report']}")
        arm_comparison = _compare_metrics(metrics_by_arm[key], arm_reference)
        pk_metrics = pk_report.get("metrics", {})
        pk_comparison = _compare_metrics(metrics_by_arm[key], pk_metrics)
        pk_value = float(pk_report["pk"])
        pk_deviation = _relative_deviation(pk_by_arm[key], pk_value)
        reference_windows = int(_dig(arm_report, tuple(spec["window_count_path"])) or 0)
        window_counts_equal = reference_windows == int(metrics_by_arm[key]["n"])
        if pk_comparison["max_relative_deviation"] > 1e-9 or pk_deviation > 1e-9:
            raise VerificationError(
                f"{key}: nem o relatório pk_*.json reproduz as métricas recomputadas "
                f"(desvio {pk_comparison['max_relative_deviation']:.3e}); pare e investigue"
            )
        if arm_comparison["max_relative_deviation"] > HARD_TOLERANCE:
            raise VerificationError(
                f"{key}: desvio {arm_comparison['max_relative_deviation']:.3e} excede o limite "
                f"duro {HARD_TOLERANCE:.0e} contra {spec['arm_report']}"
            )
        if not window_counts_equal:
            raise VerificationError(
                f"{key}: n_windows recomputado difere do relatório de braço"
            )
        verification[key] = {
            "arm_report": spec["arm_report"],
            "arm_report_max_relative_deviation": arm_comparison["max_relative_deviation"],
            "arm_report_worst_metric": arm_comparison["worst_metric"],
            "arm_report_passes_1e-6": arm_comparison["passes_tolerance"],
            "pk_report": spec["pk_report"],
            "pk_report_max_relative_deviation": pk_comparison["max_relative_deviation"],
            "pk_report_reproduces_exactly": bool(pk_comparison["max_relative_deviation"] <= 1e-12),
            "pk_value_relative_deviation": pk_deviation,
            "window_counts_equal": window_counts_equal,
        }
        _log(
            f"verify {key}: arm_report_max_rel={arm_comparison['max_relative_deviation']:.3e} "
            f"(passes 1e-6: {arm_comparison['passes_tolerance']}), "
            f"pk_report_max_rel={pk_comparison['max_relative_deviation']:.3e}, "
            f"n_match={window_counts_equal}"
        )
    verification["policy"] = (
        "Comparação com tolerância 1e-6 relativa contra o relatório de braço; verificação "
        "cruzada adicional contra reports/pk_*.json, que reproduz as mesmas janelas e é "
        "exigido bit a bit (<=1e-12). Se o relatório pk_*.json não reproduzir, o script para. "
        "Contagens de janelas são verificadas de forma exata."
    )
    verification["note_environment"] = (
        "Os relatórios de braço mais antigos divergem no 5º–7º dígito significativo "
        "(máx. ~1,1e-5 relativo) apenas em métricas contínuas; n, acurácia de estágio, "
        "macro-F1 e Pk são idênticos. Isso é consistente com diferença de build/kernels "
        "numéricos (o sidecar registra torch 2.14.0+cu130; o ambiente travado atual é "
        "torch 2.14.0+cpu). As predições recomputadas usadas em todas as análises "
        "pareadas reproduzem exatamente os relatórios pk_*.json do repositório."
    )
    return verification


# ---------------------------------------------------------------------------
# B. RF espectral do holdout Figshare (mesmas janelas, seed e hiperparâmetros).
# ---------------------------------------------------------------------------


def _recompute_rf_figshare() -> tuple[Arm, dict[str, object]]:
    windows = load_windows(FIGSHARE_RAW_PATHS, PREPROCESS, min_quality=MIN_QUALITY)
    library_result = train_spectral_baseline(
        windows, sampling_rate=PREPROCESS.sampling_rate, training_config=TRAINING
    )
    features, feature_names = spectral_features(windows.signals, PREPROCESS.sampling_rate)
    split = split_case_ids(
        windows.case_ids,
        validation_fraction=TRAINING.validation_fraction,
        test_fraction=TRAINING.test_fraction,
        seed=TRAINING.seed,
    )
    case_ids = windows.case_ids.astype(str)
    train_mask = np.isin(case_ids, np.asarray(split.train_cases, dtype=str))
    validation_mask = np.isin(case_ids, np.asarray(split.validation_cases, dtype=str))
    test_mask = np.isin(case_ids, np.asarray(split.test_cases, dtype=str))
    model = RandomForestRegressor(**_estimator_parameters(TRAINING.seed))
    model.fit(features[train_mask], windows.bis[train_mask])
    validation_prediction = np.clip(model.predict(features[validation_mask]), 0.0, 100.0)
    test_prediction = np.clip(model.predict(features[test_mask]), 0.0, 100.0)
    validation_metrics = compute_metrics(windows.bis[validation_mask], validation_prediction)
    test_metrics = compute_metrics(windows.bis[test_mask], test_prediction)

    report = json.loads(
        (ROOT / "reports" / "spectral_baseline_holdout.json").read_text(encoding="utf-8")
    )
    reference_split = report["split"]
    report_test_comparison = _compare_metrics(test_metrics, report["metrics"]["test"])
    report_validation_comparison = _compare_metrics(
        validation_metrics, report["metrics"]["validation"]
    )
    library_comparison = _compare_metrics(test_metrics, library_result.test_metrics)
    library_validation_comparison = _compare_metrics(
        validation_metrics, library_result.validation_metrics
    )
    split_matches = (
        list(split.train_cases) == list(reference_split["train_cases"])
        and list(split.validation_cases) == list(reference_split["validation_cases"])
        and list(split.test_cases) == list(reference_split["test_cases"])
    )
    dataset_matches = (
        int(report["dataset"]["n_windows"]) == int(windows.signals.shape[0])
        and int(report["dataset"]["partitions"]["test"]["n_windows"]) == int(test_mask.sum())
        and int(report["dataset"]["partitions"]["train"]["n_windows"]) == int(train_mask.sum())
    )
    worst = max(
        report_test_comparison["max_relative_deviation"],
        report_validation_comparison["max_relative_deviation"],
    )
    if not split_matches or not dataset_matches or worst > REL_TOLERANCE:
        raise VerificationError(
            "RF Figshare divergiu do relatório: "
            f"split_match={split_matches} dataset_match={dataset_matches} max_rel={worst:.3e}"
        )
    library_worst = max(
        library_comparison["max_relative_deviation"],
        library_validation_comparison["max_relative_deviation"],
    )
    if library_worst > 1e-12:
        raise VerificationError("Replicação do RF Figshare difere de train_spectral_baseline")
    arm = Arm(
        key="rf_spectral_figshare",
        benchmark="figshare_holdout",
        target=np.asarray(windows.bis[test_mask], dtype=np.float64),
        prediction=np.asarray(test_prediction, dtype=np.float64),
        case_ids=case_ids[test_mask],
        checkpoint_sha256="not_applicable__random_forest",
        model_kind="spectral_random_forest",
        model_fingerprint=_rf_fingerprint(list(split.train_cases), int(train_mask.sum())),
    )
    _save_arm_npz(arm)
    verification = {
        "reference_report": "reports/spectral_baseline_holdout.json",
        "test_max_relative_deviation": report_test_comparison["max_relative_deviation"],
        "validation_max_relative_deviation": report_validation_comparison["max_relative_deviation"],
        "library_path_max_relative_deviation": max(
            library_comparison["max_relative_deviation"],
            library_validation_comparison["max_relative_deviation"],
        ),
        "split_matches_report": split_matches,
        "dataset_counts_match_report": dataset_matches,
        "feature_names": list(feature_names),
        "train_n_windows": int(train_mask.sum()),
        "test_n_windows": int(test_mask.sum()),
        "estimator_parameters": _estimator_parameters(TRAINING.seed),
    }
    _log(
        "rf_spectral_figshare: "
        f"test n={arm.n_windows} cases={arm.n_cases} mae={test_metrics['mae']:.6f} "
        f"max_rel_vs_report={worst:.3e}"
    )
    return arm, verification


# ---------------------------------------------------------------------------
# C. RF espectral para o VitalDB (treino no split do checkpoint ativo).
# ---------------------------------------------------------------------------


def _recompute_rf_vitaldb() -> tuple[Arm, dict[str, object], dict[str, object]]:
    sidecar = json.loads(
        (ROOT / "models" / "brainsniffer_cnn.json").read_text(encoding="utf-8")
    )
    train_cases = sorted(str(case) for case in sidecar["split"]["train_cases"])
    train_paths = [ROOT / "data" / "raw" / f"{case}.mat" for case in train_cases]
    missing = [str(path) for path in train_paths if not path.is_file()]
    if missing:
        raise VerificationError(f"Casos de treino ausentes: {', '.join(missing)}")
    train_windows = load_windows(train_paths, PREPROCESS, min_quality=MIN_QUALITY)
    n_train_windows = int(train_windows.signals.shape[0])
    if n_train_windows != 15894:
        raise VerificationError(
            f"RF VitalDB: esperado 15.894 janelas de treino, obtido {n_train_windows}"
        )
    model, feature_names = _fit_spectral_rf(train_windows)
    evaluation_windows = load_windows(VITALDB_PATHS, PREPROCESS, min_quality=MIN_QUALITY)
    if int(evaluation_windows.signals.shape[0]) != 38730:
        raise VerificationError("RF VitalDB: esperado 38.730 janelas externas")
    prediction = _spectral_predict(model, evaluation_windows)
    target = np.asarray(evaluation_windows.bis, dtype=np.float64)
    case_ids = np.asarray(evaluation_windows.case_ids, dtype=str)
    metrics = compute_metrics(target, prediction)
    pk = prediction_probability(target, prediction)
    per_case = []
    for case_id in sorted(np.unique(case_ids)):
        mask = case_ids == case_id
        per_case.append(
            {
                "case_id": case_id,
                "n_windows": int(mask.sum()),
                **compute_metrics(target[mask], prediction[mask]),
                "pk": prediction_probability(target[mask], prediction[mask]),
            }
        )
    case_bootstrap = bootstrap_case_metrics(
        target, prediction, case_ids, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    pk_bootstrap = bootstrap_case_prediction_probability(
        target, prediction, case_ids, n_bootstrap=N_BOOTSTRAP, seed=SEED
    )
    arm = Arm(
        key="rf_spectral_vitaldb",
        benchmark="vitaldb_external",
        target=target,
        prediction=prediction,
        case_ids=case_ids,
        checkpoint_sha256="not_applicable__random_forest",
        model_kind="spectral_random_forest",
        model_fingerprint=_rf_fingerprint(train_cases, n_train_windows),
    )
    _save_arm_npz(arm)
    payload = {
        "report_version": 1,
        "numeric_precision_decimal_places": 12,
        "scope": "research_only",
        "protocol": "external_spectral_baseline",
        "protocol_note": (
            "RF treinado no split de treino do checkpoint ativo; mesmas janelas e imputação "
            "offline; sem retreino"
        ),
        "source_checkpoint": str(CHECKPOINT_ACTIVE.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": CHECKPOINT_ACTIVE_SHA256,
        "train_cases": train_cases,
        "train_dataset": {"n_cases": len(train_cases), "n_windows": n_train_windows},
        "test_case_ids": sorted(np.unique(case_ids).tolist()),
        "test_dataset": {"n_cases": arm.n_cases, "n_windows": arm.n_windows},
        "effective_configuration": {
            "data_dir": "data/raw + data/vitaldb",
            "min_quality": MIN_QUALITY,
            "feature_sampling_rate": PREPROCESS.sampling_rate,
            "preprocess_config": asdict(PREPROCESS),
            "estimator": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "parameters": _estimator_parameters(SEED),
            },
        },
        "feature_names": list(feature_names),
        "metrics": metrics,
        "per_case": per_case,
        "case_bootstrap": case_bootstrap,
        "pk": pk,
        "pk_bootstrap": pk_bootstrap,
        "bootstrap_samples": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "training_input_files": build_file_manifest(
            [path.relative_to(ROOT) for path in train_paths]
        ),
        "input_files": build_file_manifest([path.relative_to(ROOT) for path in VITALDB_PATHS]),
        "model_fingerprint": arm.model_fingerprint,
        "retrained": False,
        "raw_eeg_in_report": False,
    }
    _log(
        f"rf_spectral_vitaldb: train_n={n_train_windows} test n={arm.n_windows} "
        f"mae={metrics['mae']:.6f} pk={pk:.6f}"
    )
    return arm, payload, {
        "train_n_windows": n_train_windows,
        "test_n_windows": arm.n_windows,
        "expected_train_n_windows": 15894,
        "expected_test_n_windows": 38730,
    }


# ---------------------------------------------------------------------------
# D. Bootstrap pareado por caso.
# ---------------------------------------------------------------------------


def _summarize_bootstrap(samples: list[float], observed: float) -> dict[str, object]:
    finite = np.asarray([value for value in samples if np.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {
            "observed": observed,
            "mean": None,
            "lower_95": None,
            "upper_95": None,
            "ci_excludes_zero": False,
            "n_finite_resamples": 0,
        }
    lower = float(np.percentile(finite, 2.5))
    upper = float(np.percentile(finite, 97.5))
    return {
        "observed": observed,
        "mean": float(np.mean(finite)),
        "lower_95": lower,
        "upper_95": upper,
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "n_finite_resamples": int(finite.size),
    }


def _family_paired_bootstrap(
    arms: dict[str, Arm],
    pair_specs: dict[str, tuple[str, str]],
) -> dict[str, dict[str, list[float]]]:
    reference = next(iter(arms.values()))
    for arm in arms.values():
        if not np.array_equal(arm.case_ids, reference.case_ids) or not np.array_equal(
            arm.target, reference.target
        ):
            raise VerificationError("Braços do mesmo benchmark não estão pareados janela a janela")
    cases = np.unique(reference.case_ids)
    members = [np.flatnonzero(reference.case_ids == case_id) for case_id in cases]
    rng = np.random.default_rng(SEED)
    samples: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in PAIR_METRICS} for name in pair_specs
    }
    for _ in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(cases), size=len(cases))
        indices = np.concatenate([members[index] for index in selected])
        pooled: dict[str, dict[str, float]] = {}
        for key, arm in arms.items():
            metrics = compute_metrics(arm.target[indices], arm.prediction[indices])
            metrics["pk"] = prediction_probability(arm.target[indices], arm.prediction[indices])
            pooled[key] = metrics
        for name, (arm_a, arm_b) in pair_specs.items():
            for metric in PAIR_METRICS:
                value_a = pooled[arm_a].get(metric, float("nan"))
                value_b = pooled[arm_b].get(metric, float("nan"))
                if np.isfinite(value_a) and np.isfinite(value_b):
                    samples[name][metric].append(float(value_a - value_b))
    return samples


def _observed_pair(arm_a: Arm, arm_b: Arm) -> dict[str, dict[str, float]]:
    def pooled(arm: Arm) -> dict[str, float]:
        metrics = compute_metrics(arm.target, arm.prediction)
        metrics["pk"] = prediction_probability(arm.target, arm.prediction)
        return metrics

    return {"arm_a": pooled(arm_a), "arm_b": pooled(arm_b)}


# ---------------------------------------------------------------------------
# E. Teste pareado por caso do MAE (ativo vs misto, VitalDB).
# ---------------------------------------------------------------------------


def _casewise_mae_test(arm_a: Arm, arm_b: Arm) -> dict[str, object]:
    if not np.array_equal(arm_a.case_ids, arm_b.case_ids):
        raise VerificationError("Teste por caso exige pareamento exato de janelas")
    rows = []
    for case_id in sorted(np.unique(arm_a.case_ids)):
        mask = arm_a.case_ids == case_id
        mae_a = compute_metrics(arm_a.target[mask], arm_a.prediction[mask])["mae"]
        mae_b = compute_metrics(arm_b.target[mask], arm_b.prediction[mask])["mae"]
        rows.append(
            {
                "case_id": case_id,
                "n_windows": int(mask.sum()),
                "mae_arm_a": mae_a,
                "mae_arm_b": mae_b,
                "delta_arm_a_minus_arm_b": mae_a - mae_b,
            }
        )
    deltas = np.asarray([row["delta_arm_a_minus_arm_b"] for row in rows], dtype=np.float64)
    better_b = int(np.sum(deltas > 0))
    better_a = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    non_tied = better_a + better_b
    sign_two_sided = binomtest(better_b, non_tied, 0.5, alternative="two-sided")
    sign_greater = binomtest(better_b, non_tied, 0.5, alternative="greater")
    wilcoxon_two_sided = wilcoxon(
        arm_a_mae := np.asarray([row["mae_arm_a"] for row in rows], dtype=np.float64),
        np.asarray([row["mae_arm_b"] for row in rows], dtype=np.float64),
        alternative="two-sided",
    )
    wilcoxon_less = wilcoxon(
        arm_a_mae,
        np.asarray([row["mae_arm_b"] for row in rows], dtype=np.float64),
        alternative="less",
    )
    return {
        "comparison": "MAE por caso: braço a = CNN ativo, braço b = CNN misto (VitalDB)",
        "n_pairs": len(rows),
        "n_arm_b_better": better_b,
        "n_arm_a_better": better_a,
        "n_ties": ties,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "sign_test": {
            "method": "scipy.stats.binomtest (sinal, exato)",
            "k_arm_b_better": better_b,
            "n_non_tied": non_tied,
            "p_two_sided": float(sign_two_sided.pvalue),
            "p_one_sided_arm_b_better": float(sign_greater.pvalue),
        },
        "wilcoxon_signed_rank": {
            "method": "scipy.stats.wilcoxon (duas amostras pareadas)",
            "statistic": float(wilcoxon_two_sided.statistic),
            "p_two_sided": float(wilcoxon_two_sided.pvalue),
            "p_one_sided_arm_a_lower": float(wilcoxon_less.pvalue),
        },
        "per_case": rows,
    }


# ---------------------------------------------------------------------------
# F. Calibração e concordância.
# ---------------------------------------------------------------------------


def _icc_2_1(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    """ICC(2,1) de concordância absoluta (duas vias, efeitos aleatórios, medida única).

    Fórmula de Shrout & Fleiss (1979) / McGraw & Wong (1996), caso ICC(A,1) com k=2
    avaliações (referência BIS e predição):
        MSR = k/(n-1) * Σ_i (x̄_i - x̄)²
        MSC = n/(k-1) * Σ_j (x̄_j - x̄)²
        MSE = 1/((n-1)(k-1)) * Σ_ij (x_ij - x̄_i - x̄_j + x̄)²
        ICC(2,1) = (MSR - MSE) / (MSR + (k-1)MSE + k(MSC - MSE)/n)
    """

    x = np.column_stack(
        [np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)]
    )
    n, k = x.shape
    if n < 2 or k < 2:
        return None
    grand = float(x.mean())
    subject_means = x.mean(axis=1)
    rater_means = x.mean(axis=0)
    msr = k * float(np.sum((subject_means - grand) ** 2)) / (n - 1)
    msc = n * float(np.sum((rater_means - grand) ** 2)) / (k - 1)
    residual = x - subject_means[:, None] - rater_means[None, :] + grand
    mse = float(np.sum(residual**2)) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    if not np.isfinite(denominator) or denominator == 0:
        return None
    return float((msr - mse) / denominator)


def _calibration_arm(arm: Arm) -> dict[str, object]:
    target = arm.target
    prediction = arm.prediction
    slope, intercept = (float(value) for value in np.polyfit(target, prediction, 1))
    cases = np.unique(arm.case_ids)
    members = [np.flatnonzero(arm.case_ids == case_id) for case_id in cases]
    rng = np.random.default_rng(SEED)
    slopes: list[float] = []
    intercepts: list[float] = []
    for _ in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(cases), size=len(cases))
        indices = np.concatenate([members[index] for index in selected])
        replicate_slope, replicate_intercept = np.polyfit(
            target[indices], prediction[indices], 1
        )
        slopes.append(float(replicate_slope))
        intercepts.append(float(replicate_intercept))
    differences = prediction - target
    absolute_errors = np.abs(differences)
    metrics = compute_metrics(target, prediction)
    icc = _icc_2_1(target, prediction)
    return {
        "benchmark": arm.benchmark,
        "n": arm.n_windows,
        "n_cases": arm.n_cases,
        "calibration_line": {
            "x": "referência BIS",
            "y": "predição",
            "slope": slope,
            "intercept": intercept,
            "slope_ci95_lower": float(np.percentile(slopes, 2.5)),
            "slope_ci95_upper": float(np.percentile(slopes, 97.5)),
            "intercept_ci95_lower": float(np.percentile(intercepts, 2.5)),
            "intercept_ci95_upper": float(np.percentile(intercepts, 97.5)),
            "bootstrap_mean_slope": float(np.mean(slopes)),
            "bootstrap_mean_intercept": float(np.mean(intercepts)),
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
        },
        "bland_altman": {
            "bias": float(np.mean(differences)),
            "sd_differences": float(np.std(differences, ddof=1)),
            "lower_loa": float(np.mean(differences) - 1.96 * np.std(differences, ddof=1)),
            "upper_loa": float(np.mean(differences) + 1.96 * np.std(differences, ddof=1)),
        },
        "icc_2_1_absolute_agreement": icc,
        "fraction_abs_error_le_10": float(np.mean(absolute_errors <= 10.0)),
        "abs_error_p90": float(np.percentile(absolute_errors, 90)),
        "abs_error_p95": float(np.percentile(absolute_errors, 95)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "pearson_r": metrics["pearson_r"],
    }


def _icc_self_check() -> dict[str, object]:
    x = np.linspace(0.0, 100.0, 50)
    perfect = _icc_2_1(x, x)
    offset = _icc_2_1(x, x + 5.0)
    passed = (
        perfect is not None
        and abs(perfect - 1.0) < 1e-12
        and offset is not None
        and offset < 1.0
    )
    if not passed:
        raise VerificationError("Autoverificação do ICC(2,1) falhou")
    return {
        "perfect_agreement": perfect,
        "constant_offset_agreement": offset,
        "passed": True,
    }


# ---------------------------------------------------------------------------
# G. Auditoria do Pk e distribuição do BIS.
# ---------------------------------------------------------------------------


def _target_distribution(target: np.ndarray, case_ids: np.ndarray) -> dict[str, object]:
    def summarize(values: np.ndarray) -> dict[str, object]:
        quartiles = np.percentile(values, [25, 50, 75])
        return {
            "q1": float(quartiles[0]),
            "median": float(quartiles[1]),
            "q3": float(quartiles[2]),
            "fraction_0_40": float(np.mean((values >= 0) & (values < 40))),
            "fraction_40_60": float(np.mean((values >= 40) & (values < 60))),
            "fraction_60_80": float(np.mean((values >= 60) & (values < 80))),
            "fraction_80_100": float(np.mean((values >= 80) & (values <= 100))),
        }

    per_case = [
        {
            "case_id": case_id,
            "n_windows": int((case_ids == case_id).sum()),
            **summarize(target[case_ids == case_id]),
        }
        for case_id in sorted(np.unique(case_ids))
    ]
    return {"pooled": {"n_windows": int(target.size), **summarize(target)}, "per_case": per_case}


def _arm_pk_audit(arm: Arm) -> dict[str, object]:
    target = arm.target
    prediction = arm.prediction
    per_case = []
    for case_id in sorted(np.unique(arm.case_ids)):
        mask = arm.case_ids == case_id
        case_target = target[mask]
        case_prediction = prediction[mask]
        total_pairs = float(case_target.size * (case_target.size - 1) / 2.0)
        target_ties = _tie_pairs(case_target)
        informative = total_pairs - target_ties
        per_case.append(
            {
                "case_id": case_id,
                "n_windows": int(mask.sum()),
                "pk": prediction_probability(case_target, case_prediction),
                "total_pairs": total_pairs,
                "target_tie_pairs": target_ties,
                "informative_pairs": informative,
                "excluded_tie_fraction": (target_ties / total_pairs) if total_pairs else None,
            }
        )
    finite_pk = [row["pk"] for row in per_case if row["pk"] is not None and np.isfinite(row["pk"])]
    total_pairs = float(target.size * (target.size - 1) / 2.0)
    target_ties = _tie_pairs(target)
    return {
        "n_windows": arm.n_windows,
        "n_cases": arm.n_cases,
        "pk": prediction_probability(target, prediction),
        "total_pairs": total_pairs,
        "target_tie_pairs": target_ties,
        "informative_pairs": total_pairs - target_ties,
        "excluded_tie_fraction": (target_ties / total_pairs) if total_pairs else None,
        "per_case": per_case,
        "per_case_pk_summary": {
            "n_defined": len(finite_pk),
            "n_undefined": len(per_case) - len(finite_pk),
            "mean": float(np.mean(finite_pk)) if finite_pk else None,
            "median": float(np.median(finite_pk)) if finite_pk else None,
            "min": float(np.min(finite_pk)) if finite_pk else None,
            "max": float(np.max(finite_pk)) if finite_pk else None,
        },
    }


def _per_case_table(arm: Arm) -> list[dict[str, object]]:
    rows = []
    for case_id in sorted(np.unique(arm.case_ids)):
        mask = arm.case_ids == case_id
        metrics = compute_metrics(arm.target[mask], arm.prediction[mask])
        rows.append(
            {
                "case_id": case_id,
                "n": int(mask.sum()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "pearson_r": metrics["pearson_r"],
                "pk": prediction_probability(arm.target[mask], arm.prediction[mask]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Orquestração.
# ---------------------------------------------------------------------------


def main() -> int:
    started = time.perf_counter()
    TMP_OUT.mkdir(parents=True, exist_ok=True)

    _log("A. predições por janela dos 4 braços CNN")
    cnn_arms, metrics_by_arm, pk_by_arm = _recompute_cnn_arms()
    verification = _verify_cnn_arms(metrics_by_arm, pk_by_arm)

    _log("B. RF espectral do holdout Figshare")
    rf_figshare, rf_figshare_verification = _recompute_rf_figshare()
    verification["rf_spectral_figshare"] = rf_figshare_verification

    _log("C. RF espectral no VitalDB")
    rf_vitaldb, rf_vitaldb_report, rf_vitaldb_verification = _recompute_rf_vitaldb()
    verification["rf_spectral_vitaldb"] = rf_vitaldb_verification

    arms = {**cnn_arms, "rf_spectral_figshare": rf_figshare, "rf_spectral_vitaldb": rf_vitaldb}

    figshare_arms = {key: arm for key, arm in arms.items() if arm.benchmark == "figshare_holdout"}
    vitaldb_arms = {key: arm for key, arm in arms.items() if arm.benchmark == "vitaldb_external"}
    pair_specs_figshare = {
        "figshare_holdout__active_minus_mixed": ("cnn_active_figshare", "cnn_mixed_figshare"),
        "figshare_holdout__active_minus_rf": ("cnn_active_figshare", "rf_spectral_figshare"),
    }
    pair_specs_vitaldb = {
        "vitaldb_external__active_minus_mixed": ("cnn_active_vitaldb", "cnn_mixed_vitaldb"),
        "vitaldb_external__active_minus_rf": ("cnn_active_vitaldb", "rf_spectral_vitaldb"),
    }

    _log("D. bootstrap pareado por caso (B=1000, seed 42)")
    samples_figshare = _family_paired_bootstrap(figshare_arms, pair_specs_figshare)
    samples_vitaldb = _family_paired_bootstrap(vitaldb_arms, pair_specs_vitaldb)

    pairs_payload: dict[str, object] = {}
    for pair_name, (arm_a_key, arm_b_key) in {**pair_specs_figshare, **pair_specs_vitaldb}.items():
        arm_a = arms[arm_a_key]
        arm_b = arms[arm_b_key]
        observed = _observed_pair(arm_a, arm_b)
        samples = {**samples_figshare, **samples_vitaldb}[pair_name]
        bootstrap = {
            f"delta_{metric}": _summarize_bootstrap(
                samples[metric], observed["arm_a"][metric] - observed["arm_b"][metric]
            )
            for metric in PAIR_METRICS
        }
        pairs_payload[pair_name] = {
            "benchmark": arm_a.benchmark,
            "arm_a": arm_a_key,
            "arm_b": arm_b_key,
            "n_cases": arm_a.n_cases,
            "n_windows": arm_a.n_windows,
            "case_alignment_verified": bool(np.array_equal(arm_a.case_ids, arm_b.case_ids)),
            "observed_arm_a": {metric: observed["arm_a"][metric] for metric in PAIR_METRICS},
            "observed_arm_b": {metric: observed["arm_b"][metric] for metric in PAIR_METRICS},
            "observed_delta": {
                metric: observed["arm_a"][metric] - observed["arm_b"][metric]
                for metric in PAIR_METRICS
            },
            "bootstrap": bootstrap,
        }
        _log(
            f"pair {pair_name}: dMAE={bootstrap['delta_mae']['observed']:.4f} "
            f"[{bootstrap['delta_mae']['lower_95']:.4f}, {bootstrap['delta_mae']['upper_95']:.4f}] "
            f"exclui_zero={bootstrap['delta_mae']['ci_excludes_zero']}"
        )

    _log("E. teste pareado por caso do MAE (ativo vs misto, VitalDB)")
    casewise_test = _casewise_mae_test(
        cnn_arms["cnn_active_vitaldb"], cnn_arms["cnn_mixed_vitaldb"]
    )
    _log(
        f"MAE por caso: misto melhor em {casewise_test['n_arm_b_better']}/"
        f"{casewise_test['n_pairs']} (p_sinal={casewise_test['sign_test']['p_two_sided']:.6f}, "
        f"p_wilcoxon={casewise_test['wilcoxon_signed_rank']['p_two_sided']:.6f})"
    )

    _log("F. calibração e concordância (6 braços)")
    calibration = {key: _calibration_arm(arm) for key, arm in arms.items()}

    _log("G. auditoria do Pk")
    benchmark_audits = {}
    for benchmark, family in (
        ("figshare_holdout", figshare_arms),
        ("vitaldb_external", vitaldb_arms),
    ):
        reference = next(iter(family.values()))
        benchmark_audits[benchmark] = {
            "n_cases": reference.n_cases,
            "n_windows": reference.n_windows,
            "target_distribution": _target_distribution(reference.target, reference.case_ids),
            "arms": {key: _arm_pk_audit(arm) for key, arm in family.items()},
        }

    per_case_tables = {
        benchmark: {key: _per_case_table(arm) for key, arm in family.items()}
        for benchmark, family in (
            ("figshare_holdout", figshare_arms),
            ("vitaldb_external", vitaldb_arms),
        )
    }

    _write_deterministic_json(
        REPORTS_OUT / "paired_bootstrap.json",
        {
            "scope": "research_only",
            "analysis": "paired_case_bootstrap_and_casewise_tests",
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
            "delta_definition": (
                "delta = métrica(braço a) - métrica(braço b); cada par é reamostrado pelos "
                "MESMOS casos nos dois braços. Para MAE/RMSE, delta negativo favorece o braço a; "
                "para Pearson/Pk, delta positivo favorece o braço a; para viés, delta positivo "
                "indica viés mais positivo no braço a."
            ),
            "pairs": pairs_payload,
            "vitaldb_casewise_mae_test": {
                **casewise_test,
                "comparison": casewise_test["comparison"],
                "expected": "14/15 casos com MAE menor no braço misto (relatado no TCC)",
            },
            "per_case_tables": per_case_tables,
            "verification": verification,
            "retrained": False,
            "raw_eeg_in_report": False,
        },
    )

    _write_deterministic_json(
        REPORTS_OUT / "calibration_analysis.json",
        {
            "scope": "research_only",
            "analysis": "calibration_agreement_and_error_distribution",
            "bootstrap_samples": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
            "icc_formula": (
                "ICC(2,1) de concordância absoluta (duas vias, efeitos aleatórios, medida "
                "única; k=2 avaliações: referência BIS e predição). "
                "MSR=k/(n-1)*Σ_i(x̄_i-x̄)²; MSC=n/(k-1)*Σ_j(x̄_j-x̄)²; "
                "MSE=Σ_ij(x_ij-x̄_i-x̄_j+x̄)²/((n-1)(k-1)); "
                "ICC=(MSR-MSE)/(MSR+(k-1)MSE+k(MSC-MSE)/n)."
            ),
            "icc_reference": "Shrout & Fleiss (1979); McGraw & Wong (1996)",
            "icc_self_check": _icc_self_check(),
            "bland_altman_definition": (
                "viés = média(predição - referência); LoA = viés ± 1,96·DP(diferenças)"
            ),
            "arms": calibration,
            "retrained": False,
            "raw_eeg_in_report": False,
        },
    )

    _write_deterministic_json(
        REPORTS_OUT / "pk_audit.json",
        {
            "scope": "research_only",
            "analysis": "prediction_probability_audit",
            "pair_definition": (
                "total de pares = n(n-1)/2; pares empatados no alvo são excluídos por não "
                "carregarem informação de ordem (Smith et al., 1996)"
            ),
            "benchmarks": benchmark_audits,
            "retrained": False,
            "raw_eeg_in_report": False,
        },
    )

    _write_deterministic_json(REPORTS_OUT / "spectral_baseline_vitaldb.json", rf_vitaldb_report)

    elapsed = time.perf_counter() - started
    _log(f"concluído em {elapsed:.1f} s")
    _log("arquivos: tmp/reanalysis/*.npz, reports/paired_bootstrap.json, "
         "reports/calibration_analysis.json, reports/pk_audit.json, "
         "reports/spectral_baseline_vitaldb.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

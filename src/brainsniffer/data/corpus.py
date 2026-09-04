"""Auditable assembly of the supervised EEG-to-BIS corpus.

The project has two useful public sources, but they do not have the same role
by default: Figshare is the development corpus and the current VitalDB files
are a frozen external set.  This module makes a mixed training pool explicit,
audits cases before window construction, and records enough provenance to
reproduce the decision later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig
from .mat_reader import EEGCase, load_case
from .preprocess import bis_stage, signal_quality

CORPUS_SCHEMA_VERSION = 1
CORPUS_NAME = "brainsniffer-mixed-eeg-bis-v1"
FIGSHARE_SOURCE = "Figshare · EEG and BIS raw data"
VITALDB_SOURCE = "VitalDB Open Dataset"


@dataclass(frozen=True)
class CorpusQualityConfig:
    """Case-level gates used before a file can enter supervised training."""

    min_finite_fraction: float = 0.90
    max_gap_seconds: float = 2.0
    min_global_quality: float = 0.35
    min_bis_valid_fraction: float = 0.80
    min_window_quality: float = DEFAULT_MIN_SIGNAL_QUALITY
    stride_seconds: float | None = None

    def validate(self) -> None:
        if not 0 <= self.min_finite_fraction <= 1:
            raise ValueError("min_finite_fraction deve estar entre 0 e 1")
        if self.max_gap_seconds < 0:
            raise ValueError("max_gap_seconds deve ser não negativo")
        if not 0 <= self.min_global_quality <= 1:
            raise ValueError("min_global_quality deve estar entre 0 e 1")
        if not 0 <= self.min_bis_valid_fraction <= 1:
            raise ValueError("min_bis_valid_fraction deve estar entre 0 e 1")
        if not 0 <= self.min_window_quality <= 1:
            raise ValueError("min_window_quality deve estar entre 0 e 1")
        if self.stride_seconds is not None and self.stride_seconds <= 0:
            raise ValueError("stride_seconds deve ser positivo")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(case: EEGCase, path: Path) -> str:
    source = (case.source_dataset or "").lower()
    if "vital" in source or path.name.lower().startswith("vitaldb_"):
        return "vitaldb"
    return "figshare"


def _source_name(case: EEGCase, path: Path) -> str:
    if _source_key(case, path) == "vitaldb":
        return VITALDB_SOURCE
    return FIGSHARE_SOURCE


def _group_id(case: EEGCase, source_key: str) -> str:
    if case.group_id:
        return str(case.group_id)
    if case.subject_id:
        return f"{source_key}:subject:{case.subject_id}"
    return f"{source_key}:case:{case.case_id}"


def _max_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run without materializing samples."""

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.size or not mask.any():
        return 0
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    if not values.size:
        return None
    return float(np.quantile(values, quantile))


def _signal_summary(case: EEGCase, config: PreprocessConfig) -> dict[str, object]:
    signal = np.asarray(case.eeg, dtype=np.float64).reshape(-1)
    finite = np.isfinite(signal)
    finite_values = signal[finite]
    finite_pairs = finite[:-1] & finite[1:] if signal.size > 1 else np.empty(0, dtype=bool)
    differences = np.abs(np.diff(signal)) if signal.size > 1 else np.empty(0, dtype=float)
    differences = differences[finite_pairs] if differences.size else differences
    saturation = (
        np.abs(finite_values) >= config.clip_uv * 0.98
        if finite_values.size
        else np.empty(0, dtype=bool)
    )
    return {
        "sample_count": int(signal.size),
        "nonfinite_count": int((~finite).sum()),
        "finite_fraction": float(finite.mean()) if signal.size else 0.0,
        "max_nonfinite_gap_samples": _max_true_run(~finite),
        "max_nonfinite_gap_seconds": float(_max_true_run(~finite) / case.sampling_rate),
        "raw_min": float(finite_values.min()) if finite_values.size else None,
        "raw_max": float(finite_values.max()) if finite_values.size else None,
        "raw_p01": _quantile(finite_values, 0.01),
        "raw_median": _quantile(finite_values, 0.50),
        "raw_p99": _quantile(finite_values, 0.99),
        "abs_p99": _quantile(np.abs(finite_values), 0.99),
        "abs_diff_p99": _quantile(differences, 0.99),
        "saturation_fraction": float(saturation.mean()) if saturation.size else 0.0,
        "flatline_fraction": (
            float((differences < 1e-5).mean()) if differences.size else 0.0
        ),
        "quality_global": signal_quality(signal, config),
        "imputation_applied_offline": bool((~finite).any()),
    }


def _bis_summary(bis: np.ndarray) -> dict[str, object]:
    values = np.asarray(bis, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    valid = finite & (values >= 0) & (values <= 100)
    valid_values = values[valid]
    return {
        "sample_count": int(values.size),
        "nonfinite_count": int((~finite).sum()),
        "finite_fraction": float(finite.mean()) if values.size else 0.0,
        "valid_fraction": float(valid.mean()) if values.size else 0.0,
        "valid_count": int(valid.sum()),
        "raw_min": float(valid_values.min()) if valid_values.size else None,
        "raw_max": float(valid_values.max()) if valid_values.size else None,
        "stage_counts": {
            stage: int(sum(bis_stage(float(value)) == stage for value in valid_values))
            for stage in ("deep", "general", "light", "awake")
        },
    }


def _window_summary(
    case: EEGCase,
    preprocess: PreprocessConfig,
    quality: CorpusQualityConfig,
) -> dict[str, object]:
    stride_seconds = quality.stride_seconds or preprocess.window_seconds
    stride_samples = int(round(stride_seconds * case.sampling_rate))
    window_samples = preprocess.window_samples
    if stride_samples <= 0:
        raise ValueError("stride_seconds deve resultar em pelo menos uma amostra")
    starts = range(0, max(case.eeg.size - window_samples + 1, 0), stride_samples)
    candidate_count = 0
    label_valid_count = 0
    accepted_count = 0
    scores: list[float] = []
    for start in starts:
        candidate_count += 1
        label_time = start / case.sampling_rate + preprocess.label_offset_seconds
        label_index = int(round(label_time / case.label_interval_seconds))
        if label_index < 0 or label_index >= case.bis.size:
            continue
        if bis_stage(float(case.bis[label_index])) == "invalid":
            continue
        label_valid_count += 1
        score = signal_quality(case.eeg[start : start + window_samples], preprocess)
        scores.append(score)
        if score >= quality.min_window_quality:
            accepted_count += 1
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "stride_seconds": float(stride_seconds),
        "window_seconds": float(preprocess.window_seconds),
        "candidate_windows": candidate_count,
        "label_valid_windows": label_valid_count,
        "accepted_windows": accepted_count,
        "accepted_fraction": (
            float(accepted_count / label_valid_count) if label_valid_count else 0.0
        ),
        "quality_min": _quantile(score_array, 0.0),
        "quality_p10": _quantile(score_array, 0.10),
        "quality_median": _quantile(score_array, 0.50),
        "quality_mean": float(score_array.mean()) if score_array.size else None,
        "quality_p90": _quantile(score_array, 0.90),
        "quality_max": _quantile(score_array, 1.0),
    }


def audit_corpus_case(
    path: str | Path,
    *,
    role: str,
    preprocess_config: PreprocessConfig | None = None,
    quality_config: CorpusQualityConfig | None = None,
) -> dict[str, object]:
    """Audit one local case and return a privacy-preserving manifest record."""

    path = Path(path)
    preprocess = preprocess_config or PreprocessConfig()
    quality = quality_config or CorpusQualityConfig()
    quality.validate()
    record: dict[str, object] = {
        "path": str(path),
        "file_name": path.name,
        "role": role,
        "exists": path.is_file(),
        "eligible_for_training": False,
        "quality_status": "exclude",
        "exclusion_reasons": [],
    }
    if not path.is_file():
        record["exclusion_reasons"] = ["file_missing"]
        return record

    try:
        case = load_case(path, sampling_rate=preprocess.sampling_rate)
        source_key = _source_key(case, path)
        record.update(
            {
                "case_id": case.case_id,
                "group_id": _group_id(case, source_key),
                "subject_id": case.subject_id,
                "source_key": source_key,
                "source_dataset": _source_name(case, path),
                "sampling_rate": int(case.sampling_rate),
                "label_interval_seconds": float(case.label_interval_seconds),
                "eeg_unit": case.eeg_unit,
                "eeg_track_name": case.eeg_track_name,
                "bis_track_name": case.bis_track_name,
                "duration_seconds": float(case.duration_seconds),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "signal": _signal_summary(case, preprocess),
                "bis": _bis_summary(case.bis),
                "windows": _window_summary(case, preprocess, quality),
            }
        )
    except Exception as error:  # keep the corpus report useful when one file is bad
        record["error"] = str(error)
        record["exclusion_reasons"] = ["load_error"]
        return record

    signal = record["signal"]
    bis = record["bis"]
    windows = record["windows"]
    reasons: list[str] = []
    if float(signal["finite_fraction"]) < quality.min_finite_fraction:
        reasons.append("finite_fraction_below_gate")
    if float(signal["max_nonfinite_gap_seconds"]) > quality.max_gap_seconds:
        reasons.append("nonfinite_gap_above_gate")
    if float(signal["quality_global"]) < quality.min_global_quality:
        reasons.append("global_quality_below_gate")
    if float(bis["valid_fraction"]) < quality.min_bis_valid_fraction:
        reasons.append("bis_valid_fraction_below_gate")
    if int(windows["accepted_windows"]) == 0:
        reasons.append("no_accepted_windows")

    record["exclusion_reasons"] = reasons
    record["quality_status"] = "include" if not reasons else "quarantine"
    record["eligible_for_training"] = role == "development_pool" and not reasons
    return record


def _paths(directory: str | Path, pattern: str) -> list[Path]:
    return sorted(Path(directory).glob(pattern)) if Path(directory).is_dir() else []


def build_corpus_manifest(
    *,
    figshare_dir: str | Path = "data/raw",
    vitaldb_train_dir: str | Path = "data/vitaldb_train",
    vitaldb_external_dir: str | Path = "data/vitaldb",
    preprocess_config: PreprocessConfig | None = None,
    quality_config: CorpusQualityConfig | None = None,
) -> dict[str, object]:
    """Build the mixed-corpus manifest while keeping the external set frozen."""

    preprocess = preprocess_config or PreprocessConfig()
    quality = quality_config or CorpusQualityConfig()
    quality.validate()
    development_paths = [
        (path, "development_pool") for path in _paths(figshare_dir, "case*.mat")
    ] + [
        (path, "development_pool")
        for path in _paths(vitaldb_train_dir, "vitaldb_case*.npz")
    ]
    external_paths = [
        (path, "frozen_external")
        for path in _paths(vitaldb_external_dir, "vitaldb_case*.npz")
    ]
    records = [
        audit_corpus_case(
            path,
            role=role,
            preprocess_config=preprocess,
            quality_config=quality,
        )
        for path, role in (*development_paths, *external_paths)
    ]

    frozen_groups = {
        str(record["group_id"])
        for record in records
        if record.get("role") == "frozen_external" and record.get("group_id")
    }
    for record in records:
        if record.get("role") != "development_pool":
            continue
        if record.get("group_id") not in frozen_groups:
            continue
        reasons = list(record.get("exclusion_reasons", []))
        reasons.append("group_overlaps_frozen_external")
        record["exclusion_reasons"] = reasons
        record["quality_status"] = "quarantine"
        record["eligible_for_training"] = False

    development = [record for record in records if record.get("role") == "development_pool"]
    eligible = [record for record in development if record.get("eligible_for_training")]
    source_summary: dict[str, dict[str, object]] = {}
    for record in records:
        source = str(record.get("source_key", "unknown"))
        summary = source_summary.setdefault(
            source,
            {
                "cases": 0,
                "eligible_cases": 0,
                "quarantined_cases": 0,
                "frozen_external_cases": 0,
                "accepted_windows": 0,
                "eligible_training_windows": 0,
                "frozen_external_windows": 0,
                "duration_seconds": 0.0,
            },
        )
        summary["cases"] += 1
        summary["eligible_cases"] += int(bool(record.get("eligible_for_training")))
        summary["quarantined_cases"] += int(record.get("quality_status") == "quarantine")
        summary["frozen_external_cases"] += int(record.get("role") == "frozen_external")
        windows = record.get("windows", {})
        if isinstance(windows, dict):
            summary["accepted_windows"] += int(windows.get("accepted_windows", 0))
            if record.get("eligible_for_training"):
                summary["eligible_training_windows"] += int(
                    windows.get("accepted_windows", 0)
                )
            if record.get("role") == "frozen_external":
                summary["frozen_external_windows"] += int(
                    windows.get("accepted_windows", 0)
                )
        summary["duration_seconds"] += float(record.get("duration_seconds", 0.0) or 0.0)

    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_name": CORPUS_NAME,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "research_only",
        "task": "supervised EEG to BIS regression",
        "split_unit": "subject_id when available, otherwise source-prefixed case_id",
        "training_policy": {
            "case_balanced_sampling": True,
            "source_balanced_sampling": True,
            "offline_imputation": "short gaps must be audited; long gaps are quarantined",
            "frozen_external_is_not_training": True,
            "raw_eeg_in_report": False,
        },
        "third_source": {
            "name": "DOSE-I",
            "status": "not_merged",
            "reason": (
                "MOAA/S and state-of-consciousness labels are not a continuous BIS reference; "
                "use a separate multi-task or external-validation protocol"
            ),
            "url": "https://zenodo.org/records/18483292",
        },
        "preprocess_config": asdict(preprocess),
        "quality_config": asdict(quality),
        "directories": {
            "figshare": str(Path(figshare_dir)),
            "vitaldb_train": str(Path(vitaldb_train_dir)),
            "vitaldb_external": str(Path(vitaldb_external_dir)),
        },
        "summary": {
            "total_files": len(records),
            "development_cases": len(development),
            "eligible_training_cases": len(eligible),
            "quarantined_development_cases": sum(
                record.get("quality_status") == "quarantine" for record in development
            ),
            "frozen_external_cases": sum(
                record.get("role") == "frozen_external" for record in records
            ),
            "eligible_training_windows": sum(
                int(record.get("windows", {}).get("accepted_windows", 0))
                for record in eligible
            ),
            "source_summary": source_summary,
        },
        "cases": records,
        "eligible_cases": [record for record in eligible],
        "frozen_external_cases": [
            record for record in records if record.get("role") == "frozen_external"
        ],
    }


def write_corpus_manifest(manifest: dict[str, object], path: str | Path) -> Path:
    """Write a JSON manifest atomically and return its destination."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(target)
    return target


def load_corpus_manifest(path: str | Path) -> dict[str, object]:
    """Load and minimally validate a corpus manifest before training."""

    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"não foi possível ler o manifesto do corpus: {target}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"manifesto do corpus não é JSON válido: {target}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("manifesto do corpus ausente ou incompatível")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("manifesto do corpus não contém a lista de casos")
    return payload


def corpus_paths(
    manifest: dict[str, object],
    *,
    include_quarantined: bool = False,
) -> list[Path]:
    """Return only development files allowed by the manifest."""

    paths: list[Path] = []
    for record in manifest.get("cases", []):
        if not isinstance(record, dict) or record.get("role") != "development_pool":
            continue
        if not record.get("eligible_for_training") and not (
            include_quarantined and record.get("quality_status") == "quarantine"
        ):
            continue
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo do corpus ausente: {path}")
        paths.append(path)
    if not paths:
        raise ValueError("o manifesto não tem casos de desenvolvimento elegíveis")
    return paths

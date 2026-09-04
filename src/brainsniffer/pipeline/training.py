"""Reproducible case-level training and checkpointing."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig, TrainingConfig, resolve_device
from ..data.preprocess import WindowedEEG, bis_stage
from ..data.split import CaseSplit, split_case_ids
from ..models.cnn import Conv1DDepthEstimator
from .metrics import compute_metrics


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path | None
    split: CaseSplit
    history: list[dict[str, float]]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    device: str
    dataset_summary: dict[str, object]
    quality_threshold: float
    input_files: list[dict[str, object]]


def summarize_windows(windows: WindowedEEG) -> dict[str, object]:
    """Create a JSON-serializable audit summary for a training run."""

    stage_counts: dict[str, int] = {}
    for value in windows.bis:
        stage = bis_stage(float(value))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    quality = windows.quality.astype(np.float64)
    group_ids = (
        windows.group_ids.astype(str)
        if windows.group_ids is not None
        else windows.case_ids.astype(str)
    )
    source_datasets = (
        windows.source_datasets.astype(str)
        if windows.source_datasets is not None
        else np.full(windows.case_ids.shape, "unknown", dtype=str)
    )
    return {
        "n_windows": int(windows.signals.shape[0]),
        "n_cases": int(np.unique(windows.case_ids.astype(str)).size),
        "n_groups": int(np.unique(group_ids).size),
        "window_shape": list(windows.signals.shape[1:]),
        "stage_counts": stage_counts,
        "source_counts": {
            source: int((source_datasets == source).sum())
            for source in sorted(np.unique(source_datasets).tolist())
        },
        "quality": {
            "min": float(quality.min()) if quality.size else None,
            "mean": float(quality.mean()) if quality.size else None,
            "median": float(np.median(quality)) if quality.size else None,
            "max": float(quality.max()) if quality.size else None,
        },
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, str | None]:
    """Capture the software environment needed to interpret a checkpoint."""

    return {
        "project": _installed_version("brainsniffer") or "0.1.0",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": _installed_version("scipy"),
        "scikit_learn": _installed_version("scikit-learn"),
    }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of an artifact without loading it all at once."""

    if chunk_size < 1:
        raise ValueError("chunk_size deve ser positivo")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    """Record path, size, and SHA-256 for the files used by an experiment."""

    manifest: list[dict[str, object]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def verify_file_manifest(manifest: object) -> None:
    """Fail closed when a recorded input file is missing or has changed."""

    if manifest in (None, []):
        return
    if not isinstance(manifest, list):
        raise ValueError("O manifesto de arquivos de entrada é inválido")
    for entry in manifest:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("O manifesto de arquivos de entrada é inválido")
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise ValueError(f"Arquivo de entrada do manifesto ausente: {path}")
        actual = sha256_file(path)
        if actual != str(entry["sha256"]):
            raise ValueError(
                f"SHA-256 do arquivo de entrada não coincide: {path} "
                f"(esperado {entry['sha256']}, obtido {actual})"
            )


def _mask_for_cases(case_ids: np.ndarray, cases: tuple[str, ...]) -> np.ndarray:
    return np.isin(case_ids.astype(str), np.asarray(cases, dtype=str))


@torch.inference_mode()
def predict_model(
    model: nn.Module,
    signals: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for start in range(0, signals.shape[0], batch_size):
        batch = torch.from_numpy(signals[start : start + batch_size]).float().to(device)
        values.append(model(batch).detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _loader(
    signals: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    config: TrainingConfig,
    shuffle: bool,
    *,
    group_ids: np.ndarray | None = None,
    source_ids: np.ndarray | None = None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(signals[mask]).float(),
        torch.from_numpy(labels[mask]).float(),
    )
    sampler = None
    if config.balance_groups:
        if group_ids is None:
            raise ValueError("balance_groups requer group_ids")
        selected_groups = group_ids[mask].astype(str)
        unique, counts = np.unique(selected_groups, return_counts=True)
        weights_by_group = {
            group: 1.0 / float(count) for group, count in zip(unique, counts, strict=False)
        }
        if config.balance_sources:
            if source_ids is None:
                raise ValueError("balance_sources requer source_ids")
            selected_sources = source_ids[mask].astype(str)
            group_source = {
                group: source
                for group, source in zip(selected_groups, selected_sources, strict=False)
            }
            source_group_counts = {
                source: sum(group_source.get(group) == source for group in unique)
                for source in np.unique(selected_sources)
            }
            weights = torch.as_tensor(
                [
                    weights_by_group[group]
                    / max(source_group_counts[group_source[group]], 1)
                    for group in selected_groups
                ],
                dtype=torch.double,
            )
        else:
            weights = torch.as_tensor(
                [weights_by_group[group] for group in selected_groups],
                dtype=torch.double,
            )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(selected_groups),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=False,
    )


def train_model(
    windows: WindowedEEG,
    *,
    preprocess_config: PreprocessConfig | None = None,
    training_config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
    min_quality: float = DEFAULT_MIN_SIGNAL_QUALITY,
    input_files: Sequence[str | Path] | None = None,
    corpus_manifest_path: str | Path | None = None,
) -> TrainingResult:
    """Train the baseline CNN and evaluate only on unseen surgical cases."""

    preprocess_config = preprocess_config or PreprocessConfig()
    training_config = training_config or TrainingConfig()
    if not 0 <= min_quality <= 1:
        raise ValueError("min_quality deve estar entre 0 e 1")
    if windows.signals.shape[0] == 0:
        raise ValueError("Nenhuma janela válida disponível para treinamento")
    if windows.quality.size and float(windows.quality.min()) + 1e-6 < min_quality:
        raise ValueError(
            "As janelas contêm sinais abaixo de min_quality; filtre-as antes do treinamento"
        )
    dataset_summary = summarize_windows(windows)
    input_file_manifest = build_file_manifest(input_files) if input_files is not None else []
    environment = runtime_metadata()
    set_seed(training_config.seed)
    device = resolve_device(training_config.device)
    split_ids = (
        windows.group_ids.astype(str)
        if windows.group_ids is not None
        else windows.case_ids.astype(str)
    )
    split_unit = "group" if windows.group_ids is not None else "case"
    split = split_case_ids(
        split_ids,
        validation_fraction=training_config.validation_fraction,
        test_fraction=training_config.test_fraction,
        seed=training_config.seed,
    )
    train_mask = _mask_for_cases(split_ids, split.train_cases)
    validation_mask = _mask_for_cases(split_ids, split.validation_cases)
    test_mask = _mask_for_cases(split_ids, split.test_cases)

    model = Conv1DDepthEstimator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    train_loader = _loader(
        windows.signals,
        windows.bis,
        train_mask,
        training_config,
        shuffle=True,
        group_ids=split_ids,
        source_ids=(
            windows.source_datasets.astype(str)
            if windows.source_datasets is not None
            else None
        ),
    )
    dataset_summary["split_unit"] = split_unit
    dataset_summary["balanced_groups"] = bool(training_config.balance_groups)
    dataset_summary["balanced_sources"] = bool(training_config.balance_sources)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mae = float("inf")

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.to(device))
            loss = criterion(prediction, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_prediction = predict_model(
            model, windows.signals[validation_mask], device=device
        )
        validation_metrics = compute_metrics(windows.bis[validation_mask], validation_prediction)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_mae": validation_metrics.get("mae", float("nan")),
            "validation_rmse": validation_metrics.get("rmse", float("nan")),
        }
        history.append(row)
        if row["validation_mae"] < best_validation_mae:
            best_validation_mae = row["validation_mae"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    validation_prediction = predict_model(model, windows.signals[validation_mask], device=device)
    test_prediction = predict_model(model, windows.signals[test_mask], device=device)
    validation_metrics = compute_metrics(windows.bis[validation_mask], validation_prediction)
    test_metrics = compute_metrics(windows.bis[test_mask], test_prediction)

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "model_name": "Conv1DDepthEstimator",
            "preprocess_config": asdict(preprocess_config),
            "training_config": asdict(training_config),
            "split": asdict(split),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "dataset_summary": dataset_summary,
            "min_quality": min_quality,
            "history": history,
            "environment": environment,
            "input_files": input_file_manifest,
            "corpus_manifest": (
                {
                    "path": str(corpus_manifest_path),
                    "sha256": sha256_file(corpus_manifest_path),
                }
                if corpus_manifest_path is not None
                else None
            ),
            "split_unit": split_unit,
        }
        torch.save(checkpoint_payload, checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        checkpoint.with_suffix(".json").write_text(
            json.dumps(
                {
                    "model_name": "Conv1DDepthEstimator",
                    "preprocess_config": asdict(preprocess_config),
                    "training_config": asdict(training_config),
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "split": asdict(split),
                    "dataset_summary": dataset_summary,
                    "min_quality": min_quality,
                    "history": history,
                    "environment": environment,
                    "input_files": input_file_manifest,
                    "corpus_manifest": (
                        {
                            "path": str(corpus_manifest_path),
                            "sha256": sha256_file(corpus_manifest_path),
                        }
                        if corpus_manifest_path is not None
                        else None
                    ),
                    "split_unit": split_unit,
                    "checkpoint_sha256": checkpoint_sha256,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return TrainingResult(
        checkpoint_path=checkpoint,
        split=split,
        history=history,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        device=device,
        dataset_summary=dataset_summary,
        quality_threshold=min_quality,
        input_files=input_file_manifest,
    )


def load_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[nn.Module, PreprocessConfig, dict]:
    """Load a saved CNN and its preprocessing configuration."""

    path = Path(path)
    metadata_path = path.with_suffix(".json")
    sidecar: dict[str, object] = {}
    if metadata_path.exists():
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha256 = sidecar.get("checkpoint_sha256")
        if expected_sha256:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "SHA-256 do checkpoint não coincide com o manifesto: "
                    f"esperado {expected_sha256}, obtido {actual_sha256}"
                )
    payload = torch.load(path, map_location=device, weights_only=False)
    for key in ("checkpoint_sha256", "input_files"):
        if key in sidecar:
            payload[key] = sidecar[key]
    model = Conv1DDepthEstimator().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    preprocess = PreprocessConfig(**payload.get("preprocess_config", {}))
    return model, preprocess, payload

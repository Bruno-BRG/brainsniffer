"""Transparent spectral baseline for comparison with the CNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestRegressor

from ..config import TrainingConfig
from ..data.preprocess import WindowedEEG
from ..data.split import CaseSplit, group_kfold_case_ids, split_case_ids
from .metrics import compute_metrics

BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class BaselineResult:
    split: CaseSplit
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class CrossValidationResult:
    n_splits: int
    folds: tuple[dict[str, float], ...]
    mean: dict[str, float]
    std: dict[str, float]


def spectral_features(
    signals: np.ndarray, sampling_rate: int = 128
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract interpretable band-power, edge, entropy, and amplitude features."""

    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 3:
        signals = signals[:, 0, :]
    if signals.ndim != 2:
        raise ValueError("signals deve ter forma (janelas, amostras) ou (janelas, 1, amostras)")
    feature_names = [name for name, _, _ in BANDS]
    feature_names += [f"relative_{name}" for name, _, _ in BANDS]
    feature_names += ["spectral_edge_90hz", "spectral_entropy", "rms", "line_length"]
    rows: list[list[float]] = []
    for signal in signals:
        frequencies, power = welch(
            signal,
            fs=sampling_rate,
            nperseg=min(256, signal.size),
            noverlap=0,
        )
        total_power = float(np.trapezoid(power, frequencies)) + 1e-12
        absolute: list[float] = []
        for _, low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            absolute.append(
                float(np.trapezoid(power[mask], frequencies[mask])) if mask.any() else 0.0
            )
        probability = power / (float(power.sum()) + 1e-12)
        entropy = float(
            -(probability * np.log(probability + 1e-12)).sum() / np.log(max(power.size, 2))
        )
        cumulative = np.cumsum(power)
        edge_index = int(np.searchsorted(cumulative, cumulative[-1] * 0.9))
        edge_hz = float(frequencies[min(edge_index, frequencies.size - 1)])
        row = absolute + [value / total_power for value in absolute]
        row += [
            edge_hz,
            entropy,
            float(np.sqrt(np.mean(signal**2))),
            float(np.mean(np.abs(np.diff(signal)))),
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), tuple(feature_names)


def train_spectral_baseline(
    windows: WindowedEEG,
    *,
    sampling_rate: int = 128,
    training_config: TrainingConfig | None = None,
) -> BaselineResult:
    """Fit a small random forest only on training cases."""

    training_config = training_config or TrainingConfig()
    features, feature_names = spectral_features(windows.signals, sampling_rate)
    split = split_case_ids(
        windows.case_ids,
        validation_fraction=training_config.validation_fraction,
        test_fraction=training_config.test_fraction,
        seed=training_config.seed,
    )
    case_ids = windows.case_ids.astype(str)
    train_mask = np.isin(case_ids, split.train_cases)
    validation_mask = np.isin(case_ids, split.validation_cases)
    test_mask = np.isin(case_ids, split.test_cases)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=training_config.seed,
        n_jobs=-1,
    )
    model.fit(features[train_mask], windows.bis[train_mask])
    validation_prediction = np.clip(model.predict(features[validation_mask]), 0.0, 100.0)
    test_prediction = np.clip(model.predict(features[test_mask]), 0.0, 100.0)
    return BaselineResult(
        split=split,
        validation_metrics=compute_metrics(windows.bis[validation_mask], validation_prediction),
        test_metrics=compute_metrics(windows.bis[test_mask], test_prediction),
        feature_names=feature_names,
    )


def cross_validate_spectral_baseline(
    windows: WindowedEEG,
    *,
    sampling_rate: int = 128,
    n_splits: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """Evaluate the spectral baseline with patient-level test folds."""

    features, _ = spectral_features(windows.signals, sampling_rate)
    case_ids = windows.case_ids.astype(str)
    folds = group_kfold_case_ids(case_ids, n_splits=n_splits, seed=seed)
    fold_metrics: list[dict[str, float]] = []
    for fold in folds:
        train_mask = np.isin(case_ids, fold.train_cases)
        test_mask = np.isin(case_ids, fold.test_cases)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            random_state=seed + fold.fold_index,
            n_jobs=-1,
        )
        model.fit(features[train_mask], windows.bis[train_mask])
        prediction = np.clip(model.predict(features[test_mask]), 0.0, 100.0)
        metrics = compute_metrics(windows.bis[test_mask], prediction)
        fold_metrics.append({"fold": float(fold.fold_index), **metrics})
    metric_names = ["mae", "rmse", "bias", "pearson_r", "stage_accuracy", "stage_macro_f1"]
    mean = {name: float(np.nanmean([row[name] for row in fold_metrics])) for name in metric_names}
    std = {name: float(np.nanstd([row[name] for row in fold_metrics])) for name in metric_names}
    return CrossValidationResult(
        n_splits=n_splits,
        folds=tuple(fold_metrics),
        mean=mean,
        std=std,
    )

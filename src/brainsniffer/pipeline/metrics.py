"""Metrics for continuous BIS-reference prediction and staged reporting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..data.preprocess import bis_stage


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute metrics while excluding non-finite pairs."""

    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.size != prediction.size:
        raise ValueError(
            "target e prediction devem ter o mesmo número de elementos "
            f"(obtido {target.size} e {prediction.size})"
        )
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    if target.size == 0:
        return {"n": 0.0}
    target_stage = [bis_stage(value) for value in target]
    prediction_stage = [bis_stage(value) for value in prediction]
    return {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "bias": float(np.mean(prediction - target)),
        "pearson_r": _correlation(target, prediction),
        "stage_accuracy": float(accuracy_score(target_stage, prediction_stage)),
        "stage_macro_f1": float(
            f1_score(target_stage, prediction_stage, average="macro", zero_division=0)
        ),
    }


def bootstrap_case_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    case_ids: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate case-cluster bootstrap intervals for continuous/stage metrics.

    Resampling whole cases, rather than individual windows, preserves the main
    dependence structure of a longitudinal surgical recording. The result is an
    exploratory uncertainty summary, not a clinical confidence statement.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap deve ser positivo")
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    case_ids = np.asarray(case_ids).reshape(-1)
    if target.size != prediction.size or target.size != case_ids.size:
        raise ValueError("target, prediction e case_ids devem ter o mesmo tamanho")
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    case_ids = case_ids[valid]
    unique_cases = np.unique(case_ids)
    if unique_cases.size < 2:
        raise ValueError("bootstrap por caso requer pelo menos dois casos")
    members = [np.flatnonzero(case_ids == case_id) for case_id in unique_cases]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in selected])
        rows.append(compute_metrics(target[indices], prediction[indices]))

    metric_names = [
        "mae",
        "rmse",
        "bias",
        "pearson_r",
        "stage_accuracy",
        "stage_macro_f1",
    ]
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray([row.get(name, np.nan) for row in rows], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            continue
        summary[name] = {
            "mean": float(np.mean(finite_values)),
            "lower_95": float(np.percentile(finite_values, 2.5)),
            "upper_95": float(np.percentile(finite_values, 97.5)),
        }
    return summary

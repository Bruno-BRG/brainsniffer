"""Metrics for continuous BIS-reference prediction and staged reporting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..data.preprocess import bis_stage

# Match bis_stage's valid research taxonomy; never infer labels from a sample.
STAGE_LABELS = ("deep", "general", "light", "awake")

# Zone stratification for BIS-reference reporting.
# Research bands follow bis_stage: deep [0,40), general [40,60),
# light [60,80), awake [80,100]. Clinical shorthand groups them as
# acordado (awake), transicional/sedacao leve (light),
# geral adequada (general) e profunda (deep). Isoelectric risk is
# reported as an exploratory BIS<20 subset of deep (near burst
# suppression/flat EEG), not a fifth mutually exclusive class.
ZONE_LABELS = ("deep", "general", "light", "awake")
ZONE_RANGES = {
    "deep": (0.0, 40.0),
    "general": (40.0, 60.0),
    "light": (60.0, 80.0),
    "awake": (80.0, 100.0),
}
ZONE_CLINICAL_PT = {
    "deep": "profunda (0-40)",
    "general": "geral adequada (40-60)",
    "light": "leve/transicional (60-80)",
    "awake": "acordado (80-100)",
}
ISOELECTRIC_THRESHOLD = 20.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def confusion_matrix_by_stage(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    """Count true-vs-predicted research bands in fixed STAGE_LABELS order."""
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    target_stage = [bis_stage(v) for v in target]
    prediction_stage = [bis_stage(v) for v in prediction]
    index = {label: i for i, label in enumerate(STAGE_LABELS)}
    counts = [[0 for _ in STAGE_LABELS] for _ in STAGE_LABELS]
    for t, p in zip(target_stage, prediction_stage):
        if t in index and p in index:
            counts[index[t]][index[p]] += 1
    row_totals = [sum(row) for row in counts]
    recall: dict[str, float | None] = {}
    for i, label in enumerate(STAGE_LABELS):
        total = row_totals[i]
        recall[label] = (counts[i][i] / total) if total else None
    return {"order": list(STAGE_LABELS), "counts": counts, "recall": recall}


def compute_zone_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, object]:
    """Stratify continuous metrics by true BIS research band.

    Pearson inside a narrow band is unstable (restricted range) and is
    reported for completeness only; MAE/RMSE/bias and recall carry the
    zone reading. Isoelectric BIS<20 is an exploratory subset of deep.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    total = int(target.size)
    target_stage = np.asarray([bis_stage(v) for v in target])
    zones: dict[str, object] = {}
    for label in ZONE_LABELS:
        mask = target_stage == label
        n = int(mask.sum())
        if n == 0:
            zones[label] = {
                "n": 0,
                "share": 0.0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "pearson_r": None,
                "recall": None,
            }
            continue
        t = target[mask]
        p = prediction[mask]
        pred_stage = np.asarray([bis_stage(v) for v in p])
        zones[label] = {
            "n": n,
            "share": (n / total) if total else 0.0,
            "mae": float(np.mean(np.abs(t - p))),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "bias": float(np.mean(p - t)),
            "pearson_r": _correlation(t, p),
            "recall": float((pred_stage == label).mean()),
        }
    deep_mask = target < ISOELECTRIC_THRESHOLD
    iso_n = int(deep_mask.sum())
    if iso_n:
        t = target[deep_mask]
        p = prediction[deep_mask]
        isoelectric = {
            "n": iso_n,
            "share": (iso_n / total) if total else 0.0,
            "mae": float(np.mean(np.abs(t - p))),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "bias": float(np.mean(p - t)),
            "pearson_r": _correlation(t, p),
        }
    else:
        isoelectric = {"n": 0, "share": 0.0, "mae": None, "rmse": None, "bias": None, "pearson_r": None}
    return {
        "zones": zones,
        "isoelectric_subset_lt20": isoelectric,
        "confusion": confusion_matrix_by_stage(target, prediction),
        "n": total,
    }


def compute_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute metrics while excluding non-finite pairs.

    Stage macro-F1 averages the four fixed BIS bands, assigning zero to absent
    bands. This differs from historical reports using sample-inferred labels,
    including bootstrap replicates missing bands. Finite out-of-range values
    retain bis_stage's ``invalid`` sentinel (not a fifth averaged class); no
    clipping or additional pair exclusion is applied here.
    """

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
            f1_score(
                target_stage,
                prediction_stage,
                labels=STAGE_LABELS,
                average="macro",
                zero_division=0,
            )
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
    Metrics remain pooled over windows, not equally weighted case metrics.
    Stage macro-F1 uses the same four fixed bands in every replicate. ``mean``
    is the bootstrap mean, not the observed point estimate. Callers must supply
    the largest known independent grouping (e.g. subject for repeat cases).
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

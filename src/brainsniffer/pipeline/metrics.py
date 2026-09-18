"""Metrics for continuous BIS-reference prediction and staged reporting."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score

from ..data.preprocess import bis_stage

# Match bis_stage's valid research taxonomy; never infer labels from a sample.
STAGE_LABELS = ("deep", "general", "light", "awake")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])




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


def _tie_pairs(values: np.ndarray) -> float:
    """Count pairs tied on one scale, for the Kendall tau-b correction."""

    _, counts = np.unique(values, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.sum(counts * (counts - 1) / 2.0))


def prediction_probability(target: np.ndarray, prediction: np.ndarray) -> float:
    """Prediction probability Pk (Smith, Dutton and Smith, Anesthesiology 1996).

    Pk is a rescaled variant of Kim ordinal association, that is, Somers d with
    the observed scale as the dependent variable, between an indicator, here
    the CNN estimate, and an observed depth scale, here the reference BIS. It
    is 1 when the indicator ranks observed depth perfectly and 0.5 when it is
    no better than chance. Pairs tied on the observed scale are excluded
    because they carry no ordering information, and pairs tied on the
    indicator count as 0.5, per the original definition. The measure is
    nonparametric: it depends only on the ranking of both scales, so it is
    invariant to affine rescaling of the indicator.

    The concordant-minus-discordant count is recovered from Kendall tau-b,
    whose denominator is the tie-corrected pair count, keeping the statistic
    exact and O(n log n) even when the reference holds thousands of distinct
    interpolated values.

    Smith et al. define observed depth from response-to-stimulus testing. Here
    the observed scale is the reference BIS of the monitor, so Pk quantifies
    rank agreement with that index rather than with a stimulus-response
    endpoint.
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
    size = int(target.size)
    if size < 2:
        return float("nan")
    total_pairs = size * (size - 1) / 2.0
    informative = total_pairs - _tie_pairs(target)
    if informative <= 0:
        return float("nan")
    comparable = total_pairs - _tie_pairs(prediction)
    if comparable <= 0:
        # Every informative pair is tied on the indicator: Pk is chance level.
        return 0.5
    tau = float(kendalltau(prediction, target).statistic)
    if not np.isfinite(tau):
        return float("nan")
    difference = tau * float(np.sqrt(comparable * informative))
    return float(0.5 + difference / (2.0 * informative))


def bootstrap_case_prediction_probability(
    target: np.ndarray,
    prediction: np.ndarray,
    case_ids: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Case-cluster bootstrap interval for Pk (exploratory, not clinical)."""

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
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in selected])
        values.append(prediction_probability(target[indices], prediction[indices]))
    finite = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    if not finite.size:
        return {}
    return {
        "mean": float(np.mean(finite)),
        "lower_95": float(np.percentile(finite, 2.5)),
        "upper_95": float(np.percentile(finite, 97.5)),
    }

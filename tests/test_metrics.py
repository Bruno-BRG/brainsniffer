import numpy as np
import pytest

from brainsniffer.pipeline.metrics import bootstrap_case_metrics, compute_metrics


def test_compute_metrics_rejects_misaligned_vectors():
    with pytest.raises(ValueError, match="mesmo número de elementos"):
        compute_metrics(np.asarray([40.0, 50.0]), np.asarray([40.0]))


def test_compute_metrics_ignores_nonfinite_pairs_without_broadcasting():
    result = compute_metrics(
        np.asarray([40.0, np.nan, 60.0]),
        np.asarray([42.0, 55.0, np.inf]),
    )
    assert result["n"] == 1.0
    assert result["mae"] == pytest.approx(2.0)


def test_bootstrap_case_metrics_resamples_groups():
    target = np.asarray([40.0, 42.0, 60.0, 62.0])
    prediction = np.asarray([41.0, 41.0, 59.0, 65.0])
    cases = np.asarray(["case1", "case1", "case2", "case2"])

    result = bootstrap_case_metrics(target, prediction, cases, n_bootstrap=50, seed=7)

    assert set(result) == {
        "mae",
        "rmse",
        "bias",
        "pearson_r",
        "stage_accuracy",
        "stage_macro_f1",
    }
    assert result["mae"]["lower_95"] <= result["mae"]["mean"]
    assert result["mae"]["mean"] <= result["mae"]["upper_95"]


def test_bootstrap_case_metrics_requires_multiple_cases():
    with pytest.raises(ValueError, match="dois casos"):
        bootstrap_case_metrics(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0]),
            np.asarray(["case1", "case1"]),
            n_bootstrap=10,
        )


def test_bootstrap_case_metrics_is_reproducible_for_same_seed():
    target = np.asarray([35.0, 42.0, 58.0, 71.0, 84.0, 91.0])
    prediction = np.asarray([37.0, 40.0, 61.0, 68.0, 80.0, 94.0])
    cases = np.asarray(["case1", "case1", "case2", "case2", "case3", "case3"])

    first = bootstrap_case_metrics(target, prediction, cases, n_bootstrap=100, seed=42)
    second = bootstrap_case_metrics(target, prediction, cases, n_bootstrap=100, seed=42)

    assert first == second

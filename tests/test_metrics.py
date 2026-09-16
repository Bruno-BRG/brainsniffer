import numpy as np
import pytest

from brainsniffer.pipeline.metrics import bootstrap_case_metrics, compute_metrics


@pytest.mark.parametrize(
    ("values", "expected"),
    [([40.0], 0.25), ([20.0, 40.0], 0.5), ([20.0, 40.0, 60.0], 0.75),
     ([20.0, 40.0, 60.0, 80.0], 1.0)],
)
def test_macro_f1_uses_four_bands_even_when_absent(values, expected):
    result = compute_metrics(np.asarray(values), np.asarray(values))
    assert result["stage_macro_f1"] == pytest.approx(expected)
    assert result["stage_accuracy"] == 1.0
    assert result["mae"] == 0.0


def test_macro_f1_counts_prediction_only_band_and_false_negatives():
    result = compute_metrics(np.asarray([20.0, 20.0]), np.asarray([20.0, 40.0]))
    assert result["stage_macro_f1"] == pytest.approx((2 / 3) / 4)
    assert result["stage_accuracy"] == 0.5


def test_macro_f1_respects_bis_stage_boundaries():
    result = compute_metrics(
        np.asarray([0, 39.999, 40, 59.999, 60, 79.999, 80, 100]),
        np.asarray([39, 0, 59, 40, 79, 60, 100, 80]),
    )
    assert result["stage_macro_f1"] == 1.0
    crossed = compute_metrics(np.asarray([39.999, 59.999, 79.999]), np.asarray([40, 60, 80]))
    assert crossed["stage_macro_f1"] == 0.0


def test_finite_invalid_values_are_not_a_fifth_macro_class_or_clipped():
    result = compute_metrics(np.asarray([-1, 101, 20, 40]), np.asarray([-1, 101, -1, 40]))
    assert result["n"] == 4.0
    assert result["mae"] == pytest.approx(21 / 4)
    assert result["stage_accuracy"] == 0.75
    assert result["stage_macro_f1"] == 0.25


@pytest.mark.parametrize("target,prediction", [([], []), ([np.nan], [40]), ([40], [np.inf])])
def test_compute_metrics_empty_finite_support(target, prediction):
    assert compute_metrics(np.asarray(target), np.asarray(prediction)) == {"n": 0.0}


def test_constant_target_correlation_is_undefined():
    result = compute_metrics(np.asarray([40, 40]), np.asarray([40, 41]))
    assert np.isnan(result["pearson_r"])


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


def test_bootstrap_fixed_bands_when_replicates_omit_classes():
    # Each cluster is a perfect prediction in a different band. Duplicating
    # either cluster must give .25, not the historical inferred-label 1.0.
    seed, count = 17, 40
    draws = np.random.default_rng(seed).integers(0, 2, size=(count, 2))
    expected = np.asarray([len(set(row)) / 4 for row in draws])
    result = bootstrap_case_metrics(
        np.asarray([20, 80]), np.asarray([20, 80]), np.asarray(["a", "b"]),
        n_bootstrap=count, seed=seed,
    )
    assert result["stage_macro_f1"] == pytest.approx({
        "mean": expected.mean(),
        "lower_95": np.percentile(expected, 2.5),
        "upper_95": np.percentile(expected, 97.5),
    })


def test_bootstrap_constant_band_and_nonfinite_pair_filtering():
    result = bootstrap_case_metrics(
        np.asarray([40, 40, np.nan]), np.asarray([40, 40, 80]),
        np.asarray(["a", "b", "c"]), n_bootstrap=10,
    )
    assert result["stage_macro_f1"] == {"mean": 0.25, "lower_95": 0.25, "upper_95": 0.25}
    assert "pearson_r" not in result
    with pytest.raises(ValueError, match="dois casos"):
        bootstrap_case_metrics(np.asarray([40, np.nan]), np.asarray([40, 80]),
                               np.asarray(["a", "b"]), n_bootstrap=10)


@pytest.mark.parametrize("count", [0, -1])
def test_bootstrap_rejects_nonpositive_count(count):
    with pytest.raises(ValueError, match="positivo"):
        bootstrap_case_metrics(np.asarray([20, 80]), np.asarray([20, 80]),
                               np.asarray(["a", "b"]), n_bootstrap=count)


def test_bootstrap_rejects_misaligned_groups():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        bootstrap_case_metrics(np.asarray([20, 80]), np.asarray([20, 80]), np.asarray(["a"]))


def test_bootstrap_case_metrics_is_reproducible_for_same_seed():
    target = np.asarray([35.0, 42.0, 58.0, 71.0, 84.0, 91.0])
    prediction = np.asarray([37.0, 40.0, 61.0, 68.0, 80.0, 94.0])
    cases = np.asarray(["case1", "case1", "case2", "case2", "case3", "case3"])

    first = bootstrap_case_metrics(target, prediction, cases, n_bootstrap=100, seed=42)
    second = bootstrap_case_metrics(target, prediction, cases, n_bootstrap=100, seed=42)

    assert first == second

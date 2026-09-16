import numpy as np

from brainsniffer.data.preprocess import WindowedEEG, subset_windows
from brainsniffer.data.split import group_kfold_case_ids, split_case_ids


def test_case_split_has_no_overlap_and_is_reproducible():
    case_ids = np.repeat(["case1", "case2", "case3", "case4", "case5"], 3)
    first = split_case_ids(case_ids, seed=7)
    second = split_case_ids(case_ids, seed=7)
    assert first == second
    assert set(first.train_cases).isdisjoint(first.validation_cases)
    assert set(first.train_cases).isdisjoint(first.test_cases)
    assert set(first.validation_cases).isdisjoint(first.test_cases)
    assert set(first.train_cases + first.validation_cases + first.test_cases) == set(case_ids)


def test_subset_windows_retains_all_cases_for_smoke_training():
    case_ids = np.repeat(["case1", "case2", "case3", "case4"], 5)
    windows = WindowedEEG(
        signals=np.zeros((case_ids.size, 1, 8), dtype=np.float32),
        bis=np.full(case_ids.size, 50, dtype=np.float32),
        case_ids=case_ids,
        start_seconds=np.arange(case_ids.size, dtype=np.float32),
        quality=np.ones(case_ids.size, dtype=np.float32),
    )
    limited = subset_windows(windows, 8, seed=11)
    assert limited.signals.shape[0] == 8
    assert set(limited.case_ids) == {"case1", "case2", "case3", "case4"}


def test_group_kfold_keeps_cases_whole_and_covers_all_cases():
    case_ids = np.repeat(["case1", "case2", "case3", "case4", "case5"], 2)
    folds = group_kfold_case_ids(case_ids, n_splits=5, seed=3)
    assert len(folds) == 5
    assert set().union(*(set(fold.test_cases) for fold in folds)) == set(case_ids)
    for fold in folds:
        assert set(fold.train_cases).isdisjoint(fold.test_cases)

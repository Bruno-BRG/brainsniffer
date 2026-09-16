"""Patient/case-level split utilities to prevent temporal leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CaseSplit:
    train_cases: tuple[str, ...]
    validation_cases: tuple[str, ...]
    test_cases: tuple[str, ...]


@dataclass(frozen=True)
class GroupFold:
    fold_index: int
    train_cases: tuple[str, ...]
    test_cases: tuple[str, ...]


def split_case_ids(
    case_ids: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> CaseSplit:
    """Split unique case IDs; all windows from one case stay together."""

    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("As frações de validação/teste devem ser >=0 e somar menos que 1")
    unique = np.asarray(sorted({str(item) for item in case_ids}))
    if unique.size < 3:
        raise ValueError("São necessários pelo menos 3 casos para treino, validação e teste")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(unique.size * test_fraction)))
    n_validation = max(1, int(round(unique.size * validation_fraction)))
    while n_test + n_validation >= unique.size:
        if n_validation > 1:
            n_validation -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Não foi possível reservar treino, validação e teste")
    test = tuple(sorted(shuffled[:n_test].tolist()))
    validation = tuple(sorted(shuffled[n_test : n_test + n_validation].tolist()))
    train = tuple(sorted(shuffled[n_test + n_validation :].tolist()))
    return CaseSplit(train_cases=train, validation_cases=validation, test_cases=test)


def group_kfold_case_ids(
    case_ids: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[GroupFold, ...]:
    """Create deterministic patient-level folds without splitting a case."""

    unique = np.asarray(sorted({str(item) for item in case_ids}))
    if n_splits < 2 or n_splits > unique.size:
        raise ValueError("n_splits deve estar entre 2 e o número de casos")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for index, case_id in enumerate(shuffled.tolist()):
        buckets[index % n_splits].append(case_id)
    folds: list[GroupFold] = []
    for fold_index, test_bucket in enumerate(buckets):
        test_cases = tuple(sorted(test_bucket))
        train_cases = tuple(sorted(set(unique.tolist()) - set(test_cases)))
        folds.append(
            GroupFold(
                fold_index=fold_index,
                train_cases=train_cases,
                test_cases=test_cases,
            )
        )
    return tuple(folds)

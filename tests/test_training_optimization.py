import numpy as np
import pytest
import torch
from torch import nn

import brainsniffer.pipeline.training as training_module
from brainsniffer.config import PreprocessConfig, TrainingConfig
from brainsniffer.data.preprocess import WindowedEEG


class TinyEstimator(nn.Module):
    """Fast deterministic stand-in for the CNN in pipeline tests."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return torch.full((inputs.shape[0],), 50.0, device=inputs.device) + self.bias


def _windows() -> WindowedEEG:
    return WindowedEEG(
        signals=np.zeros((6, 1, 16), dtype=np.float32),
        bis=np.asarray([20, 25, 40, 45, 70, 75], dtype=np.float32),
        case_ids=np.asarray(["case-a", "case-a", "case-b", "case-b", "case-c", "case-c"]),
        start_seconds=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32),
        quality=np.ones(6, dtype=np.float32),
    )


def test_training_records_validation_metrics_and_is_reproducible(monkeypatch):
    monkeypatch.setattr(training_module, "Conv1DDepthEstimator", TinyEstimator)
    config = TrainingConfig(
        epochs=3,
        batch_size=2,
        early_stopping_patience=None,
        gradient_clip_norm=None,
        mixed_precision=True,
    )

    first = training_module.train_model(
        _windows(),
        preprocess_config=PreprocessConfig(window_seconds=16 / 128),
        training_config=config,
    )
    second = training_module.train_model(
        _windows(),
        preprocess_config=PreprocessConfig(window_seconds=16 / 128),
        training_config=config,
    )

    for first_row, second_row in zip(first.history, second.history, strict=True):
        assert first_row.keys() == second_row.keys()
        assert all(
            np.isclose(first_row[key], second_row[key], equal_nan=True)
            for key in first_row
        )
    assert len(first.history) == 3
    assert {"validation_bias", "validation_pearson_r", "validation_stage_macro_f1"} <= set(
        first.history[0]
    )
    assert first.dataset_summary["split_sizes"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert first.dataset_summary["mixed_precision"] is False
    assert first.dataset_summary["deterministic"] is True


def test_training_rejects_unfiltered_low_quality_windows(monkeypatch):
    monkeypatch.setattr(training_module, "Conv1DDepthEstimator", TinyEstimator)
    windows = _windows()
    windows = WindowedEEG(
        signals=windows.signals,
        bis=windows.bis,
        case_ids=windows.case_ids,
        start_seconds=windows.start_seconds,
        quality=np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="min_quality"):
        training_module.train_model(
            windows,
            training_config=TrainingConfig(epochs=1),
            min_quality=0.5,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"early_stopping_patience": 0},
        {"gradient_clip_norm": 0},
        {"scheduler_factor": 1},
    ],
)
def test_training_rejects_invalid_optimization_config(monkeypatch, kwargs):
    monkeypatch.setattr(training_module, "Conv1DDepthEstimator", TinyEstimator)
    with pytest.raises(ValueError):
        training_module.train_model(_windows(), training_config=TrainingConfig(**kwargs))

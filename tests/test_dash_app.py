import numpy as np
import pytest
import torch

from brainsniffer.config import PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase
from dash_app import _fast_replay_case


class _ConstantModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0],), 55.0, device=inputs.device)


def test_dashboard_replay_interpolates_recorded_missing_samples_without_hiding_quality():
    config = PreprocessConfig()
    eeg = np.sin(np.linspace(0, 30, config.window_samples + 128)).astype(np.float32)
    eeg[20] = np.nan
    case = EEGCase(case_id="vitaldb_case_test", eeg=eeg, bis=np.full(6, 55.0, dtype=np.float32))

    predictions = _fast_replay_case(_ConstantModel(), case, config, min_quality=0.2)

    assert predictions
    assert all(item.raw_bis == pytest.approx(55.0) for item in predictions)
    assert predictions[0].quality < 1.0
    assert all(np.isfinite(item.quality) for item in predictions)


def test_dashboard_replay_rejects_a_recording_with_no_finite_eeg_sample():
    config = PreprocessConfig()
    eeg = np.full(config.window_samples, np.nan, dtype=np.float32)
    case = EEGCase(case_id="vitaldb_case_invalid", eeg=eeg, bis=np.full(6, 55.0, dtype=np.float32))

    with pytest.raises(ValueError, match="não há amostras EEG finitas"):
        _fast_replay_case(_ConstantModel(), case, config)

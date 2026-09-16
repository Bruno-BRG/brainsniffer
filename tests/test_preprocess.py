import numpy as np

from brainsniffer.config import PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase
from brainsniffer.data.preprocess import (
    StreamingPreprocessor,
    bis_stage,
    data_handling_policy,
    make_windows,
    preprocess_window,
    signal_diagnostics,
    signal_quality,
)


def test_bis_stage_boundaries_are_bounded():
    assert bis_stage(-1) == "invalid"
    assert bis_stage(39.9) == "deep"
    assert bis_stage(40) == "general"
    assert bis_stage(59.9) == "general"
    assert bis_stage(60) == "light"
    assert bis_stage(80) == "awake"
    assert bis_stage(101) == "invalid"


def test_preprocess_returns_finite_scaled_signal():
    config = PreprocessConfig()
    time = np.arange(config.window_samples) / config.sampling_rate
    raw = 20 * np.sin(2 * np.pi * 10 * time)
    raw[10] = np.nan
    processed = preprocess_window(raw, config)
    assert processed.shape == (config.window_samples,)
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 5
    assert signal_quality(raw, config) > 0


def test_signal_diagnostics_exposes_nonfinite_count_without_raw_signal():
    diagnostics = signal_diagnostics(np.asarray([1.0, np.nan, -2.0], dtype=np.float32))

    assert diagnostics["sample_count"] == 3
    assert diagnostics["nonfinite_count"] == 1
    assert diagnostics["finite_fraction"] == 2 / 3
    assert diagnostics["raw_min"] == -2.0
    assert diagnostics["raw_max"] == 1.0
    assert diagnostics["imputation_applied_offline"] is True


def test_data_handling_policy_separates_offline_and_online_paths():
    assert data_handling_policy() == {
        "mode": "offline_evaluation",
        "nonfinite_samples": "linear_interpolation_during_window_construction",
        "online_nonfinite_policy": "reject_before_filter_or_resampler",
        "raw_eeg_in_report": False,
    }


def test_streaming_preprocessor_keeps_causal_state_between_chunks():
    config = PreprocessConfig()
    processor = StreamingPreprocessor(config)
    first = processor.process(np.ones(128, dtype=np.float32))
    second = processor.process(np.ones(128, dtype=np.float32))
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()
    assert not np.array_equal(first, second)


def test_window_builder_can_use_one_causal_filter_state_per_case():
    config = PreprocessConfig()
    case = EEGCase(
        case_id="case-test",
        eeg=np.sin(np.arange(config.window_samples * 2, dtype=np.float32)),
        bis=np.array([50, 55], dtype=np.float32),
    )
    windows = make_windows(case, config)
    assert windows.signals.shape == (2, 1, config.window_samples)
    assert np.isfinite(windows.signals).all()


def test_window_builder_applies_explicit_label_offset():
    config = PreprocessConfig(label_offset_seconds=5.0)
    case = EEGCase(
        case_id="case-offset",
        eeg=np.sin(np.arange(config.window_samples * 3, dtype=np.float32)),
        bis=np.array([40, 50, 60, 70], dtype=np.float32),
    )

    windows = make_windows(case, config)

    assert windows.bis.tolist() == [50.0, 60.0, 70.0]


def test_window_builder_rejects_nonfinite_label_offset():
    config = PreprocessConfig(label_offset_seconds=float("nan"))
    case = EEGCase(
        case_id="case-offset-invalid",
        eeg=np.zeros(config.window_samples, dtype=np.float32),
        bis=np.array([50], dtype=np.float32),
    )

    with np.testing.assert_raises_regex(ValueError, "label_offset_seconds"):
        make_windows(case, config)


def test_default_quality_gate_rejects_saturated_offset_signal():
    config = PreprocessConfig()
    case = EEGCase(
        case_id="case-corrupt",
        eeg=np.full(config.window_samples, 2048.0, dtype=np.float32),
        bis=np.array([50], dtype=np.float32),
    )
    windows = make_windows(case, config)
    assert windows.signals.shape[0] == 0

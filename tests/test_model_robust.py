import pytest
import torch
from torch import nn

from brainsniffer.models.cnn import RobustConv1DDepthEstimator, parameter_count


def test_robust_cnn_is_bounded_and_uses_batch_independent_normalization():
    model = RobustConv1DDepthEstimator()

    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.modules())
    output = model(torch.zeros(3, 1, 640))

    assert output.shape == (3,)
    assert torch.all((output >= 0) & (output <= 100))
    assert parameter_count(model) < 250_000


def test_robust_cnn_supports_singleton_batches_and_shorter_windows():
    model = RobustConv1DDepthEstimator().eval()

    with torch.inference_mode():
        single = model(torch.randn(1, 1, 64))
        repeated = model(torch.randn(2, 1, 64))

    assert single.shape == (1,)
    assert repeated.shape == (2,)
    assert torch.isfinite(single).all()
    assert torch.isfinite(repeated).all()


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.zeros(1, 640), "formato"),
        (torch.zeros(1, 2, 640), "canal"),
        (torch.zeros(1, 1, 7), "ao menos"),
        (torch.tensor([[[float("nan")]]]).expand(1, 1, 8), "finitos"),
    ],
)
def test_robust_cnn_rejects_invalid_inputs(inputs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        RobustConv1DDepthEstimator()(inputs)


def test_mc_dropout_returns_bounded_mean_and_uncertainty_and_restores_state():
    model = RobustConv1DDepthEstimator(dropout=0.25)
    model.train()
    inputs = torch.randn(2, 1, 640)

    mean, std = model.predict_with_uncertainty(inputs, samples=5)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert torch.all((mean >= 0) & (mean <= 100))
    assert torch.all(std >= 0)
    assert model.training
    assert all(module.training for module in model.modules())


def test_mc_dropout_rejects_nonpositive_sample_count():
    with pytest.raises(ValueError, match="positivo"):
        RobustConv1DDepthEstimator().predict_with_uncertainty(torch.zeros(1, 1, 64), samples=0)

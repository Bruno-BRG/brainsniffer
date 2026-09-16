"""Compact 1-D CNNs for BIS regression.

``Conv1DDepthEstimator`` is kept as the checkpoint-compatible baseline.  The
``RobustConv1DDepthEstimator`` below is an opt-in architecture for new
experiments: it avoids batch-statistics at inference time and uses residual,
depthwise-separable blocks to increase receptive field without a large
parameter or latency budget.
"""

from __future__ import annotations

import torch
from torch import nn


def _validate_input(x: torch.Tensor, input_channels: int, *, min_length: int = 8) -> None:
    """Fail closed with an actionable error before calling ``Conv1d``."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x deve ser um torch.Tensor")
    if x.ndim != 3:
        raise ValueError("x deve ter formato (batch, canais, amostras)")
    if x.shape[1] != input_channels:
        raise ValueError(
            f"x deve ter {input_channels} canal(is); recebido {x.shape[1]}"
        )
    if x.shape[-1] < min_length:
        raise ValueError(f"x deve conter ao menos {min_length} amostras")
    if not torch.isfinite(x).all():
        raise ValueError("x deve conter somente valores finitos")


def _group_count(channels: int) -> int:
    """Choose a GroupNorm partition with at least one channel per group."""

    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class Conv1DDepthEstimator(nn.Module):
    """Predict a bounded 0--100 BIS-like reference from one EEG channel."""

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.features[0].in_channels)
        features = self.features(x)
        raw = self.regressor(features).squeeze(-1)
        return 100.0 * torch.sigmoid(raw)


class _DepthwiseResidualBlock(nn.Module):
    """Residual block with inexpensive temporal mixing and GroupNorm.

    GroupNorm computes statistics per example, so a batch of one (the normal
    case for a live stream) behaves like a larger evaluation batch.  Dilation
    expands temporal context while the depthwise convolutions keep the block
    small enough for CPU replay.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride deve ser 1 ou 2")
        padding = dilation * 2
        self.temporal = nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size=5,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=input_channels,
            bias=False,
        )
        self.temporal_norm = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.project = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.project_norm = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout)
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv1d(
                input_channels, output_channels, kernel_size=1, stride=stride, bias=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.temporal(x)
        out = self.activation(self.temporal_norm(out))
        out = self.dropout(out)
        out = self.project_norm(self.project(out))
        return self.activation(out + residual)


class RobustConv1DDepthEstimator(nn.Module):
    """Residual, batch-size-independent CNN for new BIS experiments.

    The class intentionally does not replace :class:`Conv1DDepthEstimator`:
    existing checkpoints remain reproducible while experiments can compare an
    architecture designed for small batches and domain shift.  Inputs have
    shape ``(batch, input_channels, samples)`` and outputs are bounded to
    ``[0, 100]`` just like the baseline.

    ``predict_with_uncertainty`` enables a lightweight epistemic uncertainty
    estimate using dropout at inference.  It is a research diagnostic, not a
    calibrated clinical confidence interval.
    """

    def __init__(
        self,
        input_channels: int = 1,
        *,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels deve ser positivo")
        if len(channels) < 2 or any(channel < 1 for channel in channels):
            raise ValueError("channels deve conter ao menos dois valores positivos")
        if not 0 <= dropout < 1:
            raise ValueError("dropout deve estar em [0, 1)")
        self.input_channels = input_channels
        self.channels = channels
        self.dropout_rate = dropout
        stem_channels = channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, stem_channels, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        dilations = (1, 1, 2, 4)
        for index, output_channels in enumerate(channels):
            input_block_channels = stem_channels if index == 0 else channels[index - 1]
            blocks.append(
                _DepthwiseResidualBlock(
                    input_block_channels,
                    output_channels,
                    stride=1 if index == 0 else 2,
                    dilation=dilations[min(index, len(dilations) - 1)],
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.pool_mean = nn.AdaptiveAvgPool1d(1)
        self.pool_peak = nn.AdaptiveMaxPool1d(1)
        final_channels = channels[-1]
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * 2, final_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.input_channels)
        features = self.blocks(self.stem(x))
        pooled = torch.cat((self.pool_mean(features), self.pool_peak(features)), dim=1)
        raw = self.regressor(pooled).squeeze(-1)
        return 100.0 * torch.sigmoid(raw)

    @torch.inference_mode()
    def predict_with_uncertainty(
        self, x: torch.Tensor, *, samples: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` from deterministic or MC-dropout inference.

        The model's original train/eval state (including nested modules) is
        restored even if inference raises.  GroupNorm remains in evaluation
        mode while only dropout layers are sampled.
        """

        if samples < 1:
            raise ValueError("samples deve ser positivo")
        _validate_input(x, self.input_channels)
        states = {module: module.training for module in self.modules()}
        try:
            self.eval()
            if samples > 1:
                for module in self.modules():
                    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                        module.train()
            predictions = torch.stack([self(x) for _ in range(samples)], dim=0)
            return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)
        finally:
            for module, training in states.items():
                module.train(training)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

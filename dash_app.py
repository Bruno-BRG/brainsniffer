"""Dash + Plotly research console for BrainSniffer.

The dashboard separates retrospective evidence from the causal replay. Model
inference is performed once per case and cached; while a replay is playing,
the browser advances a small clock and reveals the already-computed causal
outputs locally. This keeps the interaction smooth without changing the
inference semantics of ``RealtimeEstimator``.
"""

# The layout contains intentionally readable long Portuguese labels.
# ruff: noqa: E501

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from brainsniffer.config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase, load_case
from brainsniffer.data.preprocess import StreamingPreprocessor, bis_stage, signal_quality
from brainsniffer.models.cnn import parameter_count
from brainsniffer.pipeline.realtime import RealtimePrediction
from brainsniffer.pipeline.training import load_checkpoint

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("BRAINSNIFFER_DATA_DIR", APP_ROOT / "data/raw"))
VITAL_DIR = Path(os.getenv("BRAINSNIFFER_VITAL_DIR", APP_ROOT / "data/vitaldb"))
MODEL_PATH = Path(os.getenv("BRAINSNIFFER_CHECKPOINT", APP_ROOT / "models/brainsniffer_cnn.pt"))
REPORTS_DIR = APP_ROOT / "reports"
REPLAY_EEG_POINTS = 9000
REPLAY_INTERVAL_MS = 120

COLORS = {
    "navy": "#102A43",
    "ink": "#172B4D",
    "muted": "#627D98",
    "line": "#D9E2EC",
    "surface": "#FFFFFF",
    "canvas": "#F4F7FB",
    "blue": "#247BA0",
    "teal": "#20A39E",
    "orange": "#F18F01",
    "red": "#D1495B",
    "purple": "#6C63A8",
    "green": "#2D936C",
}

STAGE_LABELS = {
    "deep": "Profundo",
    "general": "Anestesia geral",
    "light": "Sedação leve",
    "awake": "Acordado",
    "abstain": "ABSTAIN · sinal insuficiente",
}

METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "bias": "Bias",
    "pearson_r": "Pearson r",
    "stage_accuracy": "Acurácia por estágio",
    "stage_macro_f1": "Macro-F1 por estágio",
}


def _read_report(filename: str) -> dict[str, object]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


HOLDOUT_REPORT = _read_report("figshare_holdout_evaluation.json")
EXTERNAL_REPORT = _read_report("vitaldb_external_validation.json")
OFFSET_REPORT = _read_report("offset_sensitivity.json")
HOLDOUT_METRICS = HOLDOUT_REPORT.get("recomputed_test_metrics", {})
EXTERNAL_METRICS = EXTERNAL_REPORT.get("metrics", {})


def _load_model() -> tuple[object | None, PreprocessConfig | None, dict[str, object], str | None]:
    if not MODEL_PATH.exists():
        return None, None, {}, f"Checkpoint não encontrado: {MODEL_PATH}"
    try:
        model, preprocess, payload = load_checkpoint(MODEL_PATH, device="cpu")
    except Exception as error:  # pragma: no cover - defensive startup guard
        return None, None, {}, f"Falha ao carregar o checkpoint: {error}"
    return model, preprocess, payload, None


MODEL, PREPROCESS, MODEL_METADATA, MODEL_ERROR = _load_model()


@dataclass(frozen=True)
class ReplayPayload:
    case: EEGCase
    prediction_times: np.ndarray
    raw_predictions: np.ndarray
    smoothed_predictions: np.ndarray
    qualities: np.ndarray
    stages: tuple[str, ...]
    eeg_times: np.ndarray
    eeg_values: np.ndarray


def _finite_number(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _json_series(values: np.ndarray | list[float]) -> list[float | None]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) if np.isfinite(value) else None for value in array]


def _available_cases() -> list[Path]:
    return sorted(DATA_DIR.glob("case*.mat")) + sorted(VITAL_DIR.glob("vitaldb_case*.npz"))


def _case_label(path: Path) -> str:
    if path.suffix.lower() == ".npz":
        return f"VitalDB · {path.stem.replace('vitaldb_', '')}"
    return f"Figshare · {path.stem}"


CASE_PATHS = _available_cases()
DEFAULT_CASE = next(
    (path for path in CASE_PATHS if path.stem == "case19"),
    CASE_PATHS[0] if CASE_PATHS else None,
)


def _fast_replay_case(
    model: torch.nn.Module,
    case: EEGCase,
    config: PreprocessConfig,
    *,
    stride_seconds: float = 1.0,
    smoothing_alpha: float = 0.25,
    min_quality: float = DEFAULT_MIN_SIGNAL_QUALITY,
    device: str = "cpu",
) -> list[RealtimePrediction]:
    """Vectorize independent model calls while preserving streaming semantics."""

    eeg = np.asarray(case.eeg, dtype=np.float32).reshape(-1)
    if eeg.size and not np.isfinite(eeg).all():
        raise ValueError("samples devem ser finitas no modo streaming")
    window_samples = config.window_samples
    stride_samples = max(1, int(round(stride_seconds * config.sampling_rate)))
    starts = np.arange(0, max(eeg.size - window_samples + 1, 0), stride_samples, dtype=int)
    if starts.size == 0:
        return []

    processed = StreamingPreprocessor(config).process(eeg)
    raw_windows = np.lib.stride_tricks.sliding_window_view(eeg, window_samples)[starts]
    processed_windows = np.lib.stride_tricks.sliding_window_view(processed, window_samples)[starts]
    qualities = np.asarray([signal_quality(window, config) for window in raw_windows], dtype=float)
    valid = qualities >= min_quality
    raw_predictions = np.full(starts.size, np.nan, dtype=float)
    if valid.any():
        model.eval()
        with torch.inference_mode():
            for batch_start in range(0, int(valid.sum()), 512):
                valid_indices = np.flatnonzero(valid)[batch_start : batch_start + 512]
                batch = torch.from_numpy(np.asarray(processed_windows[valid_indices], dtype=np.float32)[:, None, :]).to(device)
                raw_predictions[valid_indices] = torch.clamp(model(batch), 0.0, 100.0).detach().cpu().numpy()

    smoothed_predictions = np.full(starts.size, np.nan, dtype=float)
    stages: list[str] = []
    previous = float("nan")
    for index, raw_bis in enumerate(raw_predictions):
        if not np.isfinite(raw_bis):
            previous = float("nan")
            stages.append("abstain")
            continue
        previous = float(raw_bis) if not np.isfinite(previous) else smoothing_alpha * float(raw_bis) + (1 - smoothing_alpha) * previous
        smoothed_predictions[index] = previous
        stages.append(bis_stage(previous))
    return [
        RealtimePrediction(
            sample_index=int(start + window_samples),
            elapsed_seconds=float(start + window_samples) / case.sampling_rate,
            raw_bis=None if not np.isfinite(raw_bis) else float(raw_bis),
            smoothed_bis=None if not np.isfinite(smoothed) else float(smoothed),
            stage=stage,
            quality=float(quality),
        )
        for start, raw_bis, smoothed, stage, quality in zip(
            starts, raw_predictions, smoothed_predictions, stages, qualities, strict=False
        )
    ]


@lru_cache(maxsize=16)
def _replay_payload(path_string: str) -> ReplayPayload:
    """Run the causal replay once and keep a compact browser representation."""

    if MODEL is None or PREPROCESS is None:
        raise RuntimeError(MODEL_ERROR or "Checkpoint indisponível")
    case = load_case(path_string)
    predictions = _fast_replay_case(
        MODEL,
        case,
        PREPROCESS,
        stride_seconds=1.0,
        min_quality=DEFAULT_MIN_SIGNAL_QUALITY,
        device="cpu",
    )
    eeg = np.asarray(case.eeg, dtype=float)
    eeg_indices = np.linspace(0, max(eeg.size - 1, 0), min(REPLAY_EEG_POINTS, eeg.size), dtype=int)
    return ReplayPayload(
        case=case,
        prediction_times=np.asarray([item.elapsed_seconds for item in predictions], dtype=float),
        raw_predictions=np.asarray(
            [np.nan if item.raw_bis is None else item.raw_bis for item in predictions], dtype=float
        ),
        smoothed_predictions=np.asarray(
            [np.nan if item.smoothed_bis is None else item.smoothed_bis for item in predictions], dtype=float
        ),
        qualities=np.asarray([item.quality for item in predictions], dtype=float),
        stages=tuple(item.stage for item in predictions),
        eeg_times=eeg_indices.astype(float) / case.sampling_rate,
        eeg_values=eeg[eeg_indices] if eeg.size else np.empty(0, dtype=float),
    )


def _card(title: str, value: str, detail: str, tone: str = "blue") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className=f"metric-card metric-{tone}",
    )


def _live_card(title: str, value_id: str, detail_id: str, tone: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div("—", id=value_id, className="metric-value"),
            html.Div("Aguardando replay", id=detail_id, className="metric-detail"),
        ],
        className=f"metric-card metric-{tone}",
    )


def _figure_layout(title: str, *, height: int = 330) -> dict[str, object]:
    return {
        "title": {"text": title, "font": {"size": 17, "color": COLORS["ink"]}},
        "height": height,
        "margin": {"l": 58, "r": 30, "t": 62, "b": 50},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#FAFCFE",
        "font": {"family": "Inter, Arial, sans-serif", "color": COLORS["ink"]},
        "hoverlabel": {"bgcolor": COLORS["navy"], "font": {"color": "white"}},
        "legend": {"orientation": "h", "y": 1.04, "x": 0, "font": {"size": 11}},
        "hovermode": "x unified",
    }


def _empty_figure(title: str, message: str, *, height: int = 330) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**_figure_layout(title, height=height))
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": COLORS["muted"]},
    )
    return figure


def _metric_value(metrics: object, key: str) -> float:
    if not isinstance(metrics, dict):
        return float("nan")
    return _finite_number(metrics.get(key))


def _metric_format(key: str, value: object) -> str:
    number = _finite_number(value)
    if key in {"stage_accuracy", "stage_macro_f1"} and np.isfinite(number):
        return f"{number * 100:.1f}%"
    digits = 3 if key == "pearson_r" else 1 if key in {"bias", "mae", "rmse"} else 3
    return _format_number(number, digits)


def _evidence_mae_figure() -> go.Figure:
    values = [_metric_value(HOLDOUT_METRICS, "mae"), _metric_value(EXTERNAL_METRICS, "mae")]
    if not np.isfinite(values).any():
        return _empty_figure("MAE da rede contra o BIS", "Relatório de métricas indisponível")
    figure = go.Figure(
        go.Bar(
            x=["Holdout interno", "VitalDB externo"],
            y=values,
            text=[_format_number(value, 1) for value in values],
            textposition="outside",
            marker_color=[COLORS["blue"], COLORS["orange"]],
            hovertemplate="%{x}<br>MAE: %{y:.2f} pontos BIS<extra></extra>",
        )
    )
    figure.update_layout(**_figure_layout("MAE da rede contra o BIS"))
    figure.update_yaxes(title="Erro médio absoluto (pontos BIS)", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _evidence_pearson_figure() -> go.Figure:
    values = [_metric_value(HOLDOUT_METRICS, "pearson_r"), _metric_value(EXTERNAL_METRICS, "pearson_r")]
    if not np.isfinite(values).any():
        return _empty_figure("Correlação da rede com o BIS", "Relatório de métricas indisponível")
    figure = go.Figure(
        go.Bar(
            x=["Holdout interno", "VitalDB externo"],
            y=values,
            text=[_format_number(value, 3) for value in values],
            textposition="outside",
            marker_color=[COLORS["teal"], COLORS["red"]],
            hovertemplate="%{x}<br>Pearson r: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(**_figure_layout("Correlação da rede com o BIS"))
    figure.update_yaxes(title="Pearson r", range=[-0.2, 1.0], gridcolor=COLORS["line"])
    return figure


def _offset_points() -> list[dict[str, object]]:
    """Normalize the report schema used by older and newer exports."""

    raw_points = OFFSET_REPORT.get("results") or OFFSET_REPORT.get("offsets") or []
    if not isinstance(raw_points, list):
        return []
    points: list[dict[str, object]] = []
    for item in raw_points:
        if not isinstance(item, dict) or not isinstance(item.get("metrics"), dict):
            continue
        offset = _finite_number(item.get("offset_seconds"))
        mae = _metric_value(item["metrics"], "mae")
        pearson = _metric_value(item["metrics"], "pearson_r")
        if np.isfinite(offset) and np.isfinite(mae) and np.isfinite(pearson):
            points.append({"offset": offset, "mae": mae, "pearson": pearson, "n": item.get("n_windows")})
    return points


def _evidence_offset_figure() -> go.Figure:
    points = _offset_points()
    if len(points) < 2:
        return _empty_figure("Sensibilidade ao alinhamento EEG–BIS", "Relatório de offset indisponível")
    offsets = [float(item["offset"]) for item in points]
    maes = [float(item["mae"]) for item in points]
    pearsons = [float(item["pearson"]) for item in points]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=offsets,
            y=maes,
            mode="lines+markers",
            name="MAE",
            line={"color": COLORS["orange"], "width": 3},
            marker={"size": 8},
            customdata=[[item["n"]] for item in points],
            hovertemplate="offset %{x:.0f}s<br>MAE %{y:.2f}<br>janelas %{customdata[0]}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=offsets,
            y=pearsons,
            mode="lines+markers",
            name="Pearson r",
            line={"color": COLORS["teal"], "width": 3},
            marker={"size": 8},
            customdata=[[item["n"]] for item in points],
            hovertemplate="offset %{x:.0f}s<br>Pearson %{y:.3f}<br>janelas %{customdata[0]}<extra></extra>",
        ),
        secondary_y=True,
    )
    pearson_low = max(-1.0, min(pearsons) - 0.05)
    pearson_high = min(1.0, max(pearsons) + 0.05)
    figure.update_layout(**_figure_layout("Sensibilidade exploratória ao alinhamento EEG–BIS", height=340))
    figure.update_xaxes(title="Offset do rótulo BIS (s)", gridcolor=COLORS["line"])
    figure.update_yaxes(title_text="MAE (pontos BIS)", secondary_y=False, gridcolor=COLORS["line"])
    figure.update_yaxes(title_text="Pearson r", secondary_y=True, range=[pearson_low, pearson_high])
    return figure


def _reference_at_or_before(case: EEGCase, seconds: float) -> tuple[float | None, float | None]:
    if case.bis.size == 0:
        return None, None
    times = np.arange(case.bis.size, dtype=float) * case.label_interval_seconds
    index = int(np.searchsorted(times, seconds, side="right") - 1)
    if index < 0:
        return None, None
    value = _finite_number(case.bis[index])
    return (None, None) if not np.isfinite(value) else (value, float(times[index]))


def _masked_series(times: np.ndarray, values: np.ndarray, seconds: float) -> list[float | None]:
    return [
        float(value) if time <= seconds and np.isfinite(value) else None
        for time, value in zip(times, values, strict=False)
    ]


def _prediction_index(payload: ReplayPayload, seconds: float) -> int:
    if payload.prediction_times.size == 0:
        return -1
    index = int(np.searchsorted(payload.prediction_times, seconds, side="right") - 1)
    return index if index >= 0 else -1


def _replay_store_payload(payload: ReplayPayload) -> dict[str, object]:
    case = payload.case
    bis_times = np.arange(case.bis.size, dtype=float) * case.label_interval_seconds
    if case.source_dataset:
        case_label = f"VitalDB · {case.case_id.replace('vitaldb_', '')}"
    else:
        case_label = f"Figshare · {case.case_id}"
    return {
        "case_id": case.case_id,
        "case_label": case_label,
        "duration": float(case.duration_seconds),
        "bis_times": _json_series(bis_times),
        "bis_values": _json_series(case.bis),
        "prediction_times": _json_series(payload.prediction_times),
        "raw_predictions": _json_series(payload.raw_predictions),
        "smoothed_predictions": _json_series(payload.smoothed_predictions),
        "qualities": _json_series(payload.qualities),
        "stages": list(payload.stages),
    }


def _replay_figure(payload: ReplayPayload, seconds: float, eeg_window_seconds: float) -> go.Figure:
    """Build one base figure; the browser changes only revealed y-values."""

    case = payload.case
    current = max(0.0, min(float(seconds), case.duration_seconds))
    window_start = max(0.0, current - eeg_window_seconds)
    bis_times = np.arange(case.bis.size, dtype=float) * case.label_interval_seconds
    finite_eeg = payload.eeg_values[np.isfinite(payload.eeg_values)]
    eeg_min = float(finite_eeg.min()) if finite_eeg.size else -1.0
    eeg_max = float(finite_eeg.max()) if finite_eeg.size else 1.0
    if eeg_min == eeg_max:
        eeg_min, eeg_max = eeg_min - 1.0, eeg_max + 1.0
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        row_heights=[0.47, 0.53],
        subplot_titles=(
            f"EEG frontal recebido · janela causal de {eeg_window_seconds:.0f}s",
            "Saída da rede neural contra o BIS de referência",
        ),
    )
    figure.add_trace(
        go.Scattergl(
            x=payload.eeg_times,
            y=payload.eeg_values,
            mode="lines",
            name="EEG frontal",
            line={"color": COLORS["blue"], "width": 1.1},
            hovertemplate="t=%{x:.2f}s<br>EEG=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=bis_times,
            y=_masked_series(bis_times, case.bis.astype(float), current),
            mode="lines+markers",
            name="BIS referência",
            line={"color": COLORS["navy"], "width": 2.4, "shape": "hv"},
            marker={"size": 5},
            connectgaps=False,
            hovertemplate="t=%{x:.0f}s<br>BIS=%{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=payload.prediction_times,
            y=_masked_series(payload.prediction_times, payload.raw_predictions, current),
            mode="lines",
            name="CNN bruta",
            line={"color": COLORS["purple"], "width": 1.2, "dash": "dot"},
            connectgaps=False,
            hovertemplate="t=%{x:.1f}s<br>CNN bruta=%{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=payload.prediction_times,
            y=_masked_series(payload.prediction_times, payload.smoothed_predictions, current),
            mode="lines+markers",
            name="CNN suavizada",
            line={"color": COLORS["teal"], "width": 2.8},
            marker={"size": 4},
            connectgaps=False,
            hovertemplate="t=%{x:.1f}s<br>CNN suavizada=%{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=[current, current],
            y=[eeg_min, eeg_max],
            mode="lines",
            name="cursor",
            line={"color": COLORS["red"], "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[current, current],
            y=[0, 100],
            mode="lines",
            name="cursor",
            line={"color": COLORS["red"], "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for threshold in (40, 60, 80):
        figure.add_hline(y=threshold, line={"color": COLORS["line"], "width": 1, "dash": "dot"}, row=2, col=1)
    figure.update_layout(**_figure_layout("Replay sincronizado · atualização local", height=650))
    figure.update_layout(uirevision=f"replay-{case.case_id}-{eeg_window_seconds:g}", meta={"eegWindow": eeg_window_seconds})
    figure.update_xaxes(title="Tempo do replay (s)", range=[window_start, max(window_start + 5, current)], gridcolor=COLORS["line"], row=1, col=1)
    figure.update_xaxes(title="Tempo do replay (s)", range=[0, max(10, current + 2)], gridcolor=COLORS["line"], row=2, col=1)
    figure.update_yaxes(title="Amplitude", gridcolor=COLORS["line"], row=1, col=1)
    figure.update_yaxes(title="Índice BIS (0–100)", range=[0, 100], gridcolor=COLORS["line"], row=2, col=1)
    return figure


def _quality_figure(payload: ReplayPayload, seconds: float) -> go.Figure:
    current = max(0.0, min(float(seconds), payload.case.duration_seconds))
    figure = go.Figure(
        [
            go.Scatter(
                x=payload.prediction_times,
                y=_masked_series(payload.prediction_times, payload.qualities, current),
                mode="lines+markers",
                name="Qualidade",
                line={"color": COLORS["green"], "width": 2.3},
                marker={"size": 4},
                connectgaps=False,
                hovertemplate="t=%{x:.1f}s<br>qualidade=%{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=[current, current],
                y=[0, 1.05],
                mode="lines",
                name="cursor",
                line={"color": COLORS["red"], "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
    )
    figure.add_hline(y=DEFAULT_MIN_SIGNAL_QUALITY, line={"color": COLORS["orange"], "width": 1.5, "dash": "dash"}, annotation_text=f"gate {DEFAULT_MIN_SIGNAL_QUALITY:.2f}", annotation_position="bottom right")
    figure.update_layout(**_figure_layout("Qualidade do sinal · gate de emissão", height=280))
    figure.update_layout(uirevision=f"quality-{payload.case.case_id}")
    figure.update_xaxes(title="Tempo (s)", gridcolor=COLORS["line"])
    figure.update_yaxes(title="Score", range=[0, 1.05], gridcolor=COLORS["line"])
    return figure


def _case_meta(path_value: str | None) -> html.Div:
    if not path_value:
        return html.Div("Nenhum caso disponível.", className="case-meta")
    try:
        case = load_case(path_value)
    except Exception as error:
        return html.Div(f"Não foi possível abrir o caso: {error}", className="case-meta case-error")
    return html.Div(
        [
            html.Strong(_case_label(Path(path_value))),
            html.Span(f" · {_format_clock(case.duration_seconds)} · {case.sampling_rate} Hz · {case.bis.size:,} pontos BIS"),
        ],
        className="case-meta",
    )


def _case_metrics(payload: ReplayPayload) -> dict[str, float]:
    bis_times = np.arange(payload.case.bis.size, dtype=float) * payload.case.label_interval_seconds
    references = []
    for time in payload.prediction_times:
        index = int(np.searchsorted(bis_times, time, side="right") - 1)
        references.append(payload.case.bis[index] if index >= 0 else np.nan)
    target = np.asarray(references, dtype=float)
    prediction = payload.smoothed_predictions.astype(float)
    mask = np.isfinite(target) & np.isfinite(prediction)
    if not mask.any():
        return {key: float("nan") for key in ("n", "mae", "rmse", "bias", "pearson_r")}
    target = target[mask]
    prediction = prediction[mask]
    centered_target = target - target.mean()
    centered_prediction = prediction - prediction.mean()
    denominator = float(np.sqrt(np.sum(centered_target**2) * np.sum(centered_prediction**2)))
    return {
        "n": float(target.size),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "bias": float(np.mean(prediction - target)),
        "pearson_r": float(np.sum(centered_target * centered_prediction) / denominator) if denominator > 0 else float("nan"),
    }


def _causal_ema(values: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    """Apply a causal EMA for display without changing stored predictions."""

    source = np.asarray(values, dtype=float).reshape(-1)
    smoothed = np.full(source.shape, np.nan, dtype=float)
    previous = float("nan")
    for index, value in enumerate(source):
        if not np.isfinite(value):
            previous = float("nan")
            continue
        previous = float(value) if not np.isfinite(previous) else alpha * float(value) + (1 - alpha) * previous
        smoothed[index] = previous
    return smoothed


def _trajectory_figure(payload: ReplayPayload) -> tuple[go.Figure, go.Figure]:
    case = payload.case
    bis_times = np.arange(case.bis.size, dtype=float) * case.label_interval_seconds
    bis_values = case.bis.astype(float)
    reference_at_prediction = []
    for time in payload.prediction_times:
        index = int(np.searchsorted(bis_times, time, side="right") - 1)
        reference_at_prediction.append(bis_values[index] if index >= 0 else np.nan)
    references = np.asarray(reference_at_prediction, dtype=float)
    visual_predictions = _causal_ema(payload.smoothed_predictions)
    error = payload.smoothed_predictions - references
    visual_error = visual_predictions - references
    valid_error = np.isfinite(error)
    valid_visual_error = np.isfinite(visual_error)

    figure = go.Figure(
        [
            go.Scattergl(x=bis_times, y=bis_values, mode="lines", name="BIS referência", line={"color": COLORS["navy"], "width": 2.5, "shape": "hv"}, connectgaps=False, hovertemplate="t=%{x:.0f}s<br>BIS=%{y:.1f}<extra></extra>"),
            go.Scattergl(x=payload.prediction_times, y=payload.raw_predictions, mode="lines", name="CNN bruta", visible="legendonly", line={"color": COLORS["purple"], "width": 1.1, "dash": "dot"}, connectgaps=False, hovertemplate="t=%{x:.1f}s<br>CNN bruta=%{y:.1f}<extra></extra>"),
            go.Scattergl(x=payload.prediction_times, y=payload.smoothed_predictions, mode="lines", name="CNN causal original", visible="legendonly", line={"color": COLORS["teal"], "width": 1.5, "dash": "dash"}, connectgaps=False, hovertemplate="t=%{x:.1f}s<br>CNN causal=%{y:.1f}<extra></extra>"),
            go.Scattergl(x=payload.prediction_times, y=visual_predictions, mode="lines", name="CNN · EMA visual (5 s)", line={"color": COLORS["teal"], "width": 3.1}, connectgaps=False, customdata=payload.smoothed_predictions, hovertemplate="t=%{x:.1f}s<br>CNN visual=%{y:.1f}<br>CNN causal=%{customdata:.1f}<extra></extra>"),
        ]
    )
    figure.update_layout(**_figure_layout(f"Trajetória completa · {case.case_id} · BIS contra CNN", height=470), uirevision=f"trajectory-{case.case_id}")
    figure.update_xaxes(title="Tempo do caso (s)", range=[0, max(10, case.duration_seconds)], gridcolor=COLORS["line"])
    figure.update_yaxes(title="Índice BIS (0–100)", range=[0, 100], gridcolor=COLORS["line"])

    error_figure = go.Figure(
        [
            go.Scattergl(x=payload.prediction_times[valid_error], y=error[valid_error], mode="lines", name="Erro causal original", visible="legendonly", line={"color": COLORS["orange"], "width": 1.2, "dash": "dot"}, hovertemplate="t=%{x:.1f}s<br>erro causal=%{y:.1f} pontos BIS<extra></extra>"),
            go.Scattergl(x=payload.prediction_times[valid_visual_error], y=visual_error[valid_visual_error], mode="lines", name="Erro · EMA visual (5 s)", line={"color": COLORS["orange"], "width": 2.4}, hovertemplate="t=%{x:.1f}s<br>erro visual=%{y:.1f} pontos BIS<extra></extra>"),
        ]
    )
    error_figure.add_hline(y=0, line={"color": COLORS["navy"], "width": 1.5, "dash": "dash"})
    error_figure.update_layout(**_figure_layout(f"Erro ao longo do caso · {case.case_id} · CNN − BIS", height=300), uirevision=f"trajectory-error-{case.case_id}")
    error_figure.update_xaxes(title="Tempo do caso (s)", gridcolor=COLORS["line"])
    error_figure.update_yaxes(title="Erro (pontos BIS)", gridcolor=COLORS["line"])
    return figure, error_figure


def _table(headers: list[str], rows: list[list[object]], class_name: str = "data-table") -> html.Table:
    return html.Table([html.Thead(html.Tr([html.Th(header) for header in headers])), html.Tbody([html.Tr([html.Td(str(value)) for value in row]) for row in rows])], className=class_name)


def _initial_trajectory_cards() -> list[html.Div]:
    return [
        _card("MAE do caso", "—", "saída causal original contra BIS", "blue"),
        _card("RMSE do caso", "—", "Erro quadrático médio", "orange"),
        _card("Bias do caso", "—", "CNN − BIS", "purple"),
        _card("Pearson do caso", "—", "Associação temporal", "teal"),
        _card("Predições válidas", "—", "Janelas emitidas pelo gate", "green"),
    ]


def _trajectory_cards(payload: ReplayPayload) -> list[html.Div]:
    metrics = _case_metrics(payload)
    valid = int(np.isfinite(payload.smoothed_predictions).sum())
    return [
        _card("MAE do caso", _format_number(metrics["mae"], 1), "saída causal original contra BIS", "blue"),
        _card("RMSE do caso", _format_number(metrics["rmse"], 1), "Erro quadrático médio", "orange"),
        _card("Bias do caso", _format_number(metrics["bias"], 1), "CNN − BIS", "purple"),
        _card("Pearson do caso", _format_number(metrics["pearson_r"], 3), "Associação temporal", "teal"),
        _card("Predições válidas", f"{valid:,}", f"de {len(payload.prediction_times):,} janelas", "green"),
    ]


def _history_figure() -> go.Figure:
    history = MODEL_METADATA.get("history", [])
    if not isinstance(history, list) or not history:
        return _empty_figure("Histórico de treinamento", "Histórico não encontrado")
    rows = [item for item in history if isinstance(item, dict)]
    epochs = [_finite_number(item.get("epoch")) for item in rows]
    train_loss = [_finite_number(item.get("train_loss")) for item in rows]
    validation_mae = [_finite_number(item.get("validation_mae")) for item in rows]
    validation_rmse = [_finite_number(item.get("validation_rmse")) for item in rows]
    figure = go.Figure(
        [
            go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Loss treino", line={"color": COLORS["blue"], "width": 2.3}),
            go.Scatter(x=epochs, y=validation_mae, mode="lines+markers", name="MAE validação", line={"color": COLORS["teal"], "width": 2.3}),
            go.Scatter(x=epochs, y=validation_rmse, mode="lines+markers", name="RMSE validação", line={"color": COLORS["orange"], "width": 2.3}),
        ]
    )
    figure.update_layout(**_figure_layout("Histórico de treinamento e validação", height=350))
    figure.update_xaxes(title="Época", dtick=1, gridcolor=COLORS["line"])
    figure.update_yaxes(title="Valor (pontos BIS / loss)", gridcolor=COLORS["line"])
    return figure


def _model_parameter_table() -> html.Table:
    if MODEL is None or not hasattr(MODEL, "state_dict"):
        return _table(["Camada", "Formato", "Parâmetros"], [["—", "Checkpoint indisponível", "—"]])
    rows = []
    for name, tensor in MODEL.state_dict().items():
        shape = " × ".join(str(int(value)) for value in tensor.shape)
        rows.append([name, shape, f"{int(tensor.numel()):,}"])
    return _table(["Tensor / camada", "Formato", "Parâmetros"], rows)


def _model_architecture_table() -> html.Table:
    rows = [
        ["Entrada", "1 canal EEG", "640 amostras · janela de 5 s"],
        ["Conv1d + BN + GELU", "1 → 32", "kernel 7 · max-pool 2"],
        ["Conv1d + BN + GELU", "32 → 64", "kernel 7 · max-pool 2"],
        ["Conv1d + BN + GELU", "64 → 128", "kernel 5 · max-pool 2"],
        ["Conv1d + BN + GELU", "128 → 128", "kernel 5"],
        ["AdaptiveAvgPool1d", "128 → 128 × 1", "agregação temporal"],
        ["MLP", "128 → 64 → 1", "GELU · dropout 0,2"],
        ["Saída", "sigmoid × 100", "BIS estimado entre 0 e 100"],
    ]
    return _table(["Bloco", "Dimensão", "Decisão / função"], rows)


def _case_id_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (int(digits) if digits else 10**9, text)


def _case_count_mae_figure() -> go.Figure:
    values = EXTERNAL_REPORT.get("per_case", [])
    if not isinstance(values, list) or not values:
        return _empty_figure("MAE acumulado conforme entram os casos", "Métricas externas por caso indisponíveis", height=430)
    rows = [item for item in values if isinstance(item, dict)]
    rows.sort(key=lambda item: _case_id_sort_key(item.get("case_id", "")))
    if not rows:
        return _empty_figure("MAE acumulado conforme entram os casos", "Métricas externas por caso indisponíveis", height=430)
    counts = np.arange(1, len(rows) + 1, dtype=int)
    maes = np.asarray([_finite_number(item.get("mae")) for item in rows], dtype=float)
    windows = np.asarray([max(0, int(_finite_number(item.get("n_windows"), 0))) for item in rows], dtype=float)
    valid = np.isfinite(maes) & (windows > 0)
    if not valid.any():
        return _empty_figure("MAE acumulado conforme entram os casos", "Métricas externas por caso indisponíveis", height=430)
    cumulative_mae = np.cumsum(np.where(valid, maes * windows, 0.0)) / np.maximum(np.cumsum(np.where(valid, windows, 0.0)), 1.0)
    labels = [str(item.get("case_id", "caso")).replace("vitaldb_", "") for item in rows]
    customdata = np.asarray([[label, int(window)] for label, window in zip(labels, windows, strict=False)], dtype=object)
    overall_mae = _metric_value(EXTERNAL_METRICS, "mae")
    figure = go.Figure(
        [
            go.Bar(x=counts, y=maes, name="MAE de cada caso", marker_color=COLORS["orange"], opacity=0.58, customdata=customdata, hovertemplate="%{customdata[0]}<br>casos acumulados=%{x}<br>MAE do caso=%{y:.2f} pontos BIS<br>janelas=%{customdata[1]:,}<extra></extra>"),
            go.Scatter(x=counts, y=cumulative_mae, mode="lines+markers", name="MAE acumulado", line={"color": COLORS["navy"], "width": 2.8}, marker={"color": COLORS["navy"], "size": 7}, hovertemplate="%{x} casos acumulados<br>MAE agregado=%{y:.2f} pontos BIS<extra></extra>"),
        ]
    )
    figure.add_hline(y=overall_mae, line={"color": COLORS["teal"], "width": 1.5, "dash": "dash"}, annotation_text=f"agregado dos 15 casos = {overall_mae:.2f}", annotation_position="top left", annotation_font_color=COLORS["teal"])
    figure.update_layout(**_figure_layout("MAE acumulado conforme entram os casos VitalDB", height=430), barmode="overlay")
    figure.update_xaxes(title="Número de casos acumulados (ordem numérica do relatório)", dtick=1, gridcolor=COLORS["line"])
    figure.update_yaxes(title="MAE (pontos BIS)", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _statistics_cards() -> list[html.Div]:
    return [
        _card("Janelas holdout", _format_number(_metric_value(HOLDOUT_METRICS, "n"), 0), "5 casos Figshare", "blue"),
        _card("Janelas VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "n"), 0), "15 casos externos", "orange"),
        _card("MAE holdout", _format_number(_metric_value(HOLDOUT_METRICS, "mae"), 1), "pontos BIS", "teal"),
        _card("MAE VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "mae"), 1), "pontos BIS", "red"),
        _card("Pearson holdout", _format_number(_metric_value(HOLDOUT_METRICS, "pearson_r"), 3), "associação temporal", "purple"),
        _card("Pearson VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "pearson_r"), 3), "mudança de domínio", "navy"),
    ]


def _error_metric_figure() -> go.Figure:
    figure = go.Figure()
    for key, color in (("mae", COLORS["blue"]), ("rmse", COLORS["orange"])):
        figure.add_trace(go.Bar(name=METRIC_LABELS[key], x=["Holdout interno", "VitalDB externo"], y=[_metric_value(HOLDOUT_METRICS, key), _metric_value(EXTERNAL_METRICS, key)], marker_color=color, text=[_metric_format(key, _metric_value(HOLDOUT_METRICS, key)), _metric_format(key, _metric_value(EXTERNAL_METRICS, key))], textposition="outside", hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f} pontos BIS<extra></extra>"))
    figure.update_layout(**_figure_layout("Erros contínuos por conjunto", height=360), barmode="group")
    figure.update_yaxes(title="Pontos BIS", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _association_metric_figure() -> go.Figure:
    figure = go.Figure()
    for dataset, metrics, color in (("Holdout interno", HOLDOUT_METRICS, COLORS["teal"]), ("VitalDB externo", EXTERNAL_METRICS, COLORS["orange"])):
        keys = ["pearson_r", "stage_accuracy", "stage_macro_f1"]
        values = [_metric_value(metrics, key) for key in keys]
        figure.add_trace(go.Bar(name=dataset, x=[METRIC_LABELS[key] for key in keys], y=values, marker_color=color, text=[_metric_format(key, value) for key, value in zip(keys, values, strict=False)], textposition="outside", hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>"))
    figure.update_layout(**_figure_layout("Associação e classificação por conjunto", height=390), barmode="group")
    figure.update_yaxes(title="Score (0–1)", range=[0, 1.08], gridcolor=COLORS["line"])
    return figure


def _bootstrap_figure() -> go.Figure:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("MAE · IC 95% por caso", "Pearson r · IC 95% por caso"), horizontal_spacing=0.13)
    for column, key, title, color in ((1, "mae", "MAE", COLORS["blue"]), (2, "pearson_r", "Pearson r", COLORS["teal"])):
        means = []
        lower = []
        upper = []
        for report in (HOLDOUT_REPORT, EXTERNAL_REPORT):
            bootstrap = report.get("case_bootstrap", {})
            interval = bootstrap.get(key, {}) if isinstance(bootstrap, dict) else {}
            means.append(_finite_number(interval.get("mean")))
            lower.append(_finite_number(interval.get("lower_95")))
            upper.append(_finite_number(interval.get("upper_95")))
        figure.add_trace(go.Scatter(x=["Holdout", "VitalDB"], y=means, mode="markers", name=title, marker={"color": color, "size": 12}, error_y={"type": "data", "symmetric": False, "array": [hi - mean for hi, mean in zip(upper, means, strict=False)], "arrayminus": [mean - lo for mean, lo in zip(means, lower, strict=False)]}, hovertemplate="%{x}<br>média %{y:.3f}<extra></extra>", showlegend=False), row=1, col=column)
        figure.update_yaxes(title=title, gridcolor=COLORS["line"], row=1, col=column)
    figure.update_layout(**_figure_layout("Incerteza entre casos · bootstrap", height=340))
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=1)
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=2)
    return figure


def _per_case_figure() -> go.Figure:
    values = EXTERNAL_REPORT.get("per_case", [])
    if not isinstance(values, list) or not values:
        return _empty_figure("VitalDB · MAE por caso", "Métricas por caso indisponíveis", height=430)
    rows = [item for item in values if isinstance(item, dict)]
    rows.sort(key=lambda item: _finite_number(item.get("mae")), reverse=True)
    labels = [str(item.get("case_id", "caso")) for item in rows]
    maes = [_finite_number(item.get("mae")) for item in rows]
    pearsons = [_finite_number(item.get("pearson_r")) for item in rows]
    figure = go.Figure(go.Bar(x=maes, y=labels, orientation="h", marker_color=COLORS["orange"], text=[_format_number(value, 1) for value in maes], textposition="outside", customdata=np.asarray(pearsons)[:, None], hovertemplate="%{y}<br>MAE %{x:.2f}<br>Pearson %{customdata[0]:.3f}<extra></extra>"))
    figure.update_layout(**_figure_layout("VitalDB · erro por caso", height=580))
    figure.update_xaxes(title="MAE (pontos BIS)", rangemode="tozero", gridcolor=COLORS["line"])
    figure.update_yaxes(title="Caso", autorange="reversed", gridcolor=COLORS["line"])
    return figure


def _tab_intro(kicker: str, title: str, lead: str) -> html.Div:
    return html.Div([html.Div(kicker, className="section-kicker"), html.H2(title, className="section-title"), html.P(lead, className="section-lead")], className="section-intro")


def _build_overview_tab() -> html.Div:
    return html.Div(
        [
            _tab_intro("PAINEL DE EVIDÊNCIA", "A comparação começa pelo resultado, não pelo replay", "Esta aba resume a diferença entre o desempenho no holdout Figshare e a validação externa VitalDB. O objetivo é separar erro, associação e sensibilidade temporal para não transformar uma única métrica em uma conclusão exagerada."),
            html.Div([_card("Holdout · MAE", _format_number(_metric_value(HOLDOUT_METRICS, "mae"), 1), "Figshare · 5 casos", "blue"), _card("Holdout · Pearson", _format_number(_metric_value(HOLDOUT_METRICS, "pearson_r"), 3), "CNN versus BIS", "teal"), _card("VitalDB · MAE", _format_number(_metric_value(EXTERNAL_METRICS, "mae"), 1), "15 casos · sem retreino", "orange"), _card("VitalDB · Pearson", _format_number(_metric_value(EXTERNAL_METRICS, "pearson_r"), 3), "mudança de domínio", "red")], className="metric-grid"),
            html.Div([html.Div(dcc.Graph(figure=_evidence_mae_figure(), config={"displayModeBar": False}), className="chart-card"), html.Div(dcc.Graph(figure=_evidence_pearson_figure(), config={"displayModeBar": False}), className="chart-card")], className="chart-grid two-col"),
            html.Div([html.Div(dcc.Graph(figure=_evidence_offset_figure(), config={"displayModeBar": False}), className="chart-card"), html.Div([html.Div("LEITURA PARA A DECISÃO", className="mini-kicker"), html.H3("Por que os gráficos estão separados?"), html.P("1. O holdout mostra a execução reproduzível no domínio do treino."), html.P("2. O VitalDB mostra que bom desempenho interno não prova generalização."), html.P("3. O offset é exploratório e não deve ser escolhido pós-hoc para pacientes."), html.Div("O relatório agora lê a série `results` do experimento de offset e mostra todos os nove pontos calculados.", className="callout")], className="explanation-card")], className="chart-grid two-col lower-evidence"),
        ],
        className="tab-panel",
    )


def _build_trajectory_tab(options: list[dict[str, object]], default_value: str | None) -> html.Div:
    trajectory_results = html.Div(
        [
            html.Div(id="trajectory-cards", children=_initial_trajectory_cards(), className="metric-grid trajectory-metrics"),
            html.Div(dcc.Graph(id="trajectory-figure", figure=_empty_figure("Trajetória completa · BIS contra CNN", "Carregando o caso selecionado", height=470), config={"displayModeBar": False}), className="chart-card"),
            html.Div(dcc.Graph(id="trajectory-error-figure", figure=_empty_figure("Erro ao longo do caso · CNN − BIS", "Carregando o caso selecionado", height=300), config={"displayModeBar": False}), className="chart-card trajectory-error-chart"),
        ],
        className="trajectory-results",
    )
    return html.Div(
        [
            _tab_intro("VISÃO RETROSPECTIVA", "O caso inteiro de uma vez", "Aqui a curva completa já aparece como uma análise retrospectiva: a pessoa não precisa esperar o relógio do replay para ver como BIS e CNN se comportaram durante todo o caso selecionado."),
            html.Div(
                [
                    html.Div([html.Label("Caso para analisar", htmlFor="trajectory-case-selector"), dcc.Dropdown(id="trajectory-case-selector", options=options, value=default_value, clearable=False, searchable=True), html.Div(id="trajectory-meta", children=_case_meta(default_value))], className="control-block case-control"),
                    html.Div([html.Div("LEITURA CORRETA", className="mini-kicker"), html.P("Ao trocar o caso, a tela mostra um carregamento enquanto calcula as janelas causais. O título do gráfico e os cards identificam o caso novo assim que a trajetória termina de atualizar." )], className="explanation-card compact-explanation"),
                ],
                className="control-grid trajectory-controls",
            ),
            dcc.Loading(id="trajectory-loading", type="circle", color=COLORS["teal"], children=trajectory_results),
            html.Div("A linha marinho é o BIS observado; a linha turquesa espessa é uma EMA causal de 5 s aplicada somente para leitura. A saída causal original e o erro original ficam disponíveis pela legenda; cards e métricas não são recalculados com essa suavização visual.", className="legend-note"),
        ],
        className="tab-panel",
    )


def _build_replay_tab(options: list[dict[str, object]], default_value: str | None, duration: float) -> html.Div:
    marks = {0: "0:00", int(min(duration, 60)): "1:00"}
    if duration > 300:
        marks[300] = "5:00"
    marks[int(duration)] = _format_clock(duration)
    return html.Div(
        [_tab_intro("REPLAY OPERACIONAL", "Como se fosse uma cirurgia: EEG entrando, CNN respondendo", "O Replay mantém a causalidade: só revela a CNN depois que a janela EEG anterior foi recebida. O cálculo da rede é feito uma vez; o relógio e a revelação das curvas acontecem localmente no navegador para a simulação não ficar travando."), html.Div([html.Div([html.Label("Caso para reproduzir", htmlFor="case-selector"), dcc.Dropdown(id="case-selector", options=options, value=default_value, clearable=False, searchable=True, className="dark-dropdown"), html.Div(id="case-meta", children=_case_meta(default_value))], className="control-block case-control"), html.Div([html.Label("Velocidade da simulação", htmlFor="speed"), dcc.Slider(id="speed", min=0.25, max=4, step=0.25, value=1, marks={0.25: "0,25×", 1: "1×", 2: "2×", 4: "4×"}, tooltip={"placement": "bottom", "always_visible": True})], className="control-block speed-control"), html.Div([html.Label("Janela EEG exibida", htmlFor="eeg-window"), dcc.Dropdown(id="eeg-window", options=[{"label": f"{value}s", "value": value} for value in (5, 10, 20, 30)], value=10, clearable=False)], className="control-block window-control")], className="control-grid"), html.Div([html.Button("▶ Iniciar replay", id="play-button", n_clicks=0, className="button-primary"), html.Button("↺ Reiniciar", id="reset-button", n_clicks=0, className="button-secondary"), html.Div("O modelo responde após preencher a janela causal de 5 s.", className="replay-hint"), html.Div([html.Span("cursor ", className="clock-label-prefix"), html.Span("00:00", id="replay-clock-label")], className="replay-clock")], className="replay-actions"), dcc.Slider(id="replay-time", min=0, max=duration, step=0.5, value=0, marks=marks, tooltip={"placement": "bottom", "always_visible": True}, className="time-slider"), html.Div([html.Div(id="replay-progress-fill", className="replay-progress-fill")], className="replay-progress-track"), html.Div(["Progresso da simulação: ", html.Span("0,0%", id="replay-progress-text")], className="replay-progress-label"), dcc.Interval(id="replay-interval", interval=REPLAY_INTERVAL_MS, n_intervals=0), dcc.Store(id="play-state", data=False), dcc.Store(id="replay-clock", data=0.0), dcc.Store(id="replay-data", data=None), html.Div(id="replay-status", className="replay-status"), html.Div([_live_card("CNN suavizada", "replay-cnn-value", "replay-cnn-detail", "teal"), _live_card("BIS de referência", "replay-bis-value", "replay-bis-detail", "navy"), _live_card("Erro CNN − BIS", "replay-error-value", "replay-error-detail", "orange"), _live_card("Qualidade", "replay-quality-value", "replay-quality-detail", "green")], className="metric-grid replay-metrics"), html.Div(dcc.Graph(id="replay-figure", figure=_empty_figure("Replay sincronizado · atualização local", "Escolha um caso e inicie o replay", height=650), config={"displayModeBar": False}), className="chart-card replay-main-chart"), html.Div(dcc.Graph(id="quality-figure", figure=_empty_figure("Qualidade do sinal · gate de emissão", "Aguardando o replay", height=280), config={"displayModeBar": False}), className="chart-card"), html.Div([html.Strong("Como ler a tela: "), "a curva azul é o EEG recebido; a linha azul-marinho em degraus é o BIS do arquivo; a linha turquesa é a CNN suavizada; pontilhada roxa é a saída bruta. A atualização visual é local, mas a sequência de predições foi calculada causalmente."], className="legend-note")], className="tab-panel")


def _build_model_tab() -> html.Div:
    model_name = str(MODEL_METADATA.get("model_name", "Conv1DDepthEstimator"))
    preprocess = MODEL_METADATA.get("preprocess_config", {})
    training = MODEL_METADATA.get("training_config", {})
    dataset = MODEL_METADATA.get("dataset_summary", {})
    split = MODEL_METADATA.get("split", {})
    environment = MODEL_METADATA.get("environment", {})
    window_shape = dataset.get("window_shape", [1, 640]) if isinstance(dataset, dict) else [1, 640]
    param_count = parameter_count(MODEL) if MODEL is not None else 0
    train_cases = tuple(str(case) for case in split.get("train_cases", []))
    validation_cases = tuple(str(case) for case in split.get("validation_cases", []))
    test_cases = tuple(str(case) for case in split.get("test_cases", []))
    external_cases = tuple(str(case) for case in EXTERNAL_REPORT.get("case_ids", []))
    figshare_files = sum(path.suffix.lower() == ".mat" for path in CASE_PATHS)
    vitaldb_files = sum(path.suffix.lower() == ".npz" for path in CASE_PATHS)
    checkpoint_cases = int(dataset.get("n_cases", len(train_cases) + len(validation_cases) + len(test_cases))) if isinstance(dataset, dict) else len(train_cases) + len(validation_cases) + len(test_cases)
    config_rows = [["Checkpoint", model_name, str(MODEL_PATH.name)], ["Parâmetros treináveis", f"{param_count:,}", "modelo carregado em CPU" if MODEL is not None else "indisponível"], ["Entrada", " × ".join(str(value) for value in window_shape), "canal × amostras"], ["Amostragem", f"{preprocess.get('sampling_rate', '—')} Hz", "taxa esperada pelo modelo"], ["Janela", f"{preprocess.get('window_seconds', '—')} s", "contexto temporal causal"], ["Filtro", f"{preprocess.get('lowcut_hz', '—')}–{preprocess.get('highcut_hz', '—')} Hz", "band-pass"], ["Escala", f"±{preprocess.get('clip_uv', '—')} µV / {preprocess.get('amplitude_scale_uv', '—')}", "clip e normalização"], ["Gate de qualidade", str(MODEL_METADATA.get("min_quality", DEFAULT_MIN_SIGNAL_QUALITY)), "abaixo disso emite abstain"]]
    train_rows = [["Épocas", training.get("epochs", "—"), "treinamento"], ["Batch", training.get("batch_size", "—"), "janelas por atualização"], ["Learning rate", training.get("learning_rate", "—"), "AdamW"], ["Weight decay", training.get("weight_decay", "—"), "regularização"], ["Seed", training.get("seed", "—"), "reprodutibilidade"], ["Divisão", f"{len(train_cases)}/{len(validation_cases)}/{len(test_cases)}", "train / validação / teste"], ["Ambiente", environment.get("torch", "—"), f"Python {environment.get('python', '—')}"]]
    split_rows = [["Treino", len(train_cases), ", ".join(train_cases)], ["Validação", len(validation_cases), ", ".join(validation_cases)], ["Teste interno", len(test_cases), ", ".join(test_cases)], ["VitalDB externo", len(external_cases), ", ".join(external_cases)]]
    coverage_cards = [_card("Casos no checkpoint", f"{checkpoint_cases}", f"{len(train_cases)} treino + {len(validation_cases)} validação + {len(test_cases)} teste", "navy"), _card("Treino", f"{len(train_cases)}", "casos que ajustaram os pesos", "teal"), _card("Validação", f"{len(validation_cases)}", "casos para acompanhar seleção", "blue"), _card("Teste interno", f"{len(test_cases)}", "holdout Figshare", "purple"), _card("VitalDB externo", f"{len(external_cases)}", "sem retreino", "orange"), _card("Arquivos no app", f"{len(CASE_PATHS)}", f"{figshare_files} Figshare + {vitaldb_files} VitalDB", "green")]
    return html.Div(
        [
            _tab_intro("DADOS DA REDE", "O que está dentro do checkpoint", "Esta aba torna o modelo auditável: arquitetura, quantidade de parâmetros, pré-processamento, divisão de casos, ambiente e histórico de treinamento ficam visíveis sem precisar abrir o arquivo binário."),
            html.Div([_card("Modelo", "Conv1D", f"{model_name} · regressão contínua de BIS", "navy"), _card("Parâmetros treináveis", f"{param_count:,}", "estado atual do checkpoint", "teal"), _card("Casos de treino", f"{len(train_cases)}", "separação por cirurgia", "blue"), _card("Janelas usadas", f"{int(dataset.get('n_windows', 0)):,}" if isinstance(dataset, dict) else "—", "após filtro de qualidade", "orange")], className="metric-grid"),
            html.Div(coverage_cards, className="metric-grid case-coverage-grid"),
            html.Div(dcc.Graph(figure=_case_count_mae_figure(), config={"displayModeBar": False}), className="chart-card case-count-chart"),
            html.Div("As barras mostram a MAE de cada caso externo; a linha marinho recalcula a MAE agregada quando cada caso entra. Isso descreve a estabilidade da estimativa com mais casos — não é uma curva de retreinamento e não prova que aumentar a amostra, sozinho, melhora o modelo.", className="legend-note"),
            html.Div([html.H3("Casos utilizados por divisão"), _table(["Divisão", "Quantidade", "Identificadores"], split_rows)], className="table-card"),
            html.Div([html.Div([html.H3("Configuração do checkpoint"), _table(["Campo", "Valor", "Interpretação"], config_rows)], className="table-card"), html.Div([html.H3("Treinamento e ambiente"), _table(["Campo", "Valor", "Interpretação"], train_rows)], className="table-card")], className="table-grid two-col"),
            html.Div([html.H3("Arquitetura declarada"), _model_architecture_table()], className="table-card model-architecture"),
            html.Div(dcc.Graph(figure=_history_figure(), config={"displayModeBar": False}), className="chart-card"),
            html.Div([html.H3("Tensores no state_dict"), _model_parameter_table()], className="table-card"),
        ],
        className="tab-panel",
    )


def _build_statistics_tab() -> html.Div:
    metric_rows = []
    for key in ("mae", "rmse", "bias", "pearson_r", "stage_accuracy", "stage_macro_f1"):
        metric_rows.append([METRIC_LABELS[key], _metric_format(key, _metric_value(HOLDOUT_METRICS, key)), _metric_format(key, _metric_value(EXTERNAL_METRICS, key)), "pontos BIS" if key in {"mae", "rmse", "bias"} else "score"])
    return html.Div([_tab_intro("ESTATÍSTICA DO PROJETO", "Todas as métricas, com contexto e incerteza", "Os números abaixo vêm dos relatórios versionados do projeto. As métricas contínuas, os scores de estágio, os intervalos bootstrap e o recorte por caso aparecem juntos para facilitar uma leitura honesta da generalização."), html.Div(_statistics_cards(), className="metric-grid six-metrics"), html.Div([html.Div(dcc.Graph(figure=_error_metric_figure(), config={"displayModeBar": False}), className="chart-card"), html.Div(dcc.Graph(figure=_association_metric_figure(), config={"displayModeBar": False}), className="chart-card")], className="chart-grid two-col"), html.Div(dcc.Graph(figure=_bootstrap_figure(), config={"displayModeBar": False}), className="chart-card"), html.Div([html.H3("Tabela completa de métricas"), _table(["Métrica", "Holdout interno", "VitalDB externo", "Unidade"], metric_rows)], className="table-card"), html.Div([html.Div(dcc.Graph(figure=_per_case_figure(), config={"displayModeBar": False}), className="chart-card"), html.Div([html.Div("COMO INTERPRETAR", className="mini-kicker"), html.H3("Domínio e caso importam"), html.P("MAE e RMSE medem o tamanho do erro em pontos BIS; bias mostra a direção média do desvio."), html.P("Pearson mede associação temporal, não equivalência clínica. Acurácia e Macro-F1 resumem a classificação em estágios."), html.P("Os intervalos bootstrap são por caso: eles mostram a variação entre cirurgias, não uma garantia clínica."), html.Div("O VitalDB externo permanece separado do holdout para não esconder a mudança de domínio.", className="callout")], className="explanation-card")], className="chart-grid two-col lower-evidence")], className="tab-panel")


def _build_method_tab() -> html.Div:
    preprocess = MODEL_METADATA.get("preprocess_config", {})
    dataset = MODEL_METADATA.get("dataset_summary", {})
    split = MODEL_METADATA.get("split", {})
    external_diagnostics = EXTERNAL_REPORT.get("input_diagnostics", [])
    quality_values = [_finite_number(item.get("quality")) for item in external_diagnostics if isinstance(item, dict)]
    quality_mean = float(np.nanmean(quality_values)) if quality_values else float("nan")
    quality_min = float(np.nanmin(quality_values)) if quality_values else float("nan")
    rows = [["Figshare", len(HOLDOUT_REPORT.get("files", [])), HOLDOUT_REPORT.get("n_test_windows", "—"), "holdout por cirurgia"], ["VitalDB", len(EXTERNAL_REPORT.get("files", [])), EXTERNAL_REPORT.get("n_windows", "—"), "validação externa"], ["Treino", len(split.get("train_cases", [])), dataset.get("n_windows", "—") if isinstance(dataset, dict) else "—", "casos não sobrepostos ao teste"], ["Modelo", MODEL_METADATA.get("checkpoint_sha256", "—"), "—", "hash do checkpoint"]]
    preprocess_rows = [[key, value] for key, value in preprocess.items()]
    return html.Div([_tab_intro("MÉTODO E COBERTURA", "De onde vieram os números", "Esta aba documenta a proveniência dos dados, a divisão por cirurgia, o pré-processamento e os limites de qualidade. Ela existe para que cada gráfico possa ser interpretado sem adivinhar o que entrou no cálculo."), html.Div([_card("Arquivos holdout", f"{len(HOLDOUT_REPORT.get('files', []))}", "Figshare", "blue"), _card("Arquivos externos", f"{len(EXTERNAL_REPORT.get('files', []))}", "VitalDB", "orange"), _card("Qualidade VitalDB", _format_number(quality_mean, 3), f"mínimo {_format_number(quality_min, 3)}", "teal"), _card("Offset testado", f"{len(_offset_points())}", "pontos exploratórios", "purple")], className="metric-grid"), html.Div([html.H3("Cobertura dos experimentos"), _table(["Fonte", "Casos/arquivos", "Janelas", "Papel"], rows)], className="table-card"), html.Div([html.Div([html.H3("Pré-processamento"), _table(["Parâmetro", "Valor"], preprocess_rows)], className="table-card"), html.Div([html.H3("Limites de leitura"), html.P("O projeto é research-only e não controla anestésicos."), html.P("A qualidade é um gate diagnóstico de 0 a 1, não um SQI clínico."), html.P("O replay revela somente saídas causais e usa dados offline já auditados."), html.Div("Ao trocar de domínio, leia sempre a aba Estatística junto com a trajetória do caso.", className="callout")], className="explanation-card")], className="table-grid two-col")], className="tab-panel")


def _build_layout() -> html.Div:
    options = [{"label": _case_label(path), "value": str(path)} for path in CASE_PATHS]
    default_value = str(DEFAULT_CASE) if DEFAULT_CASE else None
    duration = load_case(DEFAULT_CASE).duration_seconds if DEFAULT_CASE else 600.0
    model_status = "checkpoint causal carregado em CPU" if MODEL is not None else "checkpoint indisponível · replay bloqueado"
    return html.Div([html.Header([html.Div([html.Div("BRAIN SNIFFER · RESEARCH CONSOLE", className="eyebrow"), html.H1("EEG → CNN → BIS", className="hero-title"), html.P("Uma leitura auditável do sinal, do modelo e do resultado: primeiro o panorama, depois a trajetória, o replay causal, os dados da rede e as estatísticas.", className="hero-subtitle")], className="hero-copy"), html.Div([html.Span("RESEARCH ONLY", className="research-badge"), html.Div(model_status, className="hero-status")], className="hero-side")], className="hero"), html.Div("Uso exclusivamente experimental/educacional. O BIS é referência do monitor e a CNN é uma estimativa de pesquisa; nada nesta tela comanda anestésicos ou substitui avaliação clínica.", className="safety-banner"), html.Main(dcc.Tabs(id="main-tabs", value="overview", parent_className="app-tabs", className="tabs-container", children=[dcc.Tab(label="Visão geral", value="overview", className="app-tab", selected_className="app-tab-selected", children=_build_overview_tab()), dcc.Tab(label="Trajetória completa", value="trajectory", className="app-tab", selected_className="app-tab-selected", children=_build_trajectory_tab(options, default_value)), dcc.Tab(label="Replay causal", value="replay", className="app-tab", selected_className="app-tab-selected", children=_build_replay_tab(options, default_value, duration)), dcc.Tab(label="Dados da rede", value="model", className="app-tab", selected_className="app-tab-selected", children=_build_model_tab()), dcc.Tab(label="Estatística", value="statistics", className="app-tab", selected_className="app-tab-selected", children=_build_statistics_tab()), dcc.Tab(label="Método e cobertura", value="method", className="app-tab", selected_className="app-tab-selected", children=_build_method_tab())]), className="page-content"), html.Footer("BrainSniffer · pipeline auditável · checkpoint congelado · sem uso clínico", className="footer")], className="app-shell")


app = Dash(__name__, title="BrainSniffer · EEG → CNN → BIS", update_title=None)
server = app.server
app.layout = _build_layout()


@server.get("/healthz")
@server.get("/_stcore/health")
def healthz():
    return {"status": "ok"}


@app.callback(Output("case-meta", "children"), Output("replay-time", "max"), Output("replay-time", "value"), Input("case-selector", "value"))
def select_case(path_value: str | None):
    if not path_value:
        return _case_meta(None), 600.0, 0.0
    try:
        duration = load_case(path_value).duration_seconds
    except Exception as error:
        return html.Div(f"Não foi possível abrir o caso: {error}", className="case-meta case-error"), 600.0, 0.0
    return _case_meta(path_value), duration, 0.0


@app.callback(Output("replay-data", "data"), Output("replay-figure", "figure"), Output("quality-figure", "figure"), Input("case-selector", "value"), Input("eeg-window", "value"))
def prepare_replay(path_value: str | None, eeg_window_seconds: int):
    if not path_value:
        message = "Nenhum caso selecionado"
        return None, _empty_figure("Replay sincronizado · atualização local", message, height=650), _empty_figure("Qualidade do sinal · gate de emissão", message, height=280)
    if MODEL is None or PREPROCESS is None:
        message = MODEL_ERROR or "Checkpoint indisponível"
        return None, _empty_figure("Replay sincronizado · atualização local", message, height=650), _empty_figure("Qualidade do sinal · gate de emissão", message, height=280)
    try:
        payload = _replay_payload(path_value)
    except Exception as error:
        message = f"Replay bloqueado: {error}"
        return None, _empty_figure("Replay sincronizado · atualização local", message, height=650), _empty_figure("Qualidade do sinal · gate de emissão", message, height=280)
    return _replay_store_payload(payload), _replay_figure(payload, 0.0, int(eeg_window_seconds or 10)), _quality_figure(payload, 0.0)


@app.callback(Output("play-state", "data"), Output("play-button", "children"), Output("replay-time", "value", allow_duplicate=True), Input("play-button", "n_clicks"), Input("reset-button", "n_clicks"), State("play-state", "data"), prevent_initial_call=True)
def control_replay(play_clicks: int, reset_clicks: int, playing: bool):
    del play_clicks, reset_clicks
    from dash import ctx

    if ctx.triggered_id == "reset-button":
        return False, "▶ Iniciar replay", 0.0
    next_state = not bool(playing)
    return next_state, ("❚❚ Pausar replay" if next_state else "▶ Continuar replay"), no_update


app.clientside_callback(
    """
    function(nIntervals, seekValue, playing, speed, maximum, current) {
        const triggered = window.dash_clientside.callback_context.triggered_id;
        const maxValue = Number(maximum || 0);
        const currentValue = Math.max(0, Math.min(maxValue, Number(current || 0)));
        if (triggered === "replay-time") {
            return Math.max(0, Math.min(maxValue, Number(seekValue || 0)));
        }
        if (triggered !== "replay-interval" || !playing) {
            return window.dash_clientside.no_update;
        }
        if (currentValue >= maxValue) {
            return currentValue;
        }
        const step = Math.max(0.05, 0.12 * Number(speed || 1));
        return Math.min(maxValue, currentValue + step);
    }
    """,
    Output("replay-clock", "data"),
    Input("replay-interval", "n_intervals"),
    Input("replay-time", "value"),
    State("play-state", "data"),
    State("speed", "value"),
    State("replay-time", "max"),
    State("replay-clock", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(clock, payload, baseFigure, baseQualityFigure) {
        const seconds = Math.max(0, Number(clock || 0));
        const formatNumber = (value, digits) => Number.isFinite(value) ? value.toFixed(digits) : "—";
        const formatClock = (value) => {
            const total = Math.max(0, Math.floor(value));
            return String(Math.floor(total / 60)).padStart(2, "0") + ":" + String(total % 60).padStart(2, "0");
        };
        const lastIndexAtOrBefore = (times, value) => {
            let low = 0;
            let high = times.length - 1;
            let answer = -1;
            while (low <= high) {
                const middle = Math.floor((low + high) / 2);
                if (Number(times[middle]) <= value) {
                    answer = middle;
                    low = middle + 1;
                } else {
                    high = middle - 1;
                }
            }
            return answer;
        };
        const reveal = (times, values) => times.map((time, index) => Number(time) <= seconds && values[index] !== null && Number.isFinite(Number(values[index])) ? Number(values[index]) : null);
        const stageLabels = {deep: "Profundo", general: "Anestesia geral", light: "Sedação leve", awake: "Acordado", abstain: "ABSTAIN · sinal insuficiente"};
        const progress = (payload && Number(payload.duration) > 0) ? Math.max(0, Math.min(100, seconds / Number(payload.duration) * 100)) : 0;
        const progressStyle = {width: progress.toFixed(2) + "%"};
        const progressText = progress.toFixed(1).replace(".", ",") + "%";
        if (!payload || !baseFigure || !baseQualityFigure) {
            return [baseFigure, baseQualityFigure, "—", "Aguardando replay", "—", "Aguardando replay", "—", "Disponível após a primeira predição", "—", "Gate padrão 0,20", "Aguardando caso", "00:00", progressStyle, progressText];
        }

        const figure = Object.assign({}, baseFigure);
        figure.data = (baseFigure.data || []).map((trace, index) => {
            if (index === 1) return Object.assign({}, trace, {y: reveal(payload.bis_times, payload.bis_values)});
            if (index === 2) return Object.assign({}, trace, {y: reveal(payload.prediction_times, payload.raw_predictions)});
            if (index === 3) return Object.assign({}, trace, {y: reveal(payload.prediction_times, payload.smoothed_predictions)});
            if (index === 4 || index === 5) return Object.assign({}, trace, {x: [seconds, seconds]});
            return trace;
        });
        figure.layout = Object.assign({}, baseFigure.layout);
        const eegWindow = Number((baseFigure.layout.meta || {}).eegWindow || 10);
        figure.layout.xaxis = Object.assign({}, baseFigure.layout.xaxis, {range: [Math.max(0, seconds - eegWindow), Math.max(5, seconds)]});
        figure.layout.xaxis2 = Object.assign({}, baseFigure.layout.xaxis2, {range: [0, Math.max(10, seconds + 2)]});

        const qualityFigure = Object.assign({}, baseQualityFigure);
        qualityFigure.data = (baseQualityFigure.data || []).map((trace, index) => {
            if (index === 0) return Object.assign({}, trace, {y: reveal(payload.prediction_times, payload.qualities)});
            if (index === 1) return Object.assign({}, trace, {x: [seconds, seconds]});
            return trace;
        });
        qualityFigure.layout = Object.assign({}, baseQualityFigure.layout);

        const predictionIndex = lastIndexAtOrBefore(payload.prediction_times, seconds);
        let cnnValue = "—";
        let cnnDetail = "Aguardando janela causal de 5 s";
        let bisValue = "—";
        let bisDetail = "Último ponto observado";
        let errorValue = "—";
        let errorDetail = "Disponível após a primeira predição";
        let qualityValue = "—";
        let qualityDetail = "Gate padrão 0,20";
        let revealed = 0;
        if (predictionIndex >= 0) {
            revealed = predictionIndex + 1;
            const rawCnn = Number(payload.raw_predictions[predictionIndex]);
            const smoothedCnn = Number(payload.smoothed_predictions[predictionIndex]);
            const predictionTime = Number(payload.prediction_times[predictionIndex]);
            const referenceIndex = lastIndexAtOrBefore(payload.bis_times, predictionTime);
            const reference = referenceIndex >= 0 ? Number(payload.bis_values[referenceIndex]) : NaN;
            const error = Number.isFinite(smoothedCnn) && Number.isFinite(reference) ? smoothedCnn - reference : NaN;
            cnnValue = formatNumber(smoothedCnn, 1);
            cnnDetail = (stageLabels[payload.stages[predictionIndex]] || payload.stages[predictionIndex] || "") + " · t=" + predictionTime.toFixed(1) + "s";
            bisValue = formatNumber(reference, 1);
            bisDetail = Number.isFinite(reference) ? "último ponto observado · t=" + Number(payload.bis_times[referenceIndex]).toFixed(0) + "s" : "sem ponto disponível";
            errorValue = formatNumber(error, 1);
            errorDetail = "CNN bruta " + formatNumber(rawCnn, 1) + " · positivo = acima do BIS";
            qualityValue = formatNumber(Number(payload.qualities[predictionIndex]), 3);
            qualityDetail = "gate de emissão 0,20";
        }
        const status = payload.case_label + " · " + formatClock(seconds) + " · " + revealed + " predição(ões) revelada(s) · atualização local";
        return [figure, qualityFigure, cnnValue, cnnDetail, bisValue, bisDetail, errorValue, errorDetail, qualityValue, qualityDetail, status, formatClock(seconds), progressStyle, progressText];
    }
    """,
    Output("replay-figure", "figure", allow_duplicate=True),
    Output("quality-figure", "figure", allow_duplicate=True),
    Output("replay-cnn-value", "children"),
    Output("replay-cnn-detail", "children"),
    Output("replay-bis-value", "children"),
    Output("replay-bis-detail", "children"),
    Output("replay-error-value", "children"),
    Output("replay-error-detail", "children"),
    Output("replay-quality-value", "children"),
    Output("replay-quality-detail", "children"),
    Output("replay-status", "children"),
    Output("replay-clock-label", "children"),
    Output("replay-progress-fill", "style"),
    Output("replay-progress-text", "children"),
    Input("replay-clock", "data"),
    Input("replay-data", "data"),
    State("replay-figure", "figure"),
    State("quality-figure", "figure"),
    prevent_initial_call=True,
)


@app.callback(Output("trajectory-meta", "children"), Output("trajectory-cards", "children"), Output("trajectory-figure", "figure"), Output("trajectory-error-figure", "figure"), Input("trajectory-case-selector", "value"))
def update_trajectory(path_value: str | None):
    if not path_value:
        message = "Nenhum caso selecionado"
        return _case_meta(None), _initial_trajectory_cards(), _empty_figure("Trajetória completa · BIS contra CNN", message, height=470), _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300)
    if MODEL is None or PREPROCESS is None:
        message = MODEL_ERROR or "Checkpoint indisponível"
        return _case_meta(path_value), _initial_trajectory_cards(), _empty_figure("Trajetória completa · BIS contra CNN", message, height=470), _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300)
    try:
        payload = _replay_payload(path_value)
        trajectory, error_figure = _trajectory_figure(payload)
    except Exception as error:
        message = f"Não foi possível calcular a trajetória: {error}"
        return _case_meta(path_value), _initial_trajectory_cards(), _empty_figure("Trajetória completa · BIS contra CNN", message, height=470), _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300)
    return _case_meta(path_value), _trajectory_cards(payload), trajectory, error_figure


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8501")), debug=False)

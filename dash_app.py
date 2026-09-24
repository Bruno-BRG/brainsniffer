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

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, dcc, html, no_update
from flask import Response, send_file
from plotly.subplots import make_subplots

from brainsniffer.config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase, load_case
from brainsniffer.data.preprocess import StreamingPreprocessor, bis_stage, signal_quality
from brainsniffer.models.cnn import parameter_count
from brainsniffer.pipeline.planning import (
    LEARNING_CURVE_CUTOFF_CASES,
    LEARNING_CURVE_KEY_COUNTS,
    LEARNING_CURVE_MAX_CASES,
    LEARNING_CURVE_STRONG_RETURN_CASES,
)
from brainsniffer.pipeline.planning import (
    theoretical_training_mae as _theoretical_training_mae,
)
from brainsniffer.pipeline.realtime import RealtimePrediction
from brainsniffer.pipeline.training import load_checkpoint

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("BRAINSNIFFER_DATA_DIR", APP_ROOT / "data/raw"))
VITAL_DIR = Path(os.getenv("BRAINSNIFFER_VITAL_DIR", APP_ROOT / "data/vitaldb"))
MODEL_PATH = Path(os.getenv("BRAINSNIFFER_CHECKPOINT", APP_ROOT / "models/brainsniffer_cnn.pt"))
REPORTS_DIR = APP_ROOT / "reports"
FIGURES_DIR = APP_ROOT / "docs/figures"
ARTICLE_FIGURES = (
    ("pipeline.png", "Figura 1 · Fluxo retrospectivo auditável do BrainSniffer"),
    ("bis_trajectory.png", "Figura 2 · Trajetória offline do case19"),
    ("comparison.png", "Figura 3 · Comparação dos checkpoints nos benchmarks"),
    ("pk_prediction.png", "Figura 4 · Probabilidade de predição Pk"),
    ("bootstrap_intervals.png", "Figura 5 · Bootstrap agrupado por caso"),
    ("offset_sensitivity.png", "Figura 6 · Sensibilidade ao offset do rótulo"),
    ("corpus_panels.png", "Figura 7 · Composição e qualidade do corpus"),
    ("training_panels.png", "Figura 8 · Treinamento e projeção por casos"),
)
ARTICLE_FIGURE_FILES = tuple(name for name, _ in ARTICLE_FIGURES)
REPLAY_EEG_POINTS = 9000
REPLAY_INTERVAL_MS = 120

COLORS = {
    "navy": "#102A43",
    "ink": "#172B4D",
    "muted": "#486581",
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

GRAPH_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
}
DEFAULT_CHART_HEIGHT = 330
MIN_CHART_HEIGHT = 280

APP_DESCRIPTION = (
    "Dashboard científico do BrainSniffer para inspeção retrospectiva e replay "
    "offline de estimativas experimentais de índice BIS a partir de EEG."
)

ROBOTS_TEXT = "User-agent: *\nAllow: /\n"
LLMS_TEXT = """# BrainSniffer

BrainSniffer is an open research prototype for retrospective inspection and
laboratory replay of experimental BIS-reference estimates from frontal EEG.
It is not a medical device and must not guide anesthesia or drug dosing.

## Public resources

- [Dashboard](/)
- [Health check](/healthz)
- [Source code and documentation](https://github.com/Bruno-BRG/brainsniffer)
"""

FIGSHARE_HOLDOUT_LABEL = "Figshare · holdout por caso"
ACTIVE_VITALDB_LABEL = "VitalDB · avaliação cruzada de dataset"
ACTIVE_MODEL_LABEL = "Ativo · desenvolvimento Figshare-only"
MIXED_MODEL_LABEL = "Misto · desenvolvimento Figshare + VitalDB"
MIXED_FIGSHARE_BENCHMARK_LABEL = "Figshare · benchmark histórico"
MIXED_VITALDB_HOLDOUT_LABEL = "VitalDB · holdout histórico da mesma fonte"

STAGE_LABELS = {
    "deep": "Faixa estimada abaixo de 40",
    "general": "Faixa estimada de 40 a 59",
    "light": "Faixa estimada de 60 a 79",
    "awake": "Faixa estimada de 80 a 100",
    "abstain": "Sem emissão · sinal insuficiente",
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
CORPUS_MANIFEST = _read_report("corpus_manifest.json")
MIXED_FIGSHARE_REPORT = _read_report("mixed_fixed_figshare_holdout.json")
MIXED_EXTERNAL_REPORT = _read_report("mixed_vitaldb_external.json")
HOLDOUT_METRICS = HOLDOUT_REPORT.get("recomputed_test_metrics", {})
EXTERNAL_METRICS = EXTERNAL_REPORT.get("metrics", {})
PK_FIGSHARE_ACTIVE = _read_report("pk_figshare_active.json")
PK_FIGSHARE_MIXED = _read_report("pk_figshare_mixed.json")
PK_VITALDB_ACTIVE = _read_report("pk_vitaldb_active.json")
PK_VITALDB_MIXED = _read_report("pk_vitaldb_mixed.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model() -> tuple[object | None, PreprocessConfig | None, dict[str, object], str | None]:
    if not MODEL_PATH.exists():
        return None, None, {}, f"Checkpoint não encontrado: {MODEL_PATH}"
    try:
        digest = _sha256(MODEL_PATH)
        model, preprocess, payload = load_checkpoint(MODEL_PATH, device="cpu")
        if _sha256(MODEL_PATH) != digest:
            raise ValueError("Checkpoint alterado durante o carregamento; reinicie")
        payload = dict(payload, checkpoint_sha256=digest)
    except Exception as error:  # pragma: no cover - defensive startup guard
        return None, None, {}, f"Falha ao carregar o checkpoint: {error}"
    return model, preprocess, payload, None


MODEL, PREPROCESS, MODEL_METADATA, MODEL_ERROR = _load_model()
# Model and reports are an immutable startup snapshot. Replacing either requires
# a process restart; case recordings instead participate in the replay cache key.
EFFECTIVE_MIN_QUALITY = DEFAULT_MIN_SIGNAL_QUALITY


def _compatible_report(report: dict[str, object], digest: str | None) -> bool:
    return bool(digest) and report.get("checkpoint_sha256") == digest and report.get("min_quality") == EFFECTIVE_MIN_QUALITY


EVIDENCE_ERRORS = []
for _name in ("HOLDOUT_REPORT", "EXTERNAL_REPORT", "OFFSET_REPORT"):
    if not _compatible_report(globals()[_name], MODEL_METADATA.get("checkpoint_sha256")):
        EVIDENCE_ERRORS.append(f"{_name}: hash/checkpoint ou gate incompatível/ausente; métricas ocultadas")
        globals()[_name] = {}
HOLDOUT_METRICS = HOLDOUT_REPORT.get("recomputed_test_metrics", {})
EXTERNAL_METRICS = EXTERNAL_REPORT.get("metrics", {})


def _artifact_identity(path: Path) -> tuple[object, ...]:
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return (str(path), None)


FROZEN_ARTIFACTS = {
    path: _artifact_identity(path)
    for path in (MODEL_PATH, *(REPORTS_DIR / name for name in (
        "figshare_holdout_evaluation.json", "vitaldb_external_validation.json",
        "offset_sensitivity.json", "corpus_manifest.json",
        "mixed_fixed_figshare_holdout.json", "mixed_vitaldb_external.json",
    )))
}


def _require_frozen_artifacts() -> None:
    if any(_artifact_identity(path) != identity for path, identity in FROZEN_ARTIFACTS.items()):
        raise RuntimeError("Artefatos alterados desde o início; reinicie para carregar modelo e relatórios juntos")


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
    paths = []
    for root, pattern in ((DATA_DIR, "case*.mat"), (VITAL_DIR, "vitaldb_case*.npz")):
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                paths.append(path)
    return paths


def _resolve_case(value: str) -> Path:
    # Exact server-issued values only: never normalize arbitrary browser paths
    # into permission. Recheck containment to catch symlinks changed since startup.
    if not isinstance(value, str) or value not in {str(path) for path in CASE_PATHS}:
        raise ValueError("Caso não permitido")
    path = Path(value)
    root = VITAL_DIR if path.suffix.lower() == ".npz" else DATA_DIR
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Caso fora da raiz permitida")
    return resolved


def _case_label(path: Path) -> str:
    if path.suffix.lower() == ".npz":
        return f"VitalDB · {path.stem.replace('vitaldb_', '')}"
    return f"Figshare · {path.stem}"


def _case_source_label(case: EEGCase) -> str:
    """Return the dataset label without inferring it from a truthy string."""

    source = str(case.source_dataset or "").strip().lower()
    if source == "vitaldb" or case.case_id.lower().startswith("vitaldb_"):
        return f"VitalDB · {case.case_id.removeprefix('vitaldb_')}"
    return f"Figshare · {case.case_id}"


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
    """Vectorize a recorded replay while preserving causal window semantics.

    Recorded files from another dataset may contain missing samples. The dashboard is an
    offline inspection surface, so it interpolates those samples only for the
    causal filter/model input while calculating signal quality on the original
    window. The live estimator remains fail-closed for NaN/Inf input.
    """

    raw_eeg = np.asarray(case.eeg, dtype=np.float32).reshape(-1)
    finite = np.isfinite(raw_eeg)
    if raw_eeg.size and not finite.all():
        if not finite.any():
            raise ValueError("não há amostras EEG finitas para a inspeção offline")
        indices = np.arange(raw_eeg.size)
        eeg = np.interp(indices, indices[finite], raw_eeg[finite]).astype(np.float32)
    else:
        eeg = raw_eeg
    window_samples = config.window_samples
    stride_samples = max(1, int(round(stride_seconds * config.sampling_rate)))
    starts = np.arange(0, max(eeg.size - window_samples + 1, 0), stride_samples, dtype=int)
    if starts.size == 0:
        return []

    processed = StreamingPreprocessor(config).process(eeg)
    raw_windows = np.lib.stride_tricks.sliding_window_view(raw_eeg, window_samples)[starts]
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


def _replay_payload(path_string: str) -> ReplayPayload:
    """Validate even cache hits; recordings are keyed by their actual contents."""
    _require_frozen_artifacts()
    path = _resolve_case(path_string)
    digest = _sha256(path)
    payload = _cached_replay_payload(str(path), digest)
    if _resolve_case(path_string) != path or _sha256(path) != digest:
        raise RuntimeError("Caso alterado durante o replay; tente novamente")
    return payload


@lru_cache(maxsize=16)
def _cached_replay_payload(path_string: str, case_sha256: str) -> ReplayPayload:
    """Run once per case content under the immutable startup model/gate."""

    if MODEL is None or PREPROCESS is None:
        raise RuntimeError(MODEL_ERROR or "Checkpoint indisponível")
    case = load_case(path_string)
    predictions = _fast_replay_case(
        MODEL,
        case,
        PREPROCESS,
        stride_seconds=1.0,
        min_quality=EFFECTIVE_MIN_QUALITY,
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
        role="group",
        **{"aria-label": f"{title}: {value}. {detail}"},
    )


def _live_card(title: str, value_id: str, detail_id: str, tone: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(
                "—",
                id=value_id,
                className="metric-value",
                **{"aria-live": "polite", "aria-atomic": "true"},
            ),
            html.Div("Aguardando replay", id=detail_id, className="metric-detail"),
        ],
        className=f"metric-card metric-{tone}",
        role="group",
        **{"aria-label": title},
    )


def _status_children(title: str, detail: str) -> list[object]:
    return [
        html.Strong(title, className="state-title"),
        html.Span(detail, className="state-detail"),
    ]


def _status_message(
    title: str,
    detail: str,
    *,
    tone: str = "info",
    component_id: str | None = None,
    live: bool = False,
) -> html.Div:
    role = "alert" if tone == "error" else "status"
    attributes: dict[str, object] = {
        "className": f"state-message state-{tone}",
        "role": role,
    }
    if component_id:
        attributes["id"] = component_id
    if live:
        attributes.update({"aria-live": "assertive" if tone == "error" else "polite", "aria-atomic": "true"})
    return html.Div(_status_children(title, detail), **attributes)


def _chart(
    figure: go.Figure,
    caption: str,
    *,
    graph_id: str | None = None,
    class_name: str = "",
) -> html.Figure:
    layout_height = _finite_number(figure.layout.height, DEFAULT_CHART_HEIGHT)
    graph_height = max(MIN_CHART_HEIGHT, int(round(layout_height)))
    graph_kwargs: dict[str, object] = {
        "figure": figure,
        "config": GRAPH_CONFIG,
        "responsive": True,
        "style": {
            "height": f"{graph_height}px",
            "minHeight": f"{MIN_CHART_HEIGHT}px",
            "width": "100%",
        },
    }
    if graph_id:
        graph_kwargs["id"] = graph_id
    classes = " ".join(part for part in ("chart-card", class_name) if part)
    return html.Figure(
        [dcc.Graph(**graph_kwargs), html.Figcaption(caption, className="chart-caption")],
        className=classes,
        **{"aria-label": caption},
    )


def _figure_layout(title: str, *, height: int = 330) -> dict[str, object]:
    return {
        "title": {
            "text": title,
            "font": {"size": 18, "color": COLORS["ink"]},
            "x": 0.02,
            "xanchor": "left",
        },
        "height": height,
        "margin": {"l": 58, "r": 30, "t": 62, "b": 50},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#FAFCFE",
        "font": {"family": "Inter, Arial, sans-serif", "size": 12, "color": COLORS["ink"]},
        "hoverlabel": {"bgcolor": COLORS["navy"], "font": {"color": "white"}},
        "legend": {"orientation": "h", "y": 1.04, "x": 0, "font": {"size": 12}},
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
            x=[FIGSHARE_HOLDOUT_LABEL, ACTIVE_VITALDB_LABEL],
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
            x=[FIGSHARE_HOLDOUT_LABEL, ACTIVE_VITALDB_LABEL],
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
    return {
        "case_id": case.case_id,
        "min_quality": EFFECTIVE_MIN_QUALITY,
        "checkpoint_sha256": MODEL_METADATA.get("checkpoint_sha256"),
        "case_label": _case_source_label(case),
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
    figure.add_hline(y=EFFECTIVE_MIN_QUALITY, line={"color": COLORS["orange"], "width": 1.5, "dash": "dash"}, annotation_text=f"gate {EFFECTIVE_MIN_QUALITY:.2f}", annotation_position="bottom right")
    figure.update_layout(**_figure_layout("Qualidade do sinal · gate de emissão", height=280))
    figure.update_layout(uirevision=f"quality-{payload.case.case_id}")
    figure.update_xaxes(title="Tempo (s)", gridcolor=COLORS["line"])
    figure.update_yaxes(title="Score", range=[0, 1.05], gridcolor=COLORS["line"])
    return figure


def _case_meta(path_value: str | None) -> html.Div:
    if not path_value:
        return html.Div("Nenhum caso disponível.", className="case-meta")
    try:
        case = load_case(_resolve_case(path_value))
    except Exception as error:
        return html.Div(f"Não foi possível abrir o caso: {error}", className="case-meta case-error")
    nonfinite_count = int((~np.isfinite(case.eeg)).sum())
    detail = f" · {_format_clock(case.duration_seconds)} · {case.sampling_rate} Hz · {case.bis.size:,} pontos BIS"
    if nonfinite_count:
        detail += f" · {nonfinite_count:,} EEG não finitas · interpolação somente offline"
    return html.Div(
        [
            html.Strong(_case_label(Path(path_value))),
            html.Span(detail),
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
            go.Scattergl(x=payload.prediction_times, y=_json_series(error), connectgaps=False, mode="lines", name="Erro causal original", visible="legendonly", line={"color": COLORS["orange"], "width": 1.2, "dash": "dot"}, hovertemplate="t=%{x:.1f}s<br>erro causal=%{y:.1f} pontos BIS<extra></extra>"),
            go.Scattergl(x=payload.prediction_times, y=_json_series(visual_error), connectgaps=False, mode="lines", name="Erro · EMA visual (5 s)", line={"color": COLORS["orange"], "width": 2.4}, hovertemplate="t=%{x:.1f}s<br>erro visual=%{y:.1f} pontos BIS<extra></extra>"),
        ]
    )
    error_figure.add_hline(y=0, line={"color": COLORS["navy"], "width": 1.5, "dash": "dash"})
    error_figure.update_layout(**_figure_layout(f"Erro ao longo do caso · {case.case_id} · CNN − BIS", height=300), uirevision=f"trajectory-error-{case.case_id}")
    error_figure.update_xaxes(title="Tempo do caso (s)", gridcolor=COLORS["line"])
    error_figure.update_yaxes(title="Erro (pontos BIS)", gridcolor=COLORS["line"])
    return figure, error_figure


def _table(headers: list[str], rows: list[list[object]], class_name: str = "data-table") -> html.Table:
    body = [html.Tr([html.Td(str(value)) for value in row]) for row in rows]
    if not body:
        body = [
            html.Tr(
                html.Td(
                    "Nenhum dado disponível para esta tabela.",
                    colSpan=len(headers),
                    className="empty-table-cell",
                )
            )
        ]
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(header, scope="col") for header in headers])),
            html.Tbody(body),
        ],
        className=class_name,
    )


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


def _training_case_anchor() -> tuple[int, float] | None:
    split = MODEL_METADATA.get("split", {})
    train_cases = split.get("train_cases", []) if isinstance(split, dict) else []
    anchor_cases = len(train_cases) if isinstance(train_cases, (list, tuple)) else 0
    anchor_mae = _metric_value(HOLDOUT_METRICS, "mae")
    if anchor_cases <= 0 or not np.isfinite(anchor_mae) or anchor_mae <= 0:
        return None
    return anchor_cases, float(anchor_mae)


def _training_case_learning_curve_figure() -> go.Figure:
    anchor = _training_case_anchor()
    if anchor is None:
        return _empty_figure("Curva de aprendizagem por casos de treino", "Ponto medido do checkpoint indisponível", height=500)
    anchor_cases, anchor_mae = anchor
    curve_counts = np.unique(np.concatenate(([float(anchor_cases)], np.geomspace(anchor_cases, LEARNING_CURVE_MAX_CASES, 240))))
    curve_mae = _theoretical_training_mae(curve_counts, anchor_cases=anchor_cases, anchor_mae=anchor_mae)
    key_counts = np.asarray([count for count in LEARNING_CURVE_KEY_COUNTS if count >= anchor_cases], dtype=float)
    key_mae = _theoretical_training_mae(key_counts, anchor_cases=anchor_cases, anchor_mae=anchor_mae)
    curve_positions = np.log10(curve_counts)
    key_positions = np.log10(key_counts)
    anchor_position = float(np.log10(anchor_cases))
    reductions = (anchor_mae - curve_mae) / anchor_mae * 100.0
    key_reductions = (anchor_mae - key_mae) / anchor_mae * 100.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=curve_positions,
            y=curve_mae,
            mode="lines",
            name="Projeção teórica",
            line={"color": COLORS["teal"], "width": 3.2},
            customdata=np.column_stack((curve_counts, reductions)),
            hovertemplate="%{customdata[0]:.0f} casos de treino<br>MAE teórica: %{y:.2f} pontos BIS<br>redução vs. hoje: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=key_positions,
            y=key_mae,
            mode="markers",
            name="Pontos da projeção",
            marker={"color": COLORS["teal"], "size": 7, "line": {"color": "white", "width": 1}},
            customdata=np.column_stack((key_counts, key_reductions)),
            hovertemplate="%{customdata[0]:.0f} casos de treino<br>MAE projetada: %{y:.2f} pontos BIS<br>redução vs. hoje: %{customdata[1]:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[anchor_position],
            y=[anchor_mae],
            mode="markers",
            name="Medido · checkpoint atual",
            marker={"color": COLORS["navy"], "size": 14, "symbol": "diamond", "line": {"color": "white", "width": 2}},
            hovertemplate=f"{anchor_cases} casos de treino<br>MAE medida no holdout interno: {anchor_mae:.2f} pontos BIS<extra></extra>",
        )
    )
    figure.add_vrect(x0=np.log10(LEARNING_CURVE_STRONG_RETURN_CASES), x1=np.log10(LEARNING_CURVE_CUTOFF_CASES), fillcolor=COLORS["orange"], opacity=0.10, line_width=0, annotation_text="zona de retorno decrescente", annotation_position="top left", annotation_font={"color": COLORS["orange"], "size": 11})
    figure.add_vline(x=np.log10(LEARNING_CURVE_STRONG_RETURN_CASES), line={"color": COLORS["orange"], "width": 1, "dash": "dot"})
    figure.add_vline(x=np.log10(LEARNING_CURVE_CUTOFF_CASES), line={"color": COLORS["orange"], "width": 1.5, "dash": "dash"}, annotation_text="corte sugerido", annotation_position="bottom right", annotation_font={"color": COLORS["orange"], "size": 11})
    figure.update_layout(**_figure_layout("Curva de aprendizagem: quantos casos justificam o treino?", height=500))
    figure.update_xaxes(title="Casos usados para ajustar os pesos · distância logarítmica", tickmode="array", tickvals=np.log10(LEARNING_CURVE_KEY_COUNTS).tolist(), ticktext=[str(count) for count in LEARNING_CURVE_KEY_COUNTS], gridcolor=COLORS["line"])
    figure.update_yaxes(title="MAE estimada (pontos BIS · menor é melhor)", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _training_curve_summary_cards() -> list[html.Div]:
    anchor = _training_case_anchor()
    if anchor is None:
        return [_card("Ponto medido", "—", "holdout interno indisponível", "navy")]
    anchor_cases, anchor_mae = anchor
    cutoff_mae = float(_theoretical_training_mae([LEARNING_CURVE_CUTOFF_CASES], anchor_cases=anchor_cases, anchor_mae=anchor_mae)[0])
    tail_mae = float(_theoretical_training_mae([LEARNING_CURVE_MAX_CASES], anchor_cases=anchor_cases, anchor_mae=anchor_mae)[0])
    cutoff_to_tail = (cutoff_mae - tail_mae) / cutoff_mae * 100.0
    return [
        _card("Ponto medido", f"{anchor_cases} casos", f"MAE holdout = {anchor_mae:.2f} pontos BIS", "navy"),
        _card("Ganho forte projetado", "até 80", "−15,0% relativo contra o ponto atual", "teal"),
        _card("Corte sugerido", "100 casos", "a partir daqui o retorno entra em platô", "orange"),
        _card("100 → 1.000", f"−{cutoff_to_tail:.1f}%", f"só {cutoff_mae - tail_mae:.2f} ponto BIS teórico", "purple"),
    ]


def _training_curve_rows() -> list[list[str]]:
    anchor = _training_case_anchor()
    if anchor is None:
        return [["—", "—", "—", "Ponto medido do checkpoint indisponível"]]
    anchor_cases, anchor_mae = anchor
    counts = [count for count in LEARNING_CURVE_KEY_COUNTS if count >= anchor_cases]
    values = _theoretical_training_mae(counts, anchor_cases=anchor_cases, anchor_mae=anchor_mae)
    rows = []
    for count, value in zip(counts, values, strict=False):
        reduction = (anchor_mae - float(value)) / anchor_mae * 100.0
        if count == anchor_cases:
            reading = "observado · holdout interno"
        elif count == LEARNING_CURVE_STRONG_RETURN_CASES:
            reading = "hipótese · fim do ganho forte"
        elif count == LEARNING_CURVE_CUTOFF_CASES:
            reading = "hipótese · ponto de corte sugerido"
        elif count == LEARNING_CURVE_MAX_CASES:
            reading = "hipótese · platô de planejamento"
        else:
            reading = "hipótese · interpolação logarítmica"
        rows.append([f"{count:,}", _format_number(float(value), 2), f"{reduction:.1f}%", reading])
    return rows


def _corpus_records() -> list[dict[str, object]]:
    records = CORPUS_MANIFEST.get("cases", [])
    return [record for record in records if isinstance(record, dict)]


def _corpus_summary_cards() -> list[html.Div]:
    summary = CORPUS_MANIFEST.get("summary", {})
    if not isinstance(summary, dict) or not summary:
        return [_card("Manifesto do corpus", "—", "relatório ainda não gerado", "navy")]
    source_summary = summary.get("source_summary", {})
    source_count = len(source_summary) if isinstance(source_summary, dict) else 0
    return [
        _card("Casos elegíveis", f"{int(summary.get('eligible_training_cases', 0))}", "pool de treino supervisionado", "teal"),
        _card("Janelas elegíveis", f"{int(summary.get('eligible_training_windows', 0)):,}", "após o gate por qualidade", "blue"),
        _card("Quarentena", f"{int(summary.get('quarantined_development_cases', 0))}", "não entram sem revisão", "red"),
        _card("Benchmark VitalDB", f"{int(summary.get('frozen_external_cases', 0))}", "15 casos fora do ajuste · histórico reutilizado", "orange"),
        _card("Fontes supervisionadas", f"{source_count}", "Figshare + VitalDB quando elegível", "purple"),
    ]


def _corpus_source_figure() -> go.Figure:
    summary = CORPUS_MANIFEST.get("summary", {})
    source_summary = summary.get("source_summary", {}) if isinstance(summary, dict) else {}
    if not isinstance(source_summary, dict) or not source_summary:
        return _empty_figure("Composição do corpus", "Gere reports/corpus_manifest.json para visualizar", height=360)
    sources = sorted(source_summary)
    labels = {"figshare": "Figshare", "vitaldb": "VitalDB"}
    figure = go.Figure()
    figure.add_trace(go.Bar(
        name="Elegíveis para treino",
        x=[labels.get(source, source) for source in sources],
        y=[int(source_summary[source].get("eligible_cases", 0)) for source in sources],
        marker_color=COLORS["teal"],
        hovertemplate="%{x}<br>elegíveis: %{y}<extra></extra>",
    ))
    figure.add_trace(go.Bar(
        name="Quarentena",
        x=[labels.get(source, source) for source in sources],
        y=[int(source_summary[source].get("quarantined_cases", 0)) for source in sources],
        marker_color=COLORS["red"],
        hovertemplate="%{x}<br>quarentena: %{y}<extra></extra>",
    ))
    figure.add_trace(go.Bar(
        name="Benchmark histórico",
        x=[labels.get(source, source) for source in sources],
        y=[int(source_summary[source].get("frozen_external_cases", 0)) for source in sources],
        marker_color=COLORS["orange"],
        hovertemplate="%{x}<br>casos no benchmark histórico: %{y}<extra></extra>",
    ))
    figure.update_layout(**_figure_layout("Composição do corpus por fonte", height=360), barmode="stack")
    figure.update_xaxes(title="Fonte", gridcolor=COLORS["line"])
    figure.update_yaxes(title="Quantidade de casos", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _corpus_quality_figure() -> go.Figure:
    records = [record for record in _corpus_records() if isinstance(record.get("signal"), dict)]
    if not records:
        return _empty_figure("Mapa de qualidade dos casos", "Manifesto de qualidade indisponível", height=480)
    gate = CORPUS_MANIFEST.get("quality_config", {})
    min_finite = _finite_number(gate.get("min_finite_fraction"), 0.9) if isinstance(gate, dict) else 0.9
    colors = {"include": COLORS["teal"], "quarantine": COLORS["red"], "exclude": COLORS["muted"]}
    symbols = {"include": "circle", "quarantine": "x", "exclude": "diamond-open"}
    figure = go.Figure()
    for status in ("include", "quarantine", "exclude"):
        selected = [record for record in records if record.get("quality_status") == status]
        if not selected:
            continue
        x = [_finite_number(record.get("signal", {}).get("finite_fraction")) * 100 for record in selected]
        y = [_finite_number(record.get("windows", {}).get("accepted_fraction")) * 100 for record in selected]
        customdata = [
            [
                record.get("case_id", "—"),
                record.get("source_key", "—"),
                _finite_number(record.get("signal", {}).get("max_nonfinite_gap_seconds")),
                _finite_number(record.get("signal", {}).get("quality_global")),
                ", ".join(str(item) for item in record.get("exclusion_reasons", [])) or "sem motivo",
            ]
            for record in selected
        ]
        figure.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            name={"include": "Elegível", "quarantine": "Quarentena", "exclude": "Excluído"}[status],
            text=[str(record.get("case_id", "—")).replace("vitaldb_", "") for record in selected],
            textposition="top center",
            marker={
                "color": colors[status],
                "size": 11,
                "symbol": symbols[status],
                "line": {"color": "white", "width": 1},
            },
            customdata=customdata,
            hovertemplate=(
                "caso %{customdata[0]} · %{customdata[1]}<br>"
                "finitude %{x:.2f}%<br>janelas aceitas %{y:.2f}%<br>"
                "maior lacuna %{customdata[2]:.2f}s<br>qualidade global %{customdata[3]:.3f}<br>"
                "decisão: %{customdata[4]}<extra></extra>"
            ),
        ))
    figure.add_vline(x=min_finite * 100, line={"color": COLORS["orange"], "dash": "dash"}, annotation_text="gate de finitude", annotation_position="bottom right")
    figure.update_layout(**_figure_layout("Mapa de qualidade: finitude × janelas aproveitáveis", height=480))
    figure.update_xaxes(title="Amostras EEG finitas (%)", range=[0, 100.5], gridcolor=COLORS["line"])
    figure.update_yaxes(title="Janelas com BIS válido e qualidade suficiente (%)", range=[0, 100.5], gridcolor=COLORS["line"])
    return figure


def _corpus_case_rows() -> list[list[str]]:
    rows = []
    for record in _corpus_records():
        signal = record.get("signal", {})
        windows = record.get("windows", {})
        if not isinstance(signal, dict) or not isinstance(windows, dict):
            rows.append([str(record.get("file_name", "—")), "—", "—", "—", "erro de leitura"])
            continue
        source = "VitalDB" if record.get("source_key") == "vitaldb" else "Figshare"
        role = {"development_pool": "pool de desenvolvimento", "frozen_external": "benchmark histórico"}.get(str(record.get("role")), str(record.get("role", "—")))
        status = {"include": "elegível", "quarantine": "quarentena", "exclude": "excluído"}.get(str(record.get("quality_status")), "—")
        rows.append([
            f"{source} · {record.get('case_id', '—')}",
            role,
            status,
            f"{_finite_number(signal.get('finite_fraction')) * 100:.1f}%",
            f"{int(windows.get('accepted_windows', 0)):,}",
        ])
    return rows


def _mixed_metrics(report: dict[str, object]) -> dict[str, object]:
    metrics = report.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(candidate):
        return float("nan")
    return (baseline - candidate) / abs(baseline) * 100.0


def _corpus_experiment_cards() -> list[html.Div]:
    mixed_internal = _mixed_metrics(MIXED_FIGSHARE_REPORT)
    mixed_external = _mixed_metrics(MIXED_EXTERNAL_REPORT)
    base_internal = _metric_value(HOLDOUT_METRICS, "mae")
    base_external = _metric_value(EXTERNAL_METRICS, "mae")
    mixed_internal_mae = _metric_value(mixed_internal, "mae")
    mixed_external_mae = _metric_value(mixed_external, "mae")
    internal_gain = _relative_improvement(base_internal, mixed_internal_mae)
    external_gain = _relative_improvement(base_external, mixed_external_mae)
    if not np.isfinite(internal_gain) or not np.isfinite(external_gain):
        return [_card("Candidato misto", "—", "comparação histórica ainda não disponível", "navy")]
    return [
        _card("Figshare · MAE", f"{mixed_internal_mae:.2f}", f"misto · variação exploratória {internal_gain:+.1f}% vs ativo", "teal"),
        _card("VitalDB · MAE", f"{mixed_external_mae:.2f}", f"15 casos · holdout por participante · {external_gain:+.1f}% vs ativo", "orange"),
        _card("VitalDB · Pearson", f"{_metric_value(mixed_external, 'pearson_r'):.3f}", "mesma fonte após 10 casos VitalDB no desenvolvimento", "purple"),
        _card("Leitura", "exploratória", "benchmark histórico reutilizado · não confirmatória", "navy"),
    ]


def _corpus_experiment_figure() -> go.Figure:
    mixed_internal = _mixed_metrics(MIXED_FIGSHARE_REPORT)
    mixed_external = _mixed_metrics(MIXED_EXTERNAL_REPORT)
    baseline_mae = [_metric_value(HOLDOUT_METRICS, "mae"), _metric_value(EXTERNAL_METRICS, "mae")]
    candidate_mae = [_metric_value(mixed_internal, "mae"), _metric_value(mixed_external, "mae")]
    baseline_pearson = [_metric_value(HOLDOUT_METRICS, "pearson_r"), _metric_value(EXTERNAL_METRICS, "pearson_r")]
    candidate_pearson = [_metric_value(mixed_internal, "pearson_r"), _metric_value(mixed_external, "pearson_r")]
    if not np.isfinite(np.asarray(candidate_mae)).any():
        return _empty_figure("Efeito observado do corpus misto", "Relatórios do candidato ainda não disponíveis", height=390)
    labels = [MIXED_FIGSHARE_BENCHMARK_LABEL, MIXED_VITALDB_HOLDOUT_LABEL]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("MAE · menor é melhor", "Pearson r · maior é melhor"), horizontal_spacing=0.14)
    for column, baseline, candidate, title, color in ((1, baseline_mae, candidate_mae, "MAE", COLORS["teal"]), (2, baseline_pearson, candidate_pearson, "Pearson r", COLORS["purple"])):
        figure.add_trace(go.Bar(name=ACTIVE_MODEL_LABEL, x=labels, y=baseline, marker_color=COLORS["navy"], legendgroup="baseline", showlegend=column == 1, hovertemplate="%{x}<br>ativo Figshare-only: %{y:.3f}<extra></extra>"), row=1, col=column)
        figure.add_trace(go.Bar(name=MIXED_MODEL_LABEL, x=labels, y=candidate, marker_color=color, legendgroup="candidate", showlegend=column == 1, hovertemplate="%{x}<br>candidato após exposição VitalDB: %{y:.3f}<extra></extra>"), row=1, col=column)
        figure.update_yaxes(title=title, gridcolor=COLORS["line"], row=1, col=column)
    figure.update_layout(**_figure_layout("Comparação histórica: checkpoint ativo × candidato misto", height=390), barmode="group")
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=1)
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=2)
    return figure


def _corpus_experiment_rows() -> list[list[str]]:
    mixed_internal = _mixed_metrics(MIXED_FIGSHARE_REPORT)
    mixed_external = _mixed_metrics(MIXED_EXTERNAL_REPORT)
    rows = []
    for label, baseline, candidate in (
        (MIXED_FIGSHARE_BENCHMARK_LABEL, HOLDOUT_METRICS, mixed_internal),
        (MIXED_VITALDB_HOLDOUT_LABEL, EXTERNAL_METRICS, mixed_external),
    ):
        base_mae = _metric_value(baseline, "mae")
        candidate_mae = _metric_value(candidate, "mae")
        gain = _relative_improvement(base_mae, candidate_mae)
        rows.append([
            label,
            _format_number(base_mae, 2),
            _format_number(candidate_mae, 2),
            f"{gain:+.1f}%" if np.isfinite(gain) else "—",
            _format_number(_metric_value(candidate, "pearson_r"), 3),
        ])
    return rows


def _build_corpus_tab() -> html.Div:
    third_source = CORPUS_MANIFEST.get("third_source", {})
    third_source_note = (
        "Os rótulos MOAA/S e estado de consciência não são uma referência BIS contínua; "
        "essa fonte exige uma tarefa ordinal ou multitarefa separada."
        if isinstance(third_source, dict) and third_source
        else "A fonte exige uma tarefa separada por usar um alvo incompatível."
    )
    return html.Div(
        [
            _tab_intro(
                "CORPUS AUDITÁVEL",
                "Mais dados, com controle de qualidade e papéis explícitos",
                "O manifesto separa o pool de desenvolvimento Figshare + VitalDB, a quarentena por sinal e um benchmark histórico de 15 casos VitalDB. Esse benchmark é avaliação cruzada de dataset para o ativo Figshare-only, mas holdout por participante da mesma fonte para o candidato misto.",
            ),
            html.Div(_corpus_summary_cards(), className="metric-grid five-metrics"),
            html.Div(
                [
                    _chart(
                        _corpus_source_figure(),
                        "Contagem de casos por fonte e decisão de governança. As barras descrevem disponibilidade; não demonstram desempenho do modelo.",
                    ),
                    _chart(
                        _corpus_quality_figure(),
                        "Cada símbolo representa um caso. Cor e forma repetem a decisão de elegibilidade para que a leitura não dependa apenas de cor.",
                    ),
                ],
                className="chart-grid two-col",
            ),
            html.Div(
                [
                    html.H3("Decisão de incorporação"),
                    html.P("O candidato misto usa amostragem balanceada por grupo e por fonte: uma cirurgia longa não domina milhares de janelas, e o VitalDB não domina o Figshare apenas por ter gravações maiores."),
                    html.P("O desenvolvimento do candidato já incluiu 10 casos VitalDB. Os outros 15 casos VitalDB ficaram fora do ajuste e formam um holdout por participante da mesma fonte/domínio; por isso, não são uma avaliação externa de domínio para esse candidato."),
                    html.P("Para o checkpoint ativo, desenvolvido somente em Figshare, esses 15 casos continuam externos ao desenvolvimento e sustentam uma avaliação cruzada de dataset exploratória."),
                    html.P(f"Terceira fonte: DOSE-I não foi misturada. {third_source_note}"),
                    html.Div("A comparação é retrospectiva e exploratória: o benchmark VitalDB já havia sido inspecionado antes do candidato misto e foi reutilizado. Ela não é confirmatória; esse papel exige uma nova coorte pré-especificada e não usada no desenvolvimento.", className="callout"),
                ],
                className="explanation-card",
            ),
            html.Div(_corpus_experiment_cards(), className="metric-grid four-metrics"),
            _chart(
                _corpus_experiment_figure(),
                "Comparação histórica nos mesmos casos. No ativo Figshare-only, VitalDB é avaliação cruzada de dataset; no candidato, é holdout por participante da mesma fonte após 10 casos VitalDB no desenvolvimento. Trata-se de benchmark histórico reutilizado, portanto o resultado é exploratório e não confirmatório.",
            ),
            html.Div([html.H3("Resultado exploratório do candidato misto"), _table(["Benchmark histórico", "Ativo Figshare-only · MAE", "Misto Figshare + VitalDB · MAE", "Variação exploratória", "Misto · Pearson"], _corpus_experiment_rows())], className="table-card"),
            html.Div([html.H3("Manifesto caso a caso"), _table(["Caso", "Papel no manifesto", "Decisão", "EEG finito", "Janelas aceitas"], _corpus_case_rows())], className="table-card"),
        ],
        className="tab-panel",
    )


def _statistics_cards() -> list[html.Div]:
    return [
        _card("Janelas Figshare", _format_number(_metric_value(HOLDOUT_METRICS, "n"), 0), "5 casos/cirurgias · holdout do ativo", "blue"),
        _card("Janelas VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "n"), 0), "15 casos · cruzada de dataset do ativo", "orange"),
        _card("MAE Figshare", _format_number(_metric_value(HOLDOUT_METRICS, "mae"), 1), "ativo Figshare-only · holdout por caso", "teal"),
        _card("MAE VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "mae"), 1), "ativo Figshare-only · cruzada exploratória", "red"),
        _card("Pearson Figshare", _format_number(_metric_value(HOLDOUT_METRICS, "pearson_r"), 3), "associação temporal no holdout", "purple"),
        _card("Pearson VitalDB", _format_number(_metric_value(EXTERNAL_METRICS, "pearson_r"), 3), "sensibilidade à mudança de dataset", "navy"),
    ]


def _error_metric_figure() -> go.Figure:
    available = [
        _metric_value(metrics, key)
        for metrics in (HOLDOUT_METRICS, EXTERNAL_METRICS)
        for key in ("mae", "rmse")
    ]
    if not np.isfinite(available).any():
        return _empty_figure(
            "Erros contínuos por conjunto",
            "Relatórios de erro indisponíveis",
            height=360,
        )
    figure = go.Figure()
    for key, color in (("mae", COLORS["blue"]), ("rmse", COLORS["orange"])):
        figure.add_trace(go.Bar(name=METRIC_LABELS[key], x=[FIGSHARE_HOLDOUT_LABEL, ACTIVE_VITALDB_LABEL], y=[_metric_value(HOLDOUT_METRICS, key), _metric_value(EXTERNAL_METRICS, key)], marker_color=color, text=[_metric_format(key, _metric_value(HOLDOUT_METRICS, key)), _metric_format(key, _metric_value(EXTERNAL_METRICS, key))], textposition="outside", hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f} pontos BIS<extra></extra>"))
    figure.update_layout(**_figure_layout("Erros contínuos por conjunto", height=360), barmode="group")
    figure.update_yaxes(title="Pontos BIS", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _association_metric_figure() -> go.Figure:
    available = [
        _metric_value(metrics, key)
        for metrics in (HOLDOUT_METRICS, EXTERNAL_METRICS)
        for key in ("pearson_r", "stage_accuracy", "stage_macro_f1")
    ]
    if not np.isfinite(available).any():
        return _empty_figure(
            "Associação e classificação por conjunto",
            "Relatórios de associação indisponíveis",
            height=390,
        )
    figure = go.Figure()
    for dataset, metrics, color in ((FIGSHARE_HOLDOUT_LABEL, HOLDOUT_METRICS, COLORS["teal"]), (ACTIVE_VITALDB_LABEL, EXTERNAL_METRICS, COLORS["orange"])):
        keys = ["pearson_r", "stage_accuracy", "stage_macro_f1"]
        values = [_metric_value(metrics, key) for key in keys]
        figure.add_trace(go.Bar(name=dataset, x=[METRIC_LABELS[key] for key in keys], y=values, marker_color=color, text=[_metric_format(key, value) for key, value in zip(keys, values, strict=False)], textposition="outside", hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>"))
    figure.update_layout(**_figure_layout("Associação e classificação por conjunto", height=390), barmode="group")
    figure.update_yaxes(title="Score (0–1)", range=[0, 1.08], gridcolor=COLORS["line"])
    return figure


def _bootstrap_figure() -> go.Figure:
    available = []
    for report in (HOLDOUT_REPORT, EXTERNAL_REPORT):
        bootstrap = report.get("case_bootstrap", {})
        if not isinstance(bootstrap, dict):
            continue
        for key in ("mae", "pearson_r"):
            interval = bootstrap.get(key, {})
            if isinstance(interval, dict):
                available.append(_finite_number(interval.get("mean")))
    if not np.isfinite(available).any():
        return _empty_figure(
            "Incerteza entre casos · bootstrap",
            "Intervalos bootstrap indisponíveis",
            height=340,
        )
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
        figure.add_trace(go.Scatter(x=[FIGSHARE_HOLDOUT_LABEL, ACTIVE_VITALDB_LABEL], y=means, mode="markers", name=title, marker={"color": color, "size": 12}, error_y={"type": "data", "symmetric": False, "array": [hi - mean for hi, mean in zip(upper, means, strict=False)], "arrayminus": [mean - lo for mean, lo in zip(means, lower, strict=False)]}, hovertemplate="%{x}<br>média %{y:.3f}<extra></extra>", showlegend=False), row=1, col=column)
        figure.update_yaxes(title=title, gridcolor=COLORS["line"], row=1, col=column)
    figure.update_layout(**_figure_layout("Incerteza entre casos · bootstrap", height=340))
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=1)
    figure.update_xaxes(gridcolor=COLORS["line"], row=1, col=2)
    return figure


def _per_case_figure() -> go.Figure:
    values = EXTERNAL_REPORT.get("per_case", [])
    if not isinstance(values, list) or not values:
        return _empty_figure("Ativo Figshare-only · VitalDB por caso", "Métricas por caso indisponíveis", height=430)
    rows = [item for item in values if isinstance(item, dict)]
    rows.sort(key=lambda item: _finite_number(item.get("mae")), reverse=True)
    labels = [str(item.get("case_id", "caso")) for item in rows]
    maes = [_finite_number(item.get("mae")) for item in rows]
    pearsons = [_finite_number(item.get("pearson_r")) for item in rows]
    figure = go.Figure(go.Bar(x=maes, y=labels, orientation="h", marker_color=COLORS["orange"], text=[_format_number(value, 1) for value in maes], textposition="outside", customdata=np.asarray(pearsons)[:, None], hovertemplate="%{y}<br>MAE %{x:.2f}<br>Pearson %{customdata[0]:.3f}<extra></extra>"))
    figure.update_layout(**_figure_layout("Ativo Figshare-only · avaliação VitalDB por caso", height=580))
    figure.update_xaxes(title="MAE (pontos BIS)", rangemode="tozero", gridcolor=COLORS["line"])
    figure.update_yaxes(title="Caso", autorange="reversed", gridcolor=COLORS["line"])
    return figure


def _tab_intro(kicker: str, title: str, lead: str) -> html.Div:
    return html.Div([html.Div(kicker, className="section-kicker"), html.H2(title, className="section-title"), html.P(lead, className="section-lead")], className="section-intro")


def _pk_report_pair(dataset: str) -> tuple[dict[str, object], dict[str, object]]:
    if dataset == "figshare":
        return PK_FIGSHARE_ACTIVE, PK_FIGSHARE_MIXED
    return PK_VITALDB_ACTIVE, PK_VITALDB_MIXED


def _pk_point(report: dict[str, object]) -> tuple[float, float, float]:
    interval = report.get("pk_bootstrap", {})
    point = _finite_number(report.get("pk"))
    low = _finite_number(interval.get("lower_95")) if isinstance(interval, dict) else float("nan")
    high = _finite_number(interval.get("upper_95")) if isinstance(interval, dict) else float("nan")
    return point, low, high


def _pk_figure() -> go.Figure:
    panels = ("Figshare · benchmark histórico", "VitalDB · holdout histórico")
    figure = go.Figure()
    plotted = False
    for model, (label, color) in enumerate(((ACTIVE_MODEL_LABEL, COLORS["navy"]), (MIXED_MODEL_LABEL, COLORS["teal"]))):
        labels: list[str] = []
        values: list[float] = []
        upper: list[float] = []
        lower: list[float] = []
        for panel_label, key in zip(panels, ("figshare", "vitaldb"), strict=True):
            active, mixed = _pk_report_pair(key)
            point, low, high = _pk_point((active, mixed)[model])
            if not np.isfinite(point):
                continue
            labels.append(panel_label)
            values.append(point)
            upper.append(max(high - point, 0.0) if np.isfinite(high) else 0.0)
            lower.append(max(point - low, 0.0) if np.isfinite(low) else 0.0)
        if not values:
            continue
        plotted = True
        figure.add_trace(
            go.Bar(
                name=label,
                x=values,
                y=labels,
                orientation="h",
                marker_color=color,
                error_x={"type": "data", "array": upper, "arrayminus": lower},
                hovertemplate="%{y}<br>" + label + ": %{x:.3f}<extra></extra>",
            )
        )
    if not plotted:
        return _empty_figure("Pk · probabilidade de predição", "Relatórios de Pk indisponíveis", height=380)
    figure.add_vline(x=0.5, line={"color": COLORS["muted"], "width": 1, "dash": "dot"}, annotation_text="chance 0,5", annotation_position="top")
    figure.update_layout(**_figure_layout("Pk · probabilidade de predição (Smith et al. 1996)", height=380), barmode="group")
    figure.update_xaxes(title="Pk (1 = ordem perfeita · 0,5 = chance)", range=[0.4, 1.0], gridcolor=COLORS["line"])
    figure.update_yaxes(title="Benchmark", gridcolor=COLORS["line"])
    return figure


def _pk_table_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    entries = (
        ("Figshare · ativo", "figshare_active"),
        ("Figshare · misto", "figshare_mixed"),
        ("VitalDB · ativo", "vitaldb_active"),
        ("VitalDB · misto", "vitaldb_mixed"),
    )
    for dataset_label, key in entries:
        report = {
            "figshare_active": PK_FIGSHARE_ACTIVE,
            "figshare_mixed": PK_FIGSHARE_MIXED,
            "vitaldb_active": PK_VITALDB_ACTIVE,
            "vitaldb_mixed": PK_VITALDB_MIXED,
        }[key]
        point, low, high = _pk_point(report)
        if not np.isfinite(point):
            continue
        report_metrics = report.get("metrics", {})
        metrics_map = report_metrics if isinstance(report_metrics, dict) else {}
        windows = report.get("n_windows", "—")
        rows.append([
            dataset_label,
            _format_number(point, 3),
            f"{_format_number(low, 3)} a {_format_number(high, 3)}",
            _format_number(_finite_number(metrics_map.get("mae")), 2),
            _format_number(_finite_number(metrics_map.get("pearson_r")), 3),
            f"{windows:,}" if isinstance(windows, int) else str(windows),
        ])
    return rows


def _evidence_status() -> html.Div:
    missing = []
    if not HOLDOUT_METRICS:
        missing.append("holdout Figshare por caso")
    if not EXTERNAL_METRICS:
        missing.append("avaliação cruzada VitalDB do ativo")
    if not _offset_points():
        missing.append("sensibilidade temporal")
    if missing:
        return _status_message(
            "Evidência parcial",
            "Sem dados para: " + ", ".join(missing) + ". Os gráficos ausentes são sinalizados na própria tela.",
            tone="warning",
        )
    return _status_message(
        "Relatórios carregados",
        "Holdout Figshare por caso, avaliação cruzada VitalDB do ativo Figshare-only e sensibilidade temporal disponíveis para comparação experimental.",
        tone="ready",
    )


def _initial_case_status(component_id: str, purpose: str) -> html.Div:
    if MODEL is None or PREPROCESS is None:
        return _status_message(
            "Análise indisponível",
            MODEL_ERROR or "O checkpoint não pôde ser carregado.",
            tone="error",
            component_id=component_id,
            live=True,
        )
    if not CASE_PATHS:
        return _status_message(
            "Nenhuma gravação disponível",
            "Adicione um caso de pesquisa válido para habilitar esta visualização.",
            tone="warning",
            component_id=component_id,
            live=True,
        )
    return _status_message(
        "Caso pronto para análise",
        purpose,
        tone="ready",
        component_id=component_id,
        live=True,
    )


def _build_overview_tab() -> html.Div:
    return html.Div(
        [
            _tab_intro(
                "PAINEL DE EVIDÊNCIA",
                "Checkpoint ativo: comparação entre datasets",
                "Esta aba mostra somente o checkpoint ativo, desenvolvido em Figshare. Ela separa o holdout Figshare por caso da avaliação cruzada VitalDB, externa ao desenvolvimento desse modelo. "
                "Erro, associação e sensibilidade temporal aparecem em blocos distintos para evitar "
                "que uma única métrica seja tratada como conclusão clínica.",
            ),
            _evidence_status(),
            html.Div(
                [
                    _card(
                        "Holdout · MAE",
                        _format_number(_metric_value(HOLDOUT_METRICS, "mae"), 1),
                        "Figshare · 5 casos · menor é melhor",
                        "blue",
                    ),
                    _card(
                        "Holdout · Pearson",
                        _format_number(_metric_value(HOLDOUT_METRICS, "pearson_r"), 3),
                        "associação temporal · não equivalência",
                        "teal",
                    ),
                    _card(
                        "VitalDB · MAE",
                        _format_number(_metric_value(EXTERNAL_METRICS, "mae"), 1),
                        "15 casos · cruzada de dataset · menor é melhor",
                        "orange",
                    ),
                    _card(
                        "VitalDB · Pearson",
                        _format_number(_metric_value(EXTERNAL_METRICS, "pearson_r"), 3),
                        "ativo Figshare-only · associação exploratória",
                        "red",
                    ),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    _chart(
                        _evidence_mae_figure(),
                        "Checkpoint ativo Figshare-only. Erro absoluto médio em pontos BIS no holdout Figshare por caso e na avaliação cruzada VitalDB; essas métricas não medem segurança clínica.",
                    ),
                    _chart(
                        _evidence_pearson_figure(),
                        "Checkpoint ativo Figshare-only. Associação temporal no holdout Figshare e na avaliação cruzada VitalDB. Correlação não significa concordância, transporte causal nem utilidade clínica.",
                    ),
                ],
                className="chart-grid two-col",
            ),
            html.Div(
                [
                    _chart(
                        _evidence_offset_figure(),
                        "Análise pós-hoc do deslocamento temporal do rótulo. Os pontos são exploratórios e não escolhem um offset para uso em pessoas.",
                    ),
                    html.Div(
                        [
                            html.Div("LEITURA PARA A DECISÃO", className="mini-kicker"),
                            html.H3("Por que manter Figshare e VitalDB separados?"),
                            html.Ol(
                                [
                                    html.Li("O checkpoint ativo foi desenvolvido somente em Figshare, com separação por caso/cirurgia."),
                                    html.Li("Para esse ativo, VitalDB é externo ao desenvolvimento e permite uma avaliação cruzada de dataset sem retreino."),
                                    html.Li("A diferença entre as fontes indica sensibilidade à mudança de dataset; não identifica sozinha a causa nem prova transporte a outro domínio."),
                                ],
                                className="explanation-list",
                            ),
                            html.Div(
                                "Os nove offsets calculados permanecem visíveis como análise exploratória; nenhum deles constitui recomendação de monitorização.",
                                className="callout",
                            ),
                        ],
                        className="explanation-card",
                    ),
                ],
                className="chart-grid two-col lower-evidence",
            ),
        ],
        className="tab-panel",
    )


def _build_trajectory_tab(options: list[dict[str, object]], default_value: str | None) -> html.Div:
    trajectory_results = html.Div(
        [
            html.Div(id="trajectory-cards", children=_initial_trajectory_cards(), className="metric-grid trajectory-metrics"),
            _chart(
                _empty_figure(
                    "Trajetória completa · BIS contra CNN",
                    "Preparando o caso selecionado",
                    height=470,
                ),
                "Série retrospectiva completa. A linha visual suavizada facilita a leitura, enquanto as métricas usam a saída causal original.",
                graph_id="trajectory-figure",
            ),
            _chart(
                _empty_figure(
                    "Erro ao longo do caso · CNN − BIS",
                    "Preparando o caso selecionado",
                    height=300,
                ),
                "Diferença em pontos BIS ao longo do tempo. Valores positivos significam estimativa acima da referência do arquivo.",
                graph_id="trajectory-error-figure",
                class_name="trajectory-error-chart",
            ),
        ],
        className="trajectory-results",
    )
    return html.Div(
        [
            _tab_intro(
                "VISÃO RETROSPECTIVA",
                "O caso inteiro de uma vez",
                "A curva completa permite inspecionar, sem esperar o replay, como a estimativa experimental "
                "e o BIS registrado variaram durante o caso selecionado.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Caso para analisar", htmlFor="trajectory-case-selector"),
                            dcc.Dropdown(
                                id="trajectory-case-selector",
                                options=options,
                                value=default_value,
                                clearable=False,
                                searchable=True,
                                disabled=not options,
                                placeholder="Nenhum caso disponível",
                            ),
                            html.Div(id="trajectory-meta", children=_case_meta(default_value)),
                        ],
                        className="control-block case-control",
                    ),
                    html.Div(
                        [
                            html.Div("LEITURA CORRETA", className="mini-kicker"),
                            html.P("Ao trocar o caso, um estado de carregamento identifica o cálculo em andamento. Cards e títulos só mudam quando o novo caso termina de atualizar."),
                            html.P("A interpolação de amostras ausentes ocorre somente nesta inspeção offline. O score de qualidade continua usando o sinal original e não é um SQI clínico."),
                        ],
                        className="explanation-card compact-explanation",
                    ),
                ],
                className="control-grid trajectory-controls",
            ),
            _initial_case_status(
                "trajectory-data-status",
                "A trajetória do caso selecionado será calculada em modo retrospectivo e experimental.",
            ),
            dcc.Loading(
                id="trajectory-loading",
                children=trajectory_results,
                custom_spinner=html.Div(
                    [html.Strong("Calculando trajetória"), html.Span("Processando janelas causais do caso selecionado.")],
                    className="loading-state",
                    role="status",
                    **{"aria-live": "polite"},
                ),
                delay_show=250,
                delay_hide=150,
                target_components={"trajectory-figure": "figure"},
            ),
            html.Div(
                "A linha marinho é o BIS registrado; a linha turquesa espessa é uma EMA causal de 5 s usada somente para leitura. A saída causal original e o erro original continuam disponíveis na legenda; cards e métricas não usam essa suavização visual.",
                className="legend-note",
            ),
        ],
        className="tab-panel",
    )


def _build_replay_tab(options: list[dict[str, object]], default_value: str | None, duration: float) -> html.Div:
    marks = {0: "0:00", int(min(duration, 60)): "1:00"}
    if duration > 300:
        marks[300] = "5:00"
    marks[int(duration)] = _format_clock(duration)
    replay_available = bool(options) and MODEL is not None and PREPROCESS is not None
    replay_results = html.Div(
        [
            dcc.Store(id="replay-data", data=None),
            _initial_case_status(
                "replay-data-status",
                "Dados causais preparados; use os controles para revelar a gravação no tempo.",
            ),
            html.Div(
                [
                    _live_card("Estimativa CNN suavizada", "replay-cnn-value", "replay-cnn-detail", "teal"),
                    _live_card("BIS registrado", "replay-bis-value", "replay-bis-detail", "navy"),
                    _live_card("Diferença CNN − BIS", "replay-error-value", "replay-error-detail", "orange"),
                    _live_card("Qualidade técnica", "replay-quality-value", "replay-quality-detail", "green"),
                ],
                className="metric-grid replay-metrics",
            ),
            _chart(
                _empty_figure(
                    "Replay sincronizado · atualização local",
                    "Escolha um caso e inicie o replay",
                    height=650,
                ),
                "O painel superior mostra o EEG recebido; o inferior compara BIS registrado e estimativas da CNN. Esta é uma simulação offline, não monitorização em tempo real.",
                graph_id="replay-figure",
                class_name="replay-main-chart",
            ),
            _chart(
                _empty_figure(
                    "Qualidade do sinal · gate de emissão",
                    "Aguardando o replay",
                    height=280,
                ),
                "Heurística técnica de 0 a 1 usada para bloquear emissões abaixo do gate. Não é um índice clínico de qualidade do sinal.",
                graph_id="quality-figure",
            ),
            html.Div(
                [
                    html.Strong("Como ler a tela: "),
                    "azul mostra o EEG recebido; marinho em degraus mostra o BIS registrado; turquesa mostra a estimativa suavizada; roxo pontilhado mostra a saída bruta. A sequência foi calculada causalmente e é apenas revelada pelo relógio local.",
                ],
                className="legend-note",
            ),
        ],
        className="replay-results",
    )
    return html.Div(
        [
            _tab_intro(
                "SIMULAÇÃO OFFLINE",
                "Replay causal de uma gravação: EEG entrando, CNN respondendo",
                "O replay revela a estimativa somente depois que a janela EEG anterior foi recebida. "
                "A inferência é calculada uma vez e o relógio apenas controla a visualização local; "
                "a tela não representa um dispositivo ou fluxo clínico ao vivo.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Caso para reproduzir", htmlFor="case-selector"),
                            dcc.Dropdown(
                                id="case-selector",
                                options=options,
                                value=default_value,
                                clearable=False,
                                searchable=True,
                                disabled=not options,
                                placeholder="Nenhum caso disponível",
                            ),
                            html.Div(id="case-meta", children=_case_meta(default_value)),
                        ],
                        className="control-block case-control",
                    ),
                    html.Div(
                        [
                            html.Label("Velocidade da simulação", htmlFor="speed"),
                            dcc.Slider(
                                id="speed",
                                min=0.25,
                                max=4,
                                step=0.25,
                                value=1,
                                marks={0.25: "0,25×", 1: "1×", 2: "2×", 4: "4×"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Div("Altera apenas a velocidade do relógio visual.", className="control-help"),
                        ],
                        className="control-block speed-control",
                    ),
                    html.Div(
                        [
                            html.Label("Janela EEG exibida", htmlFor="eeg-window"),
                            dcc.Dropdown(
                                id="eeg-window",
                                options=[{"label": f"{value}s", "value": value} for value in (5, 10, 20, 30)],
                                value=10,
                                clearable=False,
                                disabled=not replay_available,
                            ),
                            html.Div("Controla somente o recorte visível do EEG.", className="control-help"),
                        ],
                        className="control-block window-control",
                    ),
                ],
                className="control-grid",
            ),
            html.Div(
                [
                    html.Button(
                        "Iniciar replay",
                        id="play-button",
                        n_clicks=0,
                        className="button-primary",
                        disabled=not replay_available,
                        title="Iniciar ou pausar a simulação offline",
                        **{"aria-pressed": "false"},
                    ),
                    html.Button(
                        "Reiniciar",
                        id="reset-button",
                        n_clicks=0,
                        className="button-secondary",
                        disabled=not replay_available,
                        title="Voltar o relógio da simulação para zero",
                    ),
                    html.Div("A primeira estimativa aparece após a janela causal de 5 s.", className="replay-hint"),
                    html.Div(
                        [
                            html.Span("Tempo ", className="clock-label-prefix"),
                            html.Span("00:00", id="replay-clock-label"),
                        ],
                        className="replay-clock",
                        **{"aria-label": "Tempo atual do replay"},
                    ),
                ],
                className="replay-actions",
            ),
            dcc.Slider(
                id="replay-time",
                min=0,
                max=duration,
                step=0.5,
                value=0,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
                className="time-slider",
                disabled=not replay_available,
            ),
            html.Div(
                [html.Div(id="replay-progress-fill", className="replay-progress-fill")],
                id="replay-progress-track",
                className="replay-progress-track",
                role="progressbar",
                **{
                    "aria-label": "Progresso da simulação",
                    "aria-valuemin": "0",
                    "aria-valuemax": "100",
                    "aria-valuenow": "0",
                },
            ),
            html.Div(
                ["Progresso da simulação: ", html.Span("0,0%", id="replay-progress-text")],
                className="replay-progress-label",
            ),
            dcc.Interval(id="replay-interval", interval=REPLAY_INTERVAL_MS, n_intervals=0),
            dcc.Store(id="play-state", data=False),
            dcc.Store(id="replay-clock", data=0.0),
            html.Div(
                id="replay-status",
                className="replay-status",
                role="status",
                **{"aria-live": "polite", "aria-atomic": "true"},
            ),
            dcc.Loading(
                id="replay-loading",
                children=replay_results,
                custom_spinner=html.Div(
                    [html.Strong("Preparando replay"), html.Span("Calculando as janelas causais do caso selecionado.")],
                    className="loading-state",
                    role="status",
                    **{"aria-live": "polite"},
                ),
                delay_show=250,
                delay_hide=150,
                target_components={"replay-data": "data"},
            ),
        ],
        className="tab-panel",
    )


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
    config_rows = [["Checkpoint", model_name, str(MODEL_PATH.name)], ["Parâmetros treináveis", f"{param_count:,}", "modelo carregado em CPU" if MODEL is not None else "indisponível"], ["Entrada", " × ".join(str(value) for value in window_shape), "canal × amostras"], ["Amostragem", f"{preprocess.get('sampling_rate', '—')} Hz", "taxa esperada pelo modelo"], ["Janela", f"{preprocess.get('window_seconds', '—')} s", "contexto temporal causal"], ["Filtro", f"{preprocess.get('lowcut_hz', '—')}–{preprocess.get('highcut_hz', '—')} Hz", "band-pass"], ["Escala", f"±{preprocess.get('clip_uv', '—')} µV / {preprocess.get('amplitude_scale_uv', '—')}", "clip e normalização"], ["Gate de qualidade", str(EFFECTIVE_MIN_QUALITY), "abaixo disso emite abstain"]]
    train_rows = [["Épocas", training.get("epochs", "—"), "treinamento"], ["Batch", training.get("batch_size", "—"), "janelas por atualização"], ["Learning rate", training.get("learning_rate", "—"), "AdamW"], ["Weight decay", training.get("weight_decay", "—"), "regularização"], ["Seed", training.get("seed", "—"), "reprodutibilidade"], ["Divisão", f"{len(train_cases)}/{len(validation_cases)}/{len(test_cases)}", "train / validação / teste"], ["Ambiente", environment.get("torch", "—"), f"Python {environment.get('python', '—')}"]]
    split_rows = [["Treino", len(train_cases), ", ".join(train_cases)], ["Validação", len(validation_cases), ", ".join(validation_cases)], ["Teste interno", len(test_cases), ", ".join(test_cases)], ["Avaliação cruzada VitalDB", len(external_cases), ", ".join(external_cases)]]
    coverage_cards = [_card("Casos no checkpoint", f"{checkpoint_cases}", f"{len(train_cases)} treino + {len(validation_cases)} validação + {len(test_cases)} teste", "navy"), _card("Treino", f"{len(train_cases)}", "casos que ajustaram os pesos", "teal"), _card("Validação", f"{len(validation_cases)}", "casos para acompanhar seleção", "blue"), _card("Teste interno", f"{len(test_cases)}", "holdout Figshare por caso", "purple"), _card("Cruzada VitalDB", f"{len(external_cases)}", "externa ao desenvolvimento Figshare-only", "orange"), _card("Arquivos no app", f"{len(CASE_PATHS)}", f"{figshare_files} Figshare + {vitaldb_files} VitalDB", "green")]
    return html.Div(
        [
            _tab_intro("DADOS DA REDE", "O que está dentro do checkpoint", "Esta aba torna o modelo auditável: arquitetura, quantidade de parâmetros, pré-processamento, divisão de casos, ambiente e histórico de treinamento ficam visíveis sem precisar abrir o arquivo binário."),
            html.Div([_card("Modelo", "Conv1D", f"{model_name} · regressão contínua de BIS", "navy"), _card("Parâmetros treináveis", f"{param_count:,}", "estado atual do checkpoint", "teal"), _card("Casos de treino", f"{len(train_cases)}", "separação por cirurgia", "blue"), _card("Janelas usadas", f"{int(dataset.get('n_windows', 0)):,}" if isinstance(dataset, dict) else "—", "após filtro de qualidade", "orange")], className="metric-grid"),
            html.Div(coverage_cards, className="metric-grid case-coverage-grid"),
            html.Div(_training_curve_summary_cards(), className="metric-grid learning-curve-metrics"),
            _chart(
                _training_case_learning_curve_figure(),
                "O losango é a única medição observada; os demais pontos são projeções de planejamento, não resultados experimentais nem promessa de desempenho.",
                class_name="case-count-chart",
            ),
            html.Div([html.Strong("Hipótese de planejamento, não resultado: "), "há apenas uma medição, com ", html.Strong(f"{len(train_cases)} casos"), ". A curva supõe redução relativa do erro de 15% até 80 casos, 18% até 100 e 1,5% adicional até 1.000. Esses valores e o corte em 100 são escolhas ilustrativas, não limiares científicos nem justificativa para novo treino."], className="legend-note learning-curve-note"),
            html.Div([html.H3("Pontos de planejamento da curva"), _table(["Casos de treino", "MAE", "Redução vs. atual", "Leitura"], _training_curve_rows())], className="table-card learning-curve-table"),
            html.Div([html.H3("Casos utilizados por divisão"), _table(["Divisão", "Quantidade", "Identificadores"], split_rows)], className="table-card"),
            html.Div([html.Div([html.H3("Configuração do checkpoint"), _table(["Campo", "Valor", "Interpretação"], config_rows)], className="table-card"), html.Div([html.H3("Treinamento e ambiente"), _table(["Campo", "Valor", "Interpretação"], train_rows)], className="table-card")], className="table-grid two-col"),
            html.Div([html.H3("Arquitetura declarada"), _model_architecture_table()], className="table-card model-architecture"),
            _chart(
                _history_figure(),
                "Histórico registrado durante o treinamento do checkpoint. Perdas e métricas de validação descrevem esse experimento e não desempenho clínico.",
            ),
            html.Details([html.Summary("Detalhes técnicos: tensores do modelo"), _model_parameter_table()], className="table-card"),
        ],
        className="tab-panel",
    )


def _build_statistics_tab() -> html.Div:
    metric_rows = []
    for key in ("mae", "rmse", "bias", "pearson_r", "stage_accuracy", "stage_macro_f1"):
        metric_rows.append([METRIC_LABELS[key], _metric_format(key, _metric_value(HOLDOUT_METRICS, key)), _metric_format(key, _metric_value(EXTERNAL_METRICS, key)), "pontos BIS" if key in {"mae", "rmse", "bias"} else "score"])
    return html.Div(
        [
            _tab_intro(
                "ESTATÍSTICA DO PROJETO",
                "Checkpoint ativo Figshare-only, com contexto e incerteza",
                "Os números desta aba pertencem ao checkpoint ativo desenvolvido somente em Figshare. As métricas separam o holdout Figshare por caso da avaliação cruzada VitalDB, externa ao desenvolvimento desse modelo; o candidato misto é tratado separadamente na aba Corpus.",
            ),
            _evidence_status(),
            html.Div(_statistics_cards(), className="metric-grid six-metrics"),
            html.Div(
                [
                    _chart(
                        _error_metric_figure(),
                        "Checkpoint ativo Figshare-only. MAE e RMSE são erros em pontos BIS no holdout Figshare e na avaliação cruzada VitalDB; valores menores não demonstram transporte nem segurança clínica.",
                    ),
                    _chart(
                        _association_metric_figure(),
                        "Checkpoint ativo Figshare-only. Pearson resume associação temporal; acurácia e Macro-F1 resumem faixas discretizadas. Nenhuma dessas métricas demonstra equivalência clínica.",
                    ),
                ],
                className="chart-grid two-col",
            ),
            _chart(
                _bootstrap_figure(),
                "Checkpoint ativo Figshare-only. Intervalos de 95% por reamostragem de casos, ainda ponderados pelo número de janelas; expressam variação amostral exploratória, não incerteza clínica individual.",
            ),
            _chart(
                _pk_figure(),
                "Pk (probabilidade de predição, Smith, Dutton e Smith, Anesthesiology 1996, 84:38-51) nos mesmos benchmarks históricos do artigo (Figshare 5 casos/5.523 janelas; VitalDB 15/38.730). Pk é a associação ordinal reescalada, invariante a escala: 1 é ordem perfeita e 0,5 é chance. As barras são IC 95% por reamostragem de casos. Pk compara a ordem contra o BIS de referência do monitor, não contra um desfecho de resposta ao estímulo.",
            ),
            html.Div(
                [
                    html.H3("Probabilidade de predição Pk"),
                    _table(
                        ["Conjunto", "Pk", "IC 95% por caso", "MAE", "Pearson r", "Janelas"],
                        _pk_table_rows(),
                    ),
                ],
                className="table-card",
            ),
            html.Div(
                [
                    html.H3("Tabela completa de métricas"),
                    _table(
                        ["Métrica", "Figshare · holdout por caso", "VitalDB · cruzada do ativo", "Unidade"],
                        metric_rows,
                    ),
                ],
                className="table-card",
            ),
            html.Div(
                [
                    _chart(
                        _per_case_figure(),
                        "Checkpoint ativo Figshare-only na avaliação cruzada VitalDB. A distribuição por caso expõe heterogeneidade que a média agregada ocultaria.",
                    ),
                    html.Div(
                        [
                            html.Div("COMO INTERPRETAR", className="mini-kicker"),
                            html.H3("Domínio e caso importam"),
                            html.P("MAE e RMSE medem o tamanho do erro em pontos BIS; bias mostra a direção média do desvio."),
                            html.P("Pearson mede associação temporal, não equivalência clínica. Acurácia e Macro-F1 resumem a classificação em faixas."),
                            html.P("Os intervalos bootstrap são por caso: eles mostram a variação entre cirurgias, não uma garantia clínica."),
                            html.Div("Aqui, VitalDB é externo ao desenvolvimento apenas porque estas métricas são do ativo Figshare-only. Para o candidato misto, os mesmos 15 casos são holdout por participante da mesma fonte após exposição a 10 casos VitalDB, em benchmark histórico reutilizado e não confirmatório.", className="callout"),
                        ],
                        className="explanation-card",
                    ),
                ],
                className="chart-grid two-col lower-evidence",
            ),
        ],
        className="tab-panel",
    )


def _article_figures_section() -> html.Div:
    """Every figure of the article, so the site and the text cannot diverge."""

    available = [
        (name, caption) for name, caption in ARTICLE_FIGURES if (FIGURES_DIR / name).exists()
    ]
    if not available:
        return _status_message(
            "Figuras do artigo indisponíveis",
            "Nenhum PNG encontrado em docs/figures; rode docs/generate_figures.py.",
            tone="warning",
        )
    return html.Div(
        [
            html.H3("Figuras do artigo"),
            html.P(
                "As mesmas figuras do TCC, na numeração do texto. Todo painel do "
                "console corresponde a uma delas e toda figura do artigo aparece aqui."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=f"/figures/{name}",
                                alt=caption,
                                className="article-figure",
                            ),
                            html.Div(caption, className="figure-caption"),
                        ],
                        className="figure-card",
                    )
                    for name, caption in available
                ],
                className="figure-grid",
            ),
        ],
        className="table-card",
    )


def _build_method_tab() -> html.Div:
    preprocess = MODEL_METADATA.get("preprocess_config", {})
    dataset = MODEL_METADATA.get("dataset_summary", {})
    split = MODEL_METADATA.get("split", {})
    external_diagnostics = EXTERNAL_REPORT.get("input_diagnostics", [])
    quality_values = [_finite_number(item.get("quality")) for item in external_diagnostics if isinstance(item, dict)]
    quality_mean = float(np.nanmean(quality_values)) if quality_values else float("nan")
    quality_min = float(np.nanmin(quality_values)) if quality_values else float("nan")
    rows = [["Figshare", len(HOLDOUT_REPORT.get("files", [])), HOLDOUT_REPORT.get("n_test_windows", "—"), "holdout por caso/cirurgia do ativo"], ["VitalDB", len(EXTERNAL_REPORT.get("files", [])), EXTERNAL_REPORT.get("n_windows", "—"), "avaliação cruzada · externa ao desenvolvimento do ativo"], ["Treino", len(split.get("train_cases", [])), dataset.get("n_windows", "—") if isinstance(dataset, dict) else "—", "casos não sobrepostos ao teste"], ["Modelo", MODEL_METADATA.get("checkpoint_sha256", "—"), "—", "hash do checkpoint"]]
    preprocess_rows = [[key, value] for key, value in preprocess.items()]
    return html.Div([_tab_intro("MÉTODO E COBERTURA", "De onde vieram os números", "Esta aba documenta a proveniência dos dados, a divisão por caso/cirurgia, o pré-processamento e os limites de qualidade do checkpoint ativo Figshare-only. No Figshare, não há identificador que permita afirmar separação por paciente."), html.Div([_card("Arquivos holdout", f"{len(HOLDOUT_REPORT.get('files', []))}", "Figshare · por caso/cirurgia", "blue"), _card("Arquivos VitalDB", f"{len(EXTERNAL_REPORT.get('files', []))}", "cruzada de dataset do ativo", "orange"), _card("Qualidade VitalDB", _format_number(quality_mean, 3), f"mínimo {_format_number(quality_min, 3)}", "teal"), _card("Offset testado", f"{len(_offset_points())}", "pontos exploratórios", "purple")], className="metric-grid"), html.Div([html.H3("Cobertura dos experimentos"), _table(["Fonte", "Casos/arquivos", "Janelas", "Papel"], rows)], className="table-card"), html.Div([html.Div([html.H3("Pré-processamento"), _table(["Parâmetro", "Valor"], preprocess_rows)], className="table-card"), html.Div([html.H3("Limites de leitura"), html.P("O projeto é research-only e não controla anestésicos."), html.P("A qualidade é um gate diagnóstico de 0 a 1, não um SQI clínico."), html.P("O replay revela somente saídas causais e usa dados offline já auditados."), html.Div("A avaliação cruzada VitalDB desta aba pertence ao ativo Figshare-only; a comparação do candidato misto tem outra interpretação e aparece na aba Corpus.", className="callout")], className="explanation-card")], className="table-grid two-col"), _article_figures_section()], className="tab-panel")


def _default_case_state() -> tuple[float, str | None]:
    if DEFAULT_CASE is None:
        return 600.0, "Nenhum caso disponível"
    try:
        case = load_case(_resolve_case(str(DEFAULT_CASE)))
        if not np.isfinite(case.duration_seconds) or case.duration_seconds <= 0 or not np.isfinite(case.eeg).any():
            raise ValueError("Gravação vazia ou sem EEG finito")
        if PREPROCESS is not None and (
            case.eeg.size < PREPROCESS.window_samples
            or case.sampling_rate != PREPROCESS.sampling_rate
        ):
            raise ValueError("Caso sem janela completa ou amostragem incompatível com o checkpoint")
        return case.duration_seconds, None
    except Exception as error:
        return 600.0, f"Caso padrão indisponível: {error}"


def _build_layout() -> html.Div:
    options = [{"label": _case_label(path), "value": str(path)} for path in CASE_PATHS]
    default_value = str(DEFAULT_CASE) if DEFAULT_CASE else None
    duration, case_error = _default_case_state()
    model_status = "checkpoint causal carregado em CPU" if MODEL is not None else "checkpoint indisponível · replay bloqueado"
    return html.Div(
        [
            html.A("Pular para o conteúdo", href="#main-content", className="skip-link"),
            html.Header(
                [
                    html.Div(
                        [
                            html.Div("BRAIN SNIFFER · PROTÓTIPO DE PESQUISA", className="eyebrow"),
                            html.H1("EEG → CNN → BIS", className="hero-title"),
                            html.P(
                                "Problema: até que ponto o EEG permite estimar o BIS registrado? Escolha um caso em Trajetória, compare EEG/BIS no Replay e consulte os resultados e limitações. Modelo e Corpus oferecem detalhes de apoio.",
                                className="hero-subtitle",
                            ),
                        ],
                        className="hero-copy",
                    ),
                    html.Div(
                        [
                            html.Span("SOMENTE PESQUISA", className="research-badge"),
                            html.Div(model_status, className="hero-status"),
                        ],
                        className="hero-side",
                    ),
                ],
                className="hero",
            ),
            html.Div(
                "Uso exclusivamente experimental/educacional. O BIS é referência do monitor e a CNN é uma estimativa de pesquisa; esta tela não comanda anestésicos e não substitui avaliação clínica.",
                className="safety-banner",
                role="note",
                **{"aria-label": "Aviso de uso experimental"},
            ),
            _status_message(
                "Proveniência e prontidão",
                " · ".join(filter(None, [MODEL_ERROR, case_error, *EVIDENCE_ERRORS]))
                or "Modelo e relatórios compatíveis. Artefatos congelados até reinício; gate efetivo " + str(EFFECTIVE_MIN_QUALITY),
                tone="warning" if MODEL_ERROR or case_error or EVIDENCE_ERRORS else "info",
            ),
            html.Main(
                dcc.Tabs(
                    id="main-tabs",
                    value="overview",
                    mobile_breakpoint=0,
                    parent_className="app-tabs",
                    className="tabs-container",
                    children=[
                        dcc.Tab(
                            label="Visão geral",
                            value="overview",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_overview_tab(),
                        ),
                        dcc.Tab(
                            label="Trajetória completa",
                            value="trajectory",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_trajectory_tab(options, default_value),
                        ),
                        dcc.Tab(
                            label="Replay causal",
                            value="replay",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_replay_tab(options, default_value, duration),
                        ),
                        dcc.Tab(
                            label="Modelo (apoio)",
                            value="model",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_model_tab(),
                        ),
                        dcc.Tab(
                            label="Corpus",
                            value="corpus",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_corpus_tab(),
                        ),
                        dcc.Tab(
                            label="Resultados",
                            value="statistics",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_statistics_tab(),
                        ),
                        dcc.Tab(
                            label="Método e limitações",
                            value="method",
                            className="app-tab",
                            selected_className="app-tab-selected",
                            children=_build_method_tab(),
                        ),
                    ],
                ),
                id="main-content",
                className="page-content",
            ),
            html.Footer(
                "BrainSniffer · pipeline auditável · checkpoint congelado · sem uso clínico",
                className="footer",
            ),
        ],
        className="app-shell",
    )


app = Dash(
    __name__,
    title="BrainSniffer · EEG → CNN → BIS",
    update_title=None,
    meta_tags=[{"name": "description", "content": APP_DESCRIPTION}],
)
app.index_string = app.index_string.replace("<html>", '<html lang="pt-BR">', 1)
server = app.server
app.layout = _build_layout()


@server.get("/healthz")
@server.get("/_stcore/health")
def healthz():
    errors = list(EVIDENCE_ERRORS)
    if MODEL is None or PREPROCESS is None:
        errors.append(MODEL_ERROR or "Checkpoint indisponível")
    try:
        _require_frozen_artifacts()
    except RuntimeError as error:
        errors.append(str(error))
    _, case_error = _default_case_state()
    if case_error:
        errors.append(case_error)
    return {"status": "not_ready" if errors else "ok", "errors": errors}, 503 if errors else 200


@server.get("/robots.txt")
def robots_txt() -> Response:
    return Response(ROBOTS_TEXT, content_type="text/plain; charset=utf-8")


@server.get("/llms.txt")
def llms_txt() -> Response:
    return Response(LLMS_TEXT, content_type="text/markdown; charset=utf-8")


@server.get("/figures/<name>")
def article_figure(name: str) -> Response:
    """Serve the article figures so the site shows the same panels as the text."""

    if name not in ARTICLE_FIGURE_FILES:
        return Response("Figura não autorizada\n", status=404, content_type="text/plain; charset=utf-8")
    path = FIGURES_DIR / name
    if not path.exists():
        return Response("Figura indisponível\n", status=404, content_type="text/plain; charset=utf-8")
    return send_file(path, mimetype="image/png")


@app.callback(Output("case-meta", "children"), Output("replay-time", "max"), Output("replay-time", "value"), Input("case-selector", "value"))
def select_case(path_value: str | None):
    if not path_value:
        return _case_meta(None), 600.0, 0.0
    try:
        duration = load_case(_resolve_case(path_value)).duration_seconds
    except Exception as error:
        return (
            html.Div(
                f"Não foi possível abrir o caso: {error}",
                className="case-meta case-error",
                role="alert",
            ),
            600.0,
            0.0,
        )
    return _case_meta(path_value), duration, 0.0


@app.callback(
    Output("replay-data", "data"),
    Output("replay-figure", "figure"),
    Output("quality-figure", "figure"),
    Output("replay-data-status", "children"),
    Output("replay-data-status", "className"),
    Input("case-selector", "value"),
    Input("eeg-window", "value"),
)
def prepare_replay(path_value: str | None, eeg_window_seconds: int):
    if not path_value:
        message = "Nenhum caso selecionado"
        return (
            None,
            _empty_figure("Replay sincronizado · atualização local", message, height=650),
            _empty_figure("Qualidade do sinal · gate de emissão", message, height=280),
            _status_children(
                "Nenhum caso selecionado",
                "Escolha uma gravação de pesquisa para preparar a simulação offline.",
            ),
            "state-message state-warning",
        )
    if MODEL is None or PREPROCESS is None:
        message = MODEL_ERROR or "Checkpoint indisponível"
        return (
            None,
            _empty_figure("Replay sincronizado · atualização local", message, height=650),
            _empty_figure("Qualidade do sinal · gate de emissão", message, height=280),
            _status_children("Replay indisponível", message),
            "state-message state-error",
        )
    try:
        payload = _replay_payload(path_value)
    except Exception as error:
        message = f"Replay bloqueado: {error}"
        return (
            None,
            _empty_figure("Replay sincronizado · atualização local", message, height=650),
            _empty_figure("Qualidade do sinal · gate de emissão", message, height=280),
            _status_children("Não foi possível preparar o replay", str(error)),
            "state-message state-error",
        )
    if payload.prediction_times.size:
        status = _status_children(
            "Replay preparado",
            f"{_case_source_label(payload.case)} · {payload.prediction_times.size:,} janelas causais prontas para revelação local.",
        )
        status_class = "state-message state-ready"
    else:
        status = _status_children(
            "Caso carregado sem estimativas",
            "A gravação não produziu uma janela causal elegível; os gráficos permanecem sem emissão.",
        )
        status_class = "state-message state-warning"
    return (
        _replay_store_payload(payload),
        _replay_figure(payload, 0.0, int(eeg_window_seconds or 10)),
        _quality_figure(payload, 0.0),
        status,
        status_class,
    )


@app.callback(
    Output("play-state", "data"),
    Output("play-button", "children"),
    Output("replay-time", "value", allow_duplicate=True),
    Output("play-button", "aria-pressed"),
    Input("play-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    State("play-state", "data"),
    prevent_initial_call=True,
)
def control_replay(play_clicks: int, reset_clicks: int, playing: bool):
    del play_clicks, reset_clicks
    from dash import ctx

    if ctx.triggered_id == "reset-button":
        return False, "Iniciar replay", 0.0, "false"
    next_state = not bool(playing)
    return next_state, ("Pausar replay" if next_state else "Continuar replay"), no_update, str(next_state).lower()


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
        const stageLabels = {
            deep: "Faixa estimada abaixo de 40",
            general: "Faixa estimada de 40 a 59",
            light: "Faixa estimada de 60 a 79",
            awake: "Faixa estimada de 80 a 100",
            abstain: "Sem emissão · sinal insuficiente"
        };
        const progress = (payload && Number(payload.duration) > 0) ? Math.max(0, Math.min(100, seconds / Number(payload.duration) * 100)) : 0;
        const progressStyle = {width: progress.toFixed(2) + "%"};
        const progressText = progress.toFixed(1).replace(".", ",") + "%";
        const progressNow = Number(progress.toFixed(1));
        if (!payload || !baseFigure || !baseQualityFigure) {
            return [baseFigure, baseQualityFigure, "—", "Aguardando replay", "—", "Aguardando replay", "—", "Disponível após a primeira estimativa", "—", "Gate indisponível até preparar replay", "Aguardando caso", "00:00", progressStyle, progressText, progressNow];
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
        const finiteValue = value => typeof value === "number" && Number.isFinite(value) ? value : NaN;
        let qualityDetail = "gate de emissão " + formatNumber(finiteValue(payload.min_quality), 2);
        let revealed = 0;
        if (predictionIndex >= 0) {
            revealed = predictionIndex + 1;
            const rawCnn = finiteValue(payload.raw_predictions[predictionIndex]);
            const smoothedCnn = finiteValue(payload.smoothed_predictions[predictionIndex]);
            const predictionTime = Number(payload.prediction_times[predictionIndex]);
            const referenceIndex = lastIndexAtOrBefore(payload.bis_times, predictionTime);
            const reference = referenceIndex >= 0 ? finiteValue(payload.bis_values[referenceIndex]) : NaN;
            const error = Number.isFinite(smoothedCnn) && Number.isFinite(reference) ? smoothedCnn - reference : NaN;
            cnnValue = formatNumber(smoothedCnn, 1);
            cnnDetail = (stageLabels[payload.stages[predictionIndex]] || payload.stages[predictionIndex] || "") + " · t=" + predictionTime.toFixed(1) + "s";
            bisValue = formatNumber(reference, 1);
            bisDetail = Number.isFinite(reference) ? "último ponto observado · t=" + Number(payload.bis_times[referenceIndex]).toFixed(0) + "s" : "sem ponto disponível";
            errorValue = formatNumber(error, 1);
            errorDetail = "CNN bruta " + formatNumber(rawCnn, 1) + " · positivo = acima do BIS";
            qualityValue = formatNumber(finiteValue(payload.qualities[predictionIndex]), 3);
        }
        const status = payload.case_label + " · " + formatClock(seconds) + " · " + revealed + " estimativa(s) revelada(s) · atualização local";
        return [figure, qualityFigure, cnnValue, cnnDetail, bisValue, bisDetail, errorValue, errorDetail, qualityValue, qualityDetail, status, formatClock(seconds), progressStyle, progressText, progressNow];
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
    Output("replay-progress-track", "aria-valuenow"),
    Input("replay-clock", "data"),
    Input("replay-data", "data"),
    State("replay-figure", "figure"),
    State("quality-figure", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("trajectory-meta", "children"),
    Output("trajectory-cards", "children"),
    Output("trajectory-figure", "figure"),
    Output("trajectory-error-figure", "figure"),
    Output("trajectory-data-status", "children"),
    Output("trajectory-data-status", "className"),
    Input("trajectory-case-selector", "value"),
)
def update_trajectory(path_value: str | None):
    if not path_value:
        message = "Nenhum caso selecionado"
        return (
            _case_meta(None),
            _initial_trajectory_cards(),
            _empty_figure("Trajetória completa · BIS contra CNN", message, height=470),
            _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300),
            _status_children(
                "Nenhum caso selecionado",
                "Escolha uma gravação de pesquisa para calcular a trajetória retrospectiva.",
            ),
            "state-message state-warning",
        )
    if MODEL is None or PREPROCESS is None:
        message = MODEL_ERROR or "Checkpoint indisponível"
        return (
            _case_meta(path_value),
            _initial_trajectory_cards(),
            _empty_figure("Trajetória completa · BIS contra CNN", message, height=470),
            _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300),
            _status_children("Trajetória indisponível", message),
            "state-message state-error",
        )
    try:
        payload = _replay_payload(path_value)
        trajectory, error_figure = _trajectory_figure(payload)
    except Exception as error:
        message = f"Não foi possível calcular a trajetória: {error}"
        return (
            _case_meta(path_value),
            _initial_trajectory_cards(),
            _empty_figure("Trajetória completa · BIS contra CNN", message, height=470),
            _empty_figure("Erro ao longo do caso · CNN − BIS", message, height=300),
            _status_children("Não foi possível calcular a trajetória", str(error)),
            "state-message state-error",
        )
    if payload.prediction_times.size:
        status = _status_children(
            "Trajetória calculada",
            f"{_case_source_label(payload.case)} · {payload.prediction_times.size:,} janelas causais disponíveis para inspeção retrospectiva.",
        )
        status_class = "state-message state-ready"
    else:
        status = _status_children(
            "Caso carregado sem estimativas",
            "A gravação não produziu uma janela causal elegível; as figuras permanecem sem emissão.",
        )
        status_class = "state-message state-warning"
    return (
        _case_meta(path_value),
        _trajectory_cards(payload),
        trajectory,
        error_figure,
        status,
        status_class,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8501")), debug=False)

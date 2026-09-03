"""Dash + Plotly research dashboard for BrainSniffer.

Run locally with ``uv run python dash_app.py``.  The dashboard is deliberately
read-only with respect to the stored experiments: the replay feeds causal EEG
chunks to the same :class:`RealtimeEstimator` used by the command-line path and
reveals each output only as simulated time advances.
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
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from brainsniffer.config import DEFAULT_MIN_SIGNAL_QUALITY, PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase, load_case
from brainsniffer.pipeline.realtime import replay_case
from brainsniffer.pipeline.training import load_checkpoint

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("BRAINSNIFFER_DATA_DIR", APP_ROOT / "data/raw"))
VITAL_DIR = Path(os.getenv("BRAINSNIFFER_VITAL_DIR", APP_ROOT / "data/vitaldb"))
MODEL_PATH = Path(
    os.getenv("BRAINSNIFFER_CHECKPOINT", APP_ROOT / "models/brainsniffer_cnn.pt")
)
REPORTS_DIR = APP_ROOT / "reports"

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


def _load_model() -> tuple[object | None, PreprocessConfig | None, str | None]:
    if not MODEL_PATH.exists():
        return None, None, f"Checkpoint não encontrado: {MODEL_PATH}"
    try:
        model, preprocess, _ = load_checkpoint(MODEL_PATH, device="cpu")
    except Exception as error:  # pragma: no cover - defensive startup guard
        return None, None, f"Falha ao carregar o checkpoint: {error}"
    return model, preprocess, None


MODEL, PREPROCESS, MODEL_ERROR = _load_model()


@dataclass(frozen=True)
class ReplayPayload:
    case: EEGCase
    prediction_times: np.ndarray
    raw_predictions: np.ndarray
    smoothed_predictions: np.ndarray
    qualities: np.ndarray
    stages: tuple[str, ...]


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


@lru_cache(maxsize=8)
def _replay_payload(path_string: str) -> ReplayPayload:
    if MODEL is None or PREPROCESS is None:
        raise RuntimeError(MODEL_ERROR or "Checkpoint indisponível")
    case = load_case(path_string)
    predictions = replay_case(
        MODEL,
        case,
        PREPROCESS,
        stride_seconds=1.0,
        min_quality=DEFAULT_MIN_SIGNAL_QUALITY,
        device="cpu",
    )
    return ReplayPayload(
        case=case,
        prediction_times=np.asarray([item.elapsed_seconds for item in predictions], dtype=float),
        raw_predictions=np.asarray(
            [np.nan if item.raw_bis is None else item.raw_bis for item in predictions], dtype=float
        ),
        smoothed_predictions=np.asarray(
            [np.nan if item.smoothed_bis is None else item.smoothed_bis for item in predictions],
            dtype=float,
        ),
        qualities=np.asarray([item.quality for item in predictions], dtype=float),
        stages=tuple(item.stage for item in predictions),
    )


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


def _card(title: str, value: str, detail: str, tone: str = "blue") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
        className=f"metric-card metric-{tone}",
    )


def _figure_layout(title: str, *, height: int = 330) -> dict[str, object]:
    return {
        "title": {"text": title, "font": {"size": 17, "color": COLORS["ink"]}},
        "height": height,
        "margin": {"l": 52, "r": 26, "t": 58, "b": 48},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#FAFCFE",
        "font": {"family": "Inter, Arial, sans-serif", "color": COLORS["ink"]},
        "hoverlabel": {"bgcolor": COLORS["navy"], "font": {"color": "white"}},
        "legend": {"orientation": "h", "y": 1.04, "x": 0, "font": {"size": 11}},
    }


def _empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**_figure_layout(title))
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


def _evidence_mae_figure() -> go.Figure:
    values = [_metric_value(HOLDOUT_METRICS, "mae"), _metric_value(EXTERNAL_METRICS, "mae")]
    if not np.isfinite(values).any():
        return _empty_figure("MAE da rede contra o BIS", "Relatório de métricas indisponível")
    figure = go.Figure(
        go.Bar(
            x=["Holdout interno", "VitalDB externo"],
            y=values,
            text=[_format_number(value) for value in values],
            textposition="outside",
            marker_color=[COLORS["blue"], COLORS["orange"]],
            hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(**_figure_layout("MAE da rede contra o BIS"))
    figure.update_yaxes(title="Erro médio absoluto (pontos BIS)", rangemode="tozero", gridcolor=COLORS["line"])
    return figure


def _evidence_pearson_figure() -> go.Figure:
    values = [
        _metric_value(HOLDOUT_METRICS, "pearson_r"),
        _metric_value(EXTERNAL_METRICS, "pearson_r"),
    ]
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


def _evidence_offset_figure() -> go.Figure:
    points = OFFSET_REPORT.get("offsets", [])
    if not isinstance(points, list) or not points:
        return _empty_figure("Sensibilidade ao alinhamento EEG–BIS", "Relatório de offset indisponível")
    offsets = [_finite_number(item.get("offset_seconds")) for item in points if isinstance(item, dict)]
    maes = [
        _finite_number(item.get("metrics", {}).get("mae"))
        for item in points
        if isinstance(item, dict) and isinstance(item.get("metrics"), dict)
    ]
    pearsons = [
        _finite_number(item.get("metrics", {}).get("pearson_r"))
        for item in points
        if isinstance(item, dict) and isinstance(item.get("metrics"), dict)
    ]
    if not offsets or len(offsets) != len(maes) or len(offsets) != len(pearsons):
        return _empty_figure("Sensibilidade ao alinhamento EEG–BIS", "Dados insuficientes")
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=offsets,
            y=maes,
            mode="lines+markers",
            name="MAE",
            line={"color": COLORS["orange"], "width": 3},
            marker={"size": 7},
            hovertemplate="offset %{x:.0f}s<br>MAE %{y:.2f}<extra></extra>",
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
            marker={"size": 7},
            hovertemplate="offset %{x:.0f}s<br>Pearson %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(**_figure_layout("Sensibilidade exploratória ao alinhamento EEG–BIS", height=310))
    figure.update_xaxes(title="Offset do rótulo BIS (s)", gridcolor=COLORS["line"])
    figure.update_yaxes(title_text="MAE", secondary_y=False, gridcolor=COLORS["line"])
    figure.update_yaxes(title_text="Pearson r", secondary_y=True, range=[0.65, 0.85])
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


def _prediction_index(payload: ReplayPayload, seconds: float) -> int:
    if payload.prediction_times.size == 0:
        return -1
    index = int(np.searchsorted(payload.prediction_times, seconds, side="right") - 1)
    return index if index >= 0 else -1


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int = 2600) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, max_points, dtype=int)
    return x[indices], y[indices]


def _replay_figure(payload: ReplayPayload, seconds: float, eeg_window_seconds: float) -> go.Figure:
    case = payload.case
    current = max(0.0, min(float(seconds), case.duration_seconds))
    window_start = max(0.0, current - eeg_window_seconds)
    sample_start = max(0, int(window_start * case.sampling_rate))
    sample_end = min(case.eeg.size, max(sample_start + 1, int(current * case.sampling_rate)))
    eeg_x = np.arange(sample_start, sample_end, dtype=float) / case.sampling_rate
    eeg_y = case.eeg[sample_start:sample_end].astype(float)
    eeg_x, eeg_y = _downsample(eeg_x, eeg_y)

    bis_times = np.arange(case.bis.size, dtype=float) * case.label_interval_seconds
    bis_mask = bis_times <= current
    bis_values = case.bis.astype(float)
    bis_mask &= np.isfinite(bis_values)
    prediction_mask = payload.prediction_times <= current
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
            x=eeg_x,
            y=eeg_y,
            mode="lines",
            name="EEG frontal",
            line={"color": COLORS["blue"], "width": 1.2},
            hovertemplate="t=%{x:.2f}s<br>EEG=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if bis_mask.any():
        figure.add_trace(
            go.Scatter(
                x=bis_times[bis_mask],
                y=bis_values[bis_mask],
                mode="lines+markers",
                name="BIS referência",
                line={"color": COLORS["navy"], "width": 2.5, "shape": "hv"},
                marker={"size": 6},
                hovertemplate="t=%{x:.0f}s<br>BIS=%{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if prediction_mask.any():
        figure.add_trace(
            go.Scatter(
                x=payload.prediction_times[prediction_mask],
                y=payload.raw_predictions[prediction_mask],
                mode="lines",
                name="CNN bruta",
                line={"color": COLORS["purple"], "width": 1.3, "dash": "dot"},
                connectgaps=False,
                hovertemplate="t=%{x:.1f}s<br>CNN bruta=%{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=payload.prediction_times[prediction_mask],
                y=payload.smoothed_predictions[prediction_mask],
                mode="lines+markers",
                name="CNN suavizada",
                line={"color": COLORS["teal"], "width": 3},
                marker={"size": 5},
                connectgaps=False,
                hovertemplate="t=%{x:.1f}s<br>CNN suavizada=%{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    for threshold in (40, 60, 80):
        figure.add_hline(
            y=threshold,
            line={"color": COLORS["line"], "width": 1, "dash": "dot"},
            row=2,
            col=1,
        )
    figure.add_vline(x=current, line={"color": COLORS["red"], "width": 2}, row=1, col=1)
    figure.add_vline(x=current, line={"color": COLORS["red"], "width": 2}, row=2, col=1)
    figure.update_layout(**_figure_layout("Replay sincronizado", height=650))
    figure.update_xaxes(title="Tempo do replay (s)", range=[window_start, max(window_start + 5, current)], row=1, col=1)
    figure.update_xaxes(title="Tempo do replay (s)", range=[0, max(10, current + 2)], row=2, col=1)
    figure.update_yaxes(title="Amplitude", gridcolor=COLORS["line"], row=1, col=1)
    figure.update_yaxes(title="Índice BIS (0–100)", range=[0, 100], gridcolor=COLORS["line"], row=2, col=1)
    return figure


def _quality_figure(payload: ReplayPayload, seconds: float) -> go.Figure:
    mask = payload.prediction_times <= seconds
    if not mask.any():
        return _empty_figure("Qualidade e disponibilidade da saída", "Aguardando a primeira janela causal de 5 s")
    figure = go.Figure(
        go.Scatter(
            x=payload.prediction_times[mask],
            y=payload.qualities[mask],
            mode="lines+markers",
            name="Qualidade",
            line={"color": COLORS["green"], "width": 2.5},
            marker={"size": 5},
            hovertemplate="t=%{x:.1f}s<br>qualidade=%{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=DEFAULT_MIN_SIGNAL_QUALITY,
        line={"color": COLORS["orange"], "width": 1.5, "dash": "dash"},
        annotation_text=f"gate {DEFAULT_MIN_SIGNAL_QUALITY:.2f}",
        annotation_position="bottom right",
    )
    figure.add_vline(x=seconds, line={"color": COLORS["red"], "width": 2})
    figure.update_layout(**_figure_layout("Qualidade do sinal · gate de emissão", height=280))
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
            html.Span(
                f" · {_format_clock(case.duration_seconds)} · {case.sampling_rate} Hz · "
                f"{case.bis.size:,} pontos BIS"
            ),
        ],
        className="case-meta",
    )


def _initial_replay_cards() -> list[html.Div]:
    return [
        _card("CNN suavizada", "—", "Aguardando janela causal", "teal"),
        _card("BIS de referência", "—", "Último ponto observado", "navy"),
        _card("Erro CNN − BIS", "—", "Disponível após a primeira predição", "orange"),
        _card("Qualidade", "—", "Gate padrão 0,20", "green"),
    ]


def _build_layout() -> html.Div:
    options = [{"label": _case_label(path), "value": str(path)} for path in CASE_PATHS]
    default_value = str(DEFAULT_CASE) if DEFAULT_CASE else None
    holdout_mae = _metric_value(HOLDOUT_METRICS, "mae")
    holdout_r = _metric_value(HOLDOUT_METRICS, "pearson_r")
    external_mae = _metric_value(EXTERNAL_METRICS, "mae")
    external_r = _metric_value(EXTERNAL_METRICS, "pearson_r")
    model_status = (
        "checkpoint causal carregado em CPU"
        if MODEL is not None
        else "checkpoint indisponível · replay bloqueado"
    )
    duration = DEFAULT_CASE and load_case(DEFAULT_CASE).duration_seconds or 600.0
    marks = {0: "0:00", int(min(duration, 60)): "1:00", int(min(duration, 300)): "5:00"}
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Div("BRAIN SNIFFER · RESEARCH CONSOLE", className="eyebrow"),
                            html.H1("EEG → CNN → BIS", className="hero-title"),
                            html.P(
                                "Replay causal de uma cirurgia simulada para comparar, no mesmo relógio, "
                                "o sinal recebido, a saída da rede neural e o BIS de referência.",
                                className="hero-subtitle",
                            ),
                        ],
                        className="hero-copy",
                    ),
                    html.Div(
                        [html.Span("RESEARCH ONLY", className="research-badge"), html.Div(model_status, className="hero-status")],
                        className="hero-side",
                    ),
                ],
                className="hero",
            ),
            html.Div(
                "Uso exclusivamente experimental/educacional. O BIS é referência do monitor e a CNN é uma estimativa de pesquisa; nada nesta tela comanda anestésicos ou substitui avaliação clínica.",
                className="safety-banner",
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.Div("O que o experimento mostra", className="section-kicker"),
                            html.H2("A comparação certa é rede neural contra o BIS observado", className="section-title"),
                            html.P(
                                "O holdout interno mede se o checkpoint reproduz o BIS em casos não usados no treino. "
                                "A validação VitalDB fica separada para tornar visível a mudança de domínio, em vez de misturar os números.",
                                className="section-lead",
                            ),
                        ],
                        className="section-intro",
                    ),
                    html.Div(
                        [
                            _card("Holdout · MAE", _format_number(holdout_mae), "Figshare · 5 casos", "blue"),
                            _card("Holdout · Pearson", _format_number(holdout_r, 3), "CNN versus BIS", "teal"),
                            _card("VitalDB · MAE", _format_number(external_mae), "15 casos · sem retreino", "orange"),
                            _card("VitalDB · Pearson", _format_number(external_r, 3), "mudança de domínio", "red"),
                        ],
                        className="metric-grid",
                    ),
                    html.Div(
                        [
                            html.Div(dcc.Graph(figure=_evidence_mae_figure(), config={"displayModeBar": False}), className="chart-card"),
                            html.Div(dcc.Graph(figure=_evidence_pearson_figure(), config={"displayModeBar": False}), className="chart-card"),
                        ],
                        className="chart-grid two-col",
                    ),
                    html.Div(
                        [
                            html.Div(dcc.Graph(figure=_evidence_offset_figure(), config={"displayModeBar": False}), className="chart-card"),
                            html.Div(
                                [
                                    html.Div("LEITURA PARA A DECISÃO", className="mini-kicker"),
                                    html.H3("Por que os gráficos estão separados?"),
                                    html.P("1. O holdout mostra a execução reproduzível no domínio do treino."),
                                    html.P("2. O VitalDB mostra que bom desempenho interno não prova generalização."),
                                    html.P("3. O offset é exploratório e não pode ser escolhido pós-hoc para pacientes."),
                                    html.Div("A linha vermelha no replay marca o presente simulado. Dados futuros não são revelados na comparação.", className="callout"),
                                ],
                                className="explanation-card",
                            ),
                        ],
                        className="chart-grid two-col lower-evidence",
                    ),
                    html.Div("REPLAY OPERACIONAL", className="section-kicker replay-kicker"),
                    html.H2("Como se fosse uma cirurgia: EEG entrando, CNN respondendo", className="section-title"),
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
                                        className="dark-dropdown",
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
                                    ),
                                ],
                                className="control-block window-control",
                            ),
                        ],
                        className="control-grid",
                    ),
                    html.Div(
                        [
                            html.Button("▶ Iniciar replay", id="play-button", n_clicks=0, className="button-primary"),
                            html.Button("↺ Reiniciar", id="reset-button", n_clicks=0, className="button-secondary"),
                            html.Div("O modelo começa a responder após preencher a janela causal de 5 s.", className="replay-hint"),
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
                    ),
                    dcc.Interval(id="replay-interval", interval=500, n_intervals=0),
                    dcc.Store(id="play-state", data=False),
                    html.Div(id="replay-status", className="replay-status"),
                    html.Div(id="replay-cards", children=_initial_replay_cards(), className="metric-grid replay-metrics"),
                    html.Div(dcc.Graph(id="replay-figure", figure=_empty_figure("Replay sincronizado", "Escolha um caso e inicie o replay"), config={"displayModeBar": False}), className="chart-card replay-main-chart"),
                    html.Div(dcc.Graph(id="quality-figure", figure=_empty_figure("Qualidade e disponibilidade da saída", "Aguardando o replay"), config={"displayModeBar": False}), className="chart-card"),
                    html.Div(
                        [
                            html.Strong("Como ler a tela: "),
                            "a curva azul é o EEG recebido; a linha azul-marinho em degraus é o BIS do arquivo; "
                            "a linha turquesa é a CNN suavizada; pontilhada roxa é a saída bruta. "
                            "A CNN recebe somente a janela causal anterior ao cursor.",
                        ],
                        className="legend-note",
                    ),
                ],
                className="page-content",
            ),
            html.Footer(
                "BrainSniffer · pipeline auditável · checkpoint congelado · sem uso clínico",
                className="footer",
            ),
        ],
        className="app-shell",
    )


app = Dash(__name__, title="BrainSniffer · EEG → CNN → BIS", update_title=None)
server = app.server
app.layout = _build_layout()


@server.get("/healthz")
@server.get("/_stcore/health")
def healthz():
    return {"status": "ok"}


@app.callback(
    Output("case-meta", "children"),
    Output("replay-time", "max"),
    Output("replay-time", "value"),
    Input("case-selector", "value"),
)
def select_case(path_value: str | None):
    if not path_value:
        return _case_meta(None), 600.0, 0.0
    try:
        duration = load_case(path_value).duration_seconds
    except Exception as error:
        return html.Div(f"Não foi possível abrir o caso: {error}", className="case-meta case-error"), 600.0, 0.0
    return _case_meta(path_value), duration, 0.0


@app.callback(
    Output("play-state", "data"),
    Output("play-button", "children"),
    Output("replay-time", "value", allow_duplicate=True),
    Input("play-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    State("play-state", "data"),
    prevent_initial_call=True,
)
def control_replay(play_clicks: int, reset_clicks: int, playing: bool):
    del play_clicks, reset_clicks
    from dash import ctx

    if ctx.triggered_id == "reset-button":
        return False, "▶ Iniciar replay", 0.0
    next_state = not bool(playing)
    return next_state, ("❚❚ Pausar replay" if next_state else "▶ Continuar replay"), no_update


@app.callback(
    Output("replay-time", "value", allow_duplicate=True),
    Input("replay-interval", "n_intervals"),
    State("replay-time", "value"),
    State("play-state", "data"),
    State("speed", "value"),
    State("replay-time", "max"),
    prevent_initial_call=True,
)
def advance_replay(_n_intervals: int, current: float, playing: bool, speed: float, maximum: float):
    if not playing:
        return no_update
    next_value = min(float(maximum), float(current or 0) + 0.5 * float(speed or 1))
    return next_value


@app.callback(
    Output("replay-cards", "children"),
    Output("replay-status", "children"),
    Output("replay-figure", "figure"),
    Output("quality-figure", "figure"),
    Input("case-selector", "value"),
    Input("replay-time", "value"),
    Input("eeg-window", "value"),
)
def update_replay(path_value: str | None, seconds: float, eeg_window_seconds: int):
    if not path_value:
        return _initial_replay_cards(), "Nenhum caso selecionado.", _empty_figure("Replay sincronizado", "Nenhum caso selecionado"), _empty_figure("Qualidade e disponibilidade da saída", "Nenhum caso selecionado")
    if MODEL is None or PREPROCESS is None:
        message = MODEL_ERROR or "Checkpoint indisponível"
        return _initial_replay_cards(), message, _empty_figure("Replay sincronizado", message), _empty_figure("Qualidade e disponibilidade da saída", message)
    try:
        payload = _replay_payload(path_value)
    except Exception as error:
        message = f"Replay bloqueado: {error}"
        return _initial_replay_cards(), message, _empty_figure("Replay sincronizado", message), _empty_figure("Qualidade e disponibilidade da saída", message)

    current = max(0.0, min(float(seconds or 0), payload.case.duration_seconds))
    prediction_index = _prediction_index(payload, current)
    if prediction_index < 0:
        cards = _initial_replay_cards()
        status = f"{_case_label(Path(path_value))} · {_format_clock(current)} · aquecendo janela causal de 5 s"
    else:
        cnn_value = payload.smoothed_predictions[prediction_index]
        quality = payload.qualities[prediction_index]
        stage = STAGE_LABELS.get(payload.stages[prediction_index], payload.stages[prediction_index])
        pred_time = payload.prediction_times[prediction_index]
        bis_value, bis_time = _reference_at_or_before(payload.case, pred_time)
        error = cnn_value - bis_value if bis_value is not None and np.isfinite(cnn_value) else None
        cards = [
            _card("CNN suavizada", _format_number(cnn_value, 1), f"{stage} · t={pred_time:.1f}s", "teal"),
            _card("BIS de referência", _format_number(bis_value, 1), f"último ponto observado · t={bis_time:.0f}s" if bis_time is not None else "sem ponto disponível", "navy"),
            _card("Erro CNN − BIS", _format_number(error, 1), "positivo = CNN acima do BIS", "orange"),
            _card("Qualidade", _format_number(quality, 3), f"gate de emissão {DEFAULT_MIN_SIGNAL_QUALITY:.2f}", "green"),
        ]
        status = f"{_case_label(Path(path_value))} · {_format_clock(current)} · {prediction_index + 1} predição(ões) revelada(s) · saída causal ativa"
    return cards, status, _replay_figure(payload, current, int(eeg_window_seconds or 10)), _quality_figure(payload, current)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8501")), debug=False)

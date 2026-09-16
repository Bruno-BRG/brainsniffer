import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
import torch

import dash_app
from brainsniffer.config import PreprocessConfig
from brainsniffer.data.mat_reader import EEGCase
from dash_app import (
    APP_DESCRIPTION,
    LLMS_TEXT,
    MIN_CHART_HEIGHT,
    ROBOTS_TEXT,
    STAGE_LABELS,
    _fast_replay_case,
    _theoretical_training_mae,
    app,
)

STYLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.language: str | None = None
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.language = attributes.get("lang")
        if tag == "meta" and attributes.get("name") == "description":
            self.description = attributes.get("content")


def _css_tokens() -> dict[str, str]:
    css = STYLE_PATH.read_text(encoding="utf-8")
    return dict(re.findall(r"(--[a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;", css))


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    brighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (brighter + 0.05) / (darker + 0.05)


def _walk_components(component: object):
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
    elif hasattr(children, "to_plotly_json"):
        yield from _walk_components(children)


def _component_by_id(component_id: str):
    for component in _walk_components(app.layout):
        if getattr(component, "id", None) == component_id:
            return component
    raise AssertionError(f"componente ausente: {component_id}")


def _component_text(component: object) -> str:
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(_component_text(child) for child in component)
    return _component_text(getattr(component, "children", None))


def test_dashboard_document_declares_portuguese_and_a_description():
    with app.server.test_request_context("/"):
        document = app.index()

    parser = _MetadataParser()
    parser.feed(document)

    assert parser.language == "pt-BR"
    assert parser.description == APP_DESCRIPTION


def test_crawler_metadata_routes_are_plain_text_and_explicitly_research_only():
    client = app.server.test_client()

    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.mimetype == "text/plain"
    assert robots.get_data(as_text=True) == ROBOTS_TEXT

    llms = client.get("/llms.txt")
    llms_body = llms.get_data(as_text=True)
    assert llms.status_code == 200
    assert llms.mimetype == "text/markdown"
    assert llms_body == LLMS_TEXT
    assert llms_body.startswith("# BrainSniffer\n")
    assert "not a medical device" in llms_body


def test_dashboard_text_colors_keep_wcag_aa_contrast():
    tokens = _css_tokens()

    def resolve(value: str) -> str:
        return tokens.get(value, value)

    foreground_background_pairs = [
        ("--muted", "#ffffff"),
        ("--muted", "--canvas"),
        ("--teal", "#ffffff"),
        ("--red", "#ffffff"),
        ("--hero-status", "#1f7f86"),
        ("#ffffff", "--blue"),
        ("#ffffff", "--teal"),
    ]

    for foreground, background in foreground_background_pairs:
        ratio = _contrast_ratio(resolve(foreground), resolve(background))
        assert ratio >= 4.5, f"contraste {foreground}/{background} = {ratio:.2f}:1"


def test_dashboard_charts_are_responsive_and_have_explanatory_captions():
    components = list(_walk_components(app.layout))
    graphs = [component for component in components if type(component).__name__ == "Graph"]
    chart_wrappers = [component for component in components if type(component).__name__ == "Figure"]

    assert graphs
    assert len(chart_wrappers) == len(graphs)
    for graph in graphs:
        props = graph.to_plotly_json()["props"]
        layout_height = int(props["figure"].layout.height)
        expected_height = max(MIN_CHART_HEIGHT, layout_height)
        assert props["responsive"] is True
        assert props["config"]["responsive"] is True
        assert props["config"]["displayModeBar"] is False
        assert props["style"]["height"] == f"{expected_height}px"
        assert props["style"]["minHeight"] == f"{MIN_CHART_HEIGHT}px"
        assert props["style"]["width"] == "100%"
    for wrapper in chart_wrappers:
        props = wrapper.to_plotly_json()["props"]
        assert props["aria-label"]
        assert any(type(child).__name__ == "Figcaption" for child in props["children"])


def test_chart_css_does_not_expand_dash_observer_placeholder():
    normalized_css = " ".join(STYLE_PATH.read_text(encoding="utf-8").split())

    assert ".chart-card .dash-graph > div" not in normalized_css
    assert (
        ".chart-card .js-plotly-plot, .chart-card .plot-container, "
        ".chart-card .svg-container { height: 100% !important; }"
        in normalized_css
    )


def test_dashboard_keeps_mobile_navigation_and_keyboard_access_visible():
    tabs = _component_by_id("main-tabs").to_plotly_json()["props"]
    skip_link = next(
        component
        for component in _walk_components(app.layout)
        if getattr(component, "className", None) == "skip-link"
    )

    assert tabs["mobile_breakpoint"] == 0
    assert skip_link.href == "#main-content"
    assert _component_by_id("main-content")


def test_dashboard_exposes_loading_empty_and_error_states(monkeypatch):
    for component_id, expected_text in (
        ("trajectory-loading", "Calculando trajetória"),
        ("replay-loading", "Preparando replay"),
    ):
        loading = _component_by_id(component_id).to_plotly_json()["props"]
        assert expected_text in _component_text(loading["custom_spinner"])

    for component_id in ("trajectory-data-status", "replay-data-status", "replay-status"):
        status = _component_by_id(component_id).to_plotly_json()["props"]
        assert status["role"] in {"status", "alert"}
        assert status["aria-live"] in {"polite", "assertive"}

    empty_replay = dash_app.prepare_replay(None, 10)
    empty_trajectory = dash_app.update_trajectory(None)
    assert empty_replay[-1] == "state-message state-warning"
    assert "Nenhum caso selecionado" in _component_text(empty_replay[-2])
    assert empty_trajectory[-1] == "state-message state-warning"
    assert "Nenhum caso selecionado" in _component_text(empty_trajectory[-2])

    monkeypatch.setattr(dash_app, "MODEL", None)
    error_replay = dash_app.prepare_replay("caso-inexistente", 10)
    error_trajectory = dash_app.update_trajectory("caso-inexistente")
    assert error_replay[-1] == "state-message state-error"
    assert "indisponível" in _component_text(error_replay[-2]).lower()
    assert error_trajectory[-1] == "state-message state-error"
    assert "indisponível" in _component_text(error_trajectory[-2]).lower()


def test_dashboard_uses_cautious_scientific_language_for_estimated_ranges():
    assert STAGE_LABELS == {
        "deep": "Faixa estimada abaixo de 40",
        "general": "Faixa estimada de 40 a 59",
        "light": "Faixa estimada de 60 a 79",
        "awake": "Faixa estimada de 80 a 100",
        "abstain": "Sem emissão · sinal insuficiente",
    }

    page_text = _component_text(app.layout)
    assert "não medem segurança clínica" in page_text
    assert "não monitorização em tempo real" in page_text
    assert "não substitui avaliação clínica" in page_text


def test_dashboard_demo_guidance_keeps_research_limits_and_technical_support():
    page_text = _component_text(app.layout)
    assert "Problema: até que ponto o EEG permite estimar o BIS registrado?" in page_text
    assert "Escolha um caso em Trajetória" in page_text
    assert "resultados e limitações" in page_text
    assert "não limiares científicos" in page_text
    details = [
        component for component in _walk_components(app.layout)
        if type(component).__name__ == "Details"
    ]
    assert any("tensores do modelo" in _component_text(item) for item in details)
    tabs = _component_by_id("main-tabs").children
    assert {tab.label for tab in tabs} >= {"Resultados", "Método e limitações"}


def test_dashboard_distinguishes_active_cross_dataset_from_mixed_same_source_holdout(
    monkeypatch,
):
    corpus_text = _component_text(dash_app._build_corpus_tab())
    active_text = " ".join(
        [
            _component_text(dash_app._build_overview_tab()),
            _component_text(dash_app._build_statistics_tab()),
            _component_text(dash_app._build_method_tab()),
        ]
    )

    assert "checkpoint ativo" in active_text.lower()
    assert "desenvolvido somente em Figshare" in active_text
    assert "avaliação cruzada VitalDB" in active_text
    assert "externa ao desenvolvimento" in active_text

    assert "10 casos VitalDB" in corpus_text
    assert "15 casos VitalDB" in corpus_text
    assert "holdout por participante da mesma fonte/domínio" in corpus_text
    assert "benchmark histórico reutilizado" in corpus_text
    assert "não confirmatório" in corpus_text

    dashboard_text = _component_text(app.layout)
    assert "divisão por caso/cirurgia" in dashboard_text
    assert (
        "No Figshare, não há identificador que permita afirmar separação por paciente"
        in dashboard_text
    )
    for unsupported_claim in (
        "VitalDB · externo",
        "externo congelado",
        "mesmos holdouts congelados",
        "ganho de generalização",
        "15 casos externos",
        "Figshare · holdout por paciente",
        "Figshare · split por paciente",
    ):
        assert unsupported_claim not in dashboard_text

    monkeypatch.setattr(dash_app, "HOLDOUT_METRICS", {"mae": 10.0, "pearson_r": 0.2})
    monkeypatch.setattr(dash_app, "EXTERNAL_METRICS", {"mae": 12.0, "pearson_r": 0.1})
    monkeypatch.setattr(
        dash_app,
        "MIXED_FIGSHARE_REPORT",
        {"metrics": {"mae": 8.0, "pearson_r": 0.4}},
    )
    monkeypatch.setattr(
        dash_app,
        "MIXED_EXTERNAL_REPORT",
        {"metrics": {"mae": 9.0, "pearson_r": 0.5}},
    )

    comparison = dash_app._corpus_experiment_figure()
    axis_labels = {str(label) for trace in comparison.data for label in trace.x}
    trace_names = {str(trace.name) for trace in comparison.data}
    assert axis_labels == {
        dash_app.MIXED_FIGSHARE_BENCHMARK_LABEL,
        dash_app.MIXED_VITALDB_HOLDOUT_LABEL,
    }
    assert not any("extern" in label.lower() for label in axis_labels)
    assert trace_names == {dash_app.ACTIVE_MODEL_LABEL, dash_app.MIXED_MODEL_LABEL}
    assert {row[0] for row in dash_app._corpus_experiment_rows()} == axis_labels

    active_comparison = dash_app._evidence_mae_figure()
    assert list(active_comparison.data[0].x) == [
        dash_app.FIGSHARE_HOLDOUT_LABEL,
        dash_app.ACTIVE_VITALDB_LABEL,
    ]


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


@pytest.fixture
def allowed_case(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    root.mkdir()
    path = root / "case1.mat"
    path.write_bytes(b"recording one")
    monkeypatch.setattr(dash_app, "DATA_DIR", root)
    monkeypatch.setattr(dash_app, "CASE_PATHS", [path])
    monkeypatch.setattr(dash_app, "DEFAULT_CASE", path)
    return path


def test_case_allowlist_rejects_traversal_unlisted_and_external_symlinks(allowed_case):
    assert dash_app._resolve_case(str(allowed_case)) == allowed_case
    for value in (str(allowed_case.parent / ".." / "cases" / allowed_case.name),
                  str(allowed_case.parent / "case2.mat"), "/etc/passwd",
                  {"path": str(allowed_case)}):
        with pytest.raises(ValueError, match="não permitido"):
            dash_app._resolve_case(value)
    outside = allowed_case.parent.parent / "private.mat"
    outside.write_bytes(b"not a recording")
    allowed_case.unlink()
    allowed_case.symlink_to(outside)
    with pytest.raises(ValueError, match="fora da raiz"):
        dash_app._resolve_case(str(allowed_case))
    assert allowed_case not in dash_app._available_cases()


@pytest.mark.parametrize("callback", ["select_case", "prepare_replay", "update_trajectory"])
def test_callbacks_never_load_unlisted_paths(monkeypatch, callback):
    monkeypatch.setattr(dash_app, "MODEL", _ConstantModel())
    monkeypatch.setattr(dash_app, "PREPROCESS", PreprocessConfig())
    calls = []
    monkeypatch.setattr(dash_app, "load_case", lambda path: calls.append(path))
    function = getattr(dash_app, callback)
    result = (function("/etc/passwd", 10) if callback == "prepare_replay"
              else function("/etc/passwd"))
    assert calls == []
    assert "não permitido" in _component_text(result)


def test_replay_cache_uses_content_and_revalidates_allowlist(allowed_case, monkeypatch):
    calls = []
    monkeypatch.setattr(
        dash_app, "_cached_replay_payload",
        lambda path, digest: calls.append((path, digest)) or digest,
    )
    first = dash_app._replay_payload(str(allowed_case))
    allowed_case.write_bytes(b"recording two")
    second = dash_app._replay_payload(str(allowed_case))
    assert first != second
    monkeypatch.setattr(dash_app, "CASE_PATHS", [])
    with pytest.raises(ValueError):
        dash_app._replay_payload(str(allowed_case))
    assert len(calls) == 2


def test_report_binding_requires_actual_digest_and_executed_gate():
    report = {"checkpoint_sha256": "actual", "min_quality": dash_app.EFFECTIVE_MIN_QUALITY}
    assert dash_app._compatible_report(report, "actual")
    assert not dash_app._compatible_report(report, "other")
    assert not dash_app._compatible_report({}, None)
    assert not dash_app._compatible_report(dict(report, min_quality=0.9), "actual")


def test_model_digest_is_computed_not_trusted_from_metadata(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"safe mock; never unpickled")
    monkeypatch.setattr(dash_app, "MODEL_PATH", checkpoint)
    monkeypatch.setattr(
        dash_app, "load_checkpoint",
        lambda *a, **kw: (_ConstantModel(), PreprocessConfig(), {"checkpoint_sha256": "wrong"}),
    )
    _, _, metadata, error = dash_app._load_model()
    assert error is None
    assert metadata["checkpoint_sha256"] == dash_app._sha256(checkpoint)


def test_readiness_and_layout_diagnose_corrupt_default(allowed_case, monkeypatch):
    monkeypatch.setattr(dash_app, "MODEL", _ConstantModel())
    monkeypatch.setattr(dash_app, "PREPROCESS", PreprocessConfig())
    monkeypatch.setattr(dash_app, "EVIDENCE_ERRORS", [])
    def corrupt(path):
        raise ValueError("corrupt recording")
    monkeypatch.setattr(dash_app, "load_case", corrupt)
    assert "corrupt recording" in _component_text(dash_app._build_layout())
    response = app.server.test_client().get("/healthz")
    assert response.status_code == 503
    assert "corrupt recording" in str(response.json)


def test_readiness_checks_model_case_evidence_and_frozen_artifacts(allowed_case, monkeypatch):
    monkeypatch.setattr(dash_app, "MODEL", _ConstantModel())
    monkeypatch.setattr(dash_app, "PREPROCESS", PreprocessConfig())
    monkeypatch.setattr(dash_app, "EVIDENCE_ERRORS", [])
    monkeypatch.setattr(
        dash_app, "load_case",
        lambda path: EEGCase(
            case_id="case1", eeg=np.ones(PreprocessConfig().window_samples), bis=np.ones(1),
        ),
    )
    monkeypatch.setattr(
        dash_app, "FROZEN_ARTIFACTS", {allowed_case: dash_app._artifact_identity(allowed_case)},
    )
    client = app.server.test_client()
    assert client.get("/healthz").status_code == 200
    allowed_case.write_bytes(b"changed artifact")
    assert client.get("/healthz").status_code == 503
    with pytest.raises(RuntimeError, match="reinicie"):
        dash_app._replay_payload(str(allowed_case))
    monkeypatch.setattr(dash_app, "FROZEN_ARTIFACTS", {})
    monkeypatch.setattr(dash_app, "EVIDENCE_ERRORS", ["incompatible evidence"])
    assert client.get("/healthz").status_code == 503
    monkeypatch.setattr(dash_app, "EVIDENCE_ERRORS", [])
    monkeypatch.setattr(dash_app, "MODEL", None)
    assert client.get("/healthz").status_code == 503


def test_trajectory_error_retains_abstention_and_missing_reference_gaps():
    case = EEGCase(
        case_id="gap", eeg=np.ones(1024), bis=np.array([50., 50., np.nan, 50.]),
        label_interval_seconds=1.,
    )
    payload = dash_app.ReplayPayload(
        case, np.arange(4.), np.array([55., np.nan, 55., 55.]),
        np.array([55., np.nan, 55., 55.]), np.ones(4),
        ("general", "abstain", "general", "general"), np.arange(4.), np.ones(4),
    )
    _, errors = dash_app._trajectory_figure(payload)
    for trace in errors.data:
        assert list(trace.x) == [0., 1., 2., 3.]
        assert list(trace.y) == [5., None, None, 5.]
        assert trace.connectgaps is False
    assert dash_app._replay_store_payload(payload)["smoothed_predictions"][1] is None
    quality = dash_app._quality_figure(payload, 0)
    assert quality.layout.shapes[0].y0 == dash_app.EFFECTIVE_MIN_QUALITY


@pytest.mark.parametrize("raw,smooth,reference,expected", [
    (None, None, 50, ["—", "50.0", "—"]),
    (55, 55, None, ["55.0", "—", "—"]),
    (0, 0, 0, ["0.0", "0.0", "0.0"]),
])
def test_clientside_cards_execute_without_coercing_null(raw, smooth, reference, expected):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable; browser execution is separately unverified")
    # Execute the actual registered callback in Node, not a textual assertion.
    # This does not validate Plotly/DOM rendering in a browser.
    script = next(script for script in app._inline_scripts if "const rawCnn" in script)
    payload = {"case_label": "test", "duration": 10, "bis_times": [5],
               "bis_values": [reference], "prediction_times": [5], "raw_predictions": [raw],
               "smoothed_predictions": [smooth], "qualities": [0.3],
               "stages": ["abstain"], "min_quality": 0.2}
    program = "const window = {dash_clientside: {no_update: null}};\n" + script + "\n"
    program += "const funcs = window.dash_clientside._dashprivate_clientside_funcs;\n"
    program += "const fn = Object.values(funcs)[0];\n"
    program += "const figure = {data: [], layout: {meta: {}}};\n"
    program += "const result = fn(5, " + json.dumps(payload) + ", figure, figure);\n"
    program += "console.log(JSON.stringify([result[2], result[4], result[6]]));"
    result = subprocess.run([node, "-e", program], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_theoretical_training_curve_is_anchored_and_has_diminishing_returns():
    counts = np.asarray([13, 20, 30, 40, 50, 60, 80, 100, 1000], dtype=float)
    values = _theoretical_training_mae(counts, anchor_cases=13, anchor_mae=20.0)

    assert values[0] == pytest.approx(20.0)
    assert np.all(np.diff(values) <= 0)
    assert (values[0] - values[6]) / values[0] == pytest.approx(0.15)
    assert (values[6] - values[7]) / values[6] > (values[7] - values[8]) / values[7]
    assert (values[7] - values[8]) / values[7] == pytest.approx(0.015)

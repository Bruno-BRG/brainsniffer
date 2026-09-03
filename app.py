"""Streamlit research interface for BrainSniffer.

Run with: ``uv run streamlit run app.py``.
"""

# The CSS block below is intentionally kept readable as browser CSS.
# ruff: noqa: E501

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from brainsniffer.config import (
    DEFAULT_MIN_SIGNAL_QUALITY,
    PreprocessConfig,
    TrainingConfig,
    default_data_dir,
    default_model_path,
)
from brainsniffer.data.figshare import download_dataset, fetch_manifest, parse_data_files
from brainsniffer.data.mat_reader import load_case
from brainsniffer.data.preprocess import data_handling_policy, load_windows, signal_diagnostics
from brainsniffer.data.vitaldb import download_vitaldb_case
from brainsniffer.pipeline.intake import validate_intake_metadata
from brainsniffer.pipeline.metrics import bootstrap_case_metrics, compute_metrics
from brainsniffer.pipeline.realtime import RealtimeEstimator
from brainsniffer.pipeline.stream_audit import StreamAudit
from brainsniffer.pipeline.streaming import LSLSource, StreamingResampler
from brainsniffer.pipeline.training import (
    build_file_manifest,
    load_checkpoint,
    predict_model,
    runtime_metadata,
    sha256_file,
    train_model,
)

APP_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = APP_ROOT / "reports"


@st.cache_data(show_spinner=False)
def _load_json_report(filename: str) -> dict[str, object]:
    path = REPORTS_DIR / filename
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric_card(column, label: str, value: str, detail: str, tone: str = "blue") -> None:
    column.markdown(
        f"""
        <div class="metric-card {tone}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_layout(title: str) -> dict[str, object]:
    return {
        "title": {"text": title, "font": {"size": 17, "color": "#102A43"}},
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
        "height": 300,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#F8FAFC",
        "font": {"family": "Inter, sans-serif", "color": "#243B53"},
        "xaxis": {"showgrid": False},
        "yaxis": {"gridcolor": "#D9E2EC", "zeroline": False},
        "showlegend": False,
    }


st.set_page_config(page_title="BrainSniffer", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
      :root { --ink:#102A43; --muted:#627D98; --blue:#247BA0; --teal:#20A39E; --orange:#F18F01; --red:#D1495B; }
      .block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #102A43 0%, #163B5C 100%); }
      [data-testid="stSidebar"] * { color: #F0F4F8 !important; }
      [data-testid="stSidebar"] input { background: #244B6B !important; border-color: #3E6C8D !important; }
      .hero { display:flex; justify-content:space-between; align-items:flex-start; gap:2rem; padding:1.4rem 1.6rem; border-radius:22px; background:linear-gradient(120deg,#102A43 0%,#1E5A7A 62%,#20A39E 100%); box-shadow:0 16px 40px rgba(16,42,67,.18); margin-bottom:1.1rem; }
      .hero h1 { color:#fff; margin:0; font-size:2.5rem; letter-spacing:-.04em; }
      .hero p { color:#D9E2EC; margin:.35rem 0 0; font-size:1rem; }
      .hero-badge { color:#102A43; background:#FDE68A; border-radius:999px; padding:.45rem .8rem; font-size:.72rem; font-weight:800; white-space:nowrap; letter-spacing:.08em; }
      .metric-card { background:#fff; border:1px solid #D9E2EC; border-radius:16px; padding:1rem 1.1rem; min-height:120px; box-shadow:0 8px 24px rgba(16,42,67,.06); border-top:4px solid var(--blue); }
      .metric-card.teal { border-top-color:var(--teal); } .metric-card.orange { border-top-color:var(--orange); } .metric-card.red { border-top-color:var(--red); }
      .metric-label { color:var(--muted); font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
      .metric-value { color:var(--ink); font-size:2rem; font-weight:800; line-height:1.15; margin-top:.35rem; }
      .metric-detail { color:var(--muted); font-size:.82rem; margin-top:.35rem; }
      .section-kicker { color:var(--teal); font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; margin-top:.3rem; }
      div[data-testid="stMetricValue"] { color:#102A43; }
      div[data-testid="stTabs"] button { font-weight:700; }
      .stButton > button[kind="primary"] { background:linear-gradient(90deg,#247BA0,#20A39E); border:0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div><h1>🧠 BrainSniffer</h1><p>Pesquisa reproduzível para estimar uma referência BIS a partir de EEG frontal.</p></div>
      <div class="hero-badge">RESEARCH ONLY</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Uso exclusivamente experimental/educacional. A tela não comanda anestésicos, "
    "não substitui o anestesiologista e não está validada para decisão clínica."
)

holdout_report = _load_json_report("figshare_holdout_evaluation.json")
external_report = _load_json_report("vitaldb_external_validation.json")
offset_report = _load_json_report("offset_sensitivity.json")
holdout_metrics = holdout_report.get("recomputed_test_metrics", {})
external_metrics = external_report.get("metrics", {})
if isinstance(holdout_metrics, dict) and isinstance(external_metrics, dict):
    st.markdown('<div class="section-kicker">Painel de evidências</div>', unsafe_allow_html=True)
    st.subheader("Leitura rápida do experimento registrado")
    summary_columns = st.columns(4)
    _metric_card(
        summary_columns[0],
        "Holdout Figshare · MAE",
        f"{float(holdout_metrics.get('mae', 0)):.2f}",
        "5 casos · 5.523 janelas",
        "blue",
    )
    _metric_card(
        summary_columns[1],
        "Holdout Figshare · Pearson",
        f"{float(holdout_metrics.get('pearson_r', 0)):.3f}",
        "checkpoint causal congelado",
        "teal",
    )
    _metric_card(
        summary_columns[2],
        "VitalDB · MAE",
        f"{float(external_metrics.get('mae', 0)):.2f}",
        "15 casos · sem retreino",
        "orange",
    )
    _metric_card(
        summary_columns[3],
        "VitalDB · Pearson",
        f"{float(external_metrics.get('pearson_r', 0)):.3f}",
        "sinal de mudança de domínio",
        "red",
    )
    st.caption("Os cartões resumem métricas de pesquisa; não são confiança clínica nem recomendação anestésica.")

STAGE_LABELS = {
    "deep": "estágio de pesquisa: profundo (BIS estimado)",
    "general": "estágio de pesquisa: geral (BIS estimado)",
    "light": "estágio de pesquisa: leve (BIS estimado)",
    "awake": "estágio de pesquisa: acordado (BIS estimado)",
    "abstain": "ABSTAIN — sinal insuficiente",
}


def _live_session_report(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    audit: StreamAudit,
    preprocess_config: PreprocessConfig,
    stride_seconds: float,
    prediction_count: int,
    abstention_count: int,
    stale_abstention_count: int,
    prediction_qualities: list[float],
    fail_on_audit: bool,
    require_intake: bool,
    stale_timeout_seconds: float,
    intake_report: dict[str, object],
    error: str | None = None,
) -> dict[str, object]:
    """Build privacy-preserving metadata for the last LSL UI session."""

    qualities = np.asarray(prediction_qualities, dtype=np.float64)
    return {
        "report_version": 2,
        "source": "lsl",
        "status": "error" if error else "completed",
        "error": error,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "scope": {
            "intended_use": "research_only",
            "clinical_decision_support": False,
            "controls_anesthetic_delivery": False,
        },
        "runtime": {
            "environment": runtime_metadata(),
            "preprocess_config": asdict(preprocess_config),
            "stride_seconds": float(stride_seconds),
            "min_quality": audit.min_quality,
            "require_metadata": audit.require_metadata,
            "require_timestamps": audit.require_timestamps,
            "fail_on_audit": bool(fail_on_audit),
            "require_intake": bool(require_intake),
            "stale_timeout_seconds": float(stale_timeout_seconds),
        },
        "predictions": {
            "count": prediction_count,
            "abstentions": abstention_count,
            "stale_abstentions": stale_abstention_count,
            "abstention_fraction": (
                abstention_count / prediction_count if prediction_count else None
            ),
            "quality_min": float(qualities.min()) if qualities.size else None,
            "quality_mean": float(qualities.mean()) if qualities.size else None,
        },
        "audit": audit.report().as_dict(),
        "intake": intake_report,
    }


st.sidebar.markdown("## Controle de pesquisa")
st.sidebar.caption("BrainSniffer · pipeline auditável")
data_dir = Path(st.sidebar.text_input("Diretório dos dados", str(APP_ROOT / default_data_dir())))
vital_data_dir = Path(st.sidebar.text_input("Diretório VitalDB", str(APP_ROOT / "data/vitaldb")))
model_path = Path(st.sidebar.text_input("Checkpoint", str(APP_ROOT / default_model_path())))
st.sidebar.divider()
st.sidebar.caption("A aquisição ao vivo exige manifesto técnico completo e permanece restrita à bancada.")

tab_overview, tab_data, tab_explore, tab_train, tab_replay, tab_live = st.tabs(
    ["Visão geral", "Dados", "Explorar", "Treinar", "Replay em fluxo", "EEG ao vivo (LSL)"]
)

with tab_overview:
    st.markdown('<div class="section-kicker">Evidência e interpretação</div>', unsafe_allow_html=True)
    st.subheader("O que os números dizem")
    st.info(
        "O holdout interno indica que o pipeline executa de forma reproduzível no corpus Figshare. "
        "A queda no VitalDB sem retreino é o resultado mais importante para a decisão de pesquisa: "
        "o checkpoint não deve ser apresentado como generalização entre aparelhos, centros ou populações."
    )
    chart_columns = st.columns(2)
    if isinstance(holdout_metrics, dict) and isinstance(external_metrics, dict):
        with chart_columns[0]:
            figure = go.Figure(
                go.Bar(
                    x=["Figshare", "VitalDB"],
                    y=[float(holdout_metrics.get("mae", 0)), float(external_metrics.get("mae", 0))],
                    marker_color=["#247BA0", "#D1495B"],
                    text=[f"{float(holdout_metrics.get('mae', 0)):.2f}", f"{float(external_metrics.get('mae', 0)):.2f}"],
                    textposition="outside",
                )
            )
            figure.update_layout(**_plot_layout("MAE por corpus (menor é melhor)"))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        with chart_columns[1]:
            figure = go.Figure(
                go.Bar(
                    x=["Figshare", "VitalDB"],
                    y=[float(holdout_metrics.get("pearson_r", 0)), float(external_metrics.get("pearson_r", 0))],
                    marker_color=["#20A39E", "#F18F01"],
                    text=[f"{float(holdout_metrics.get('pearson_r', 0)):.3f}", f"{float(external_metrics.get('pearson_r', 0)):.3f}"],
                    textposition="outside",
                )
            )
            figure.update_layout(**_plot_layout("Correlação de Pearson por corpus"))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    if isinstance(offset_report.get("results"), list):
        offset_points = [item for item in offset_report["results"] if isinstance(item, dict)]
        offset_columns = st.columns(2)
        with offset_columns[0]:
            figure = go.Figure()
            figure.add_scatter(
                x=[float(item["offset_seconds"]) for item in offset_points],
                y=[float(item["metrics"]["mae"]) for item in offset_points],
                mode="lines+markers",
                line={"color": "#247BA0", "width": 3},
                marker={"size": 8},
                name="MAE",
            )
            figure.update_layout(**_plot_layout("Sensibilidade ao offset do rótulo"))
            figure.update_xaxes(title="offset (s)")
            figure.update_yaxes(title="MAE")
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        with offset_columns[1]:
            st.markdown("#### Decisões preservadas")
            st.markdown(
                "- split por caso antes da avaliação\n"
                "- pré-processamento causal compartilhado\n"
                "- abstenção quando a qualidade falha\n"
                "- VitalDB mantido fora do treino\n"
                "- relatórios JSON com seed e hash do checkpoint"
            )
            st.caption("A curva de offset é exploratória e não autoriza escolher um atraso pós-hoc para uso em pacientes.")

with tab_data:
    st.subheader("Dataset público")
    try:
        manifest = fetch_manifest()
        files = parse_data_files(manifest)
        st.write(f"**{manifest.get('title', 'Figshare')}** — {len(files)} casos MAT")
        st.write("Licença: CC BY 4.0 · EEG frontal a 128 Hz · BIS de referência a cada 5 s")
        ids = [int(item.case_id.removeprefix("case")) for item in files]
        selected_ids = st.multiselect("Casos para baixar", ids, default=ids)
        overwrite = st.checkbox("Sobrescrever arquivos já baixados", value=False)
        if st.button("Baixar dataset selecionado", type="primary"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def progress(filename: str, downloaded: int, expected: int) -> None:
                progress_bar.progress(min(downloaded / expected, 1.0) if expected else 0.0)
                status.info(f"Baixando {filename}: {downloaded / 1024**2:.1f} MiB")

            paths = download_dataset(
                data_dir, cases=selected_ids, overwrite=overwrite, progress=progress
            )
            status.success(f"{len(paths)} arquivo(s) disponível(is) em {data_dir}")
        local = sorted(data_dir.glob("case*.mat"))
        st.metric("Casos locais", len(local))
        if local:
            st.dataframe(
                pd.DataFrame(
                    {
                        "caso": [path.stem for path in local],
                        "arquivo": [str(path) for path in local],
                    }
                ),
                hide_index=True,
            )
    except Exception as error:
        st.error(f"Não foi possível consultar o dataset: {error}")

    st.divider()
    st.subheader("Avaliação externa exploratória — VitalDB")
    st.caption(
        "Baixa apenas os tracks EEG/BIS dos casos informados. O VitalDB tem termos "
        "de uso próprios; esta etapa não mistura os arquivos ao treino Figshare."
    )
    vital_case_text = st.text_input(
        "Casos VitalDB (separados por vírgula)", value="1", key="vital_case_text"
    )
    vital_bootstrap_samples = int(
        st.number_input(
            "Reamostragens do bootstrap por caso",
            min_value=1,
            value=1000,
            step=100,
            key="vital_bootstrap_samples",
        )
    )
    vital_bootstrap_seed = int(
        st.number_input(
            "Seed do bootstrap",
            min_value=0,
            value=42,
            step=1,
            key="vital_bootstrap_seed",
        )
    )
    if st.button("Baixar casos VitalDB", key="download_vitaldb"):
        try:
            vital_case_ids = [
                int(token.strip()) for token in vital_case_text.split(",") if token.strip()
            ]
            if not vital_case_ids:
                raise ValueError("Informe pelo menos um número de caso")
            with st.spinner("Consultando o índice e baixando os tracks selecionados..."):
                vital_paths = [
                    download_vitaldb_case(case_id, vital_data_dir)
                    for case_id in vital_case_ids
                ]
            st.success(f"{len(vital_paths)} caso(s) VitalDB disponível(is) em {vital_data_dir}")
        except Exception as error:
            st.error(f"Não foi possível baixar o VitalDB: {error}")

    vital_local = sorted(vital_data_dir.glob("vitaldb_case*.npz"))
    st.metric("Casos VitalDB locais", len(vital_local))
    if vital_local:
        st.dataframe(
            pd.DataFrame({"caso": [path.stem for path in vital_local]}),
            hide_index=True,
        )
    if vital_local and model_path.exists() and st.button(
        "Avaliar VitalDB sem retreino", key="evaluate_vitaldb"
    ):
        try:
            model, preprocess, payload = load_checkpoint(model_path)
            min_quality = float(payload.get("min_quality", DEFAULT_MIN_SIGNAL_QUALITY))
            vital_diagnostics = []
            for path in vital_local:
                case = load_case(path)
                vital_diagnostics.append(
                    {
                        "path": str(path),
                        "case_id": case.case_id,
                        **signal_diagnostics(case.eeg, preprocess),
                    }
                )
            vital_windows = load_windows(vital_local, preprocess, min_quality=min_quality)
            if not vital_windows.signals.shape[0]:
                raise ValueError("Nenhuma janela VitalDB passou pelo gate de qualidade")
            vital_prediction = predict_model(model, vital_windows.signals, device="cpu")
            vital_case_ids = vital_windows.case_ids.astype(str)
            rows = []
            for case_id in sorted(np.unique(vital_case_ids)):
                case_mask = vital_case_ids == case_id
                rows.append(
                    {
                        "caso": case_id,
                        "janelas": int(case_mask.sum()),
                        **compute_metrics(
                            vital_windows.bis[case_mask], vital_prediction[case_mask]
                        ),
                    }
                )
            aggregate_metrics = compute_metrics(vital_windows.bis, vital_prediction)
            case_bootstrap = (
                bootstrap_case_metrics(
                    vital_windows.bis,
                    vital_prediction,
                    vital_case_ids,
                    n_bootstrap=vital_bootstrap_samples,
                    seed=vital_bootstrap_seed,
                )
                if len(rows) > 1
                else {}
            )
            report = {
                "report_version": 1,
                "scope": "research_only",
                "checkpoint": str(model_path),
                "checkpoint_sha256": sha256_file(model_path),
                "files": [str(path) for path in vital_local],
                "input_files": build_file_manifest(vital_local),
                "input_diagnostics": vital_diagnostics,
                "data_handling": data_handling_policy(),
                "preprocess_config": asdict(preprocess),
                "retrained": False,
                "raw_eeg_in_report": False,
                "min_quality": min_quality,
                "n_windows": int(vital_windows.signals.shape[0]),
                "bootstrap_samples": vital_bootstrap_samples,
                "bootstrap_seed": vital_bootstrap_seed,
                "metrics": aggregate_metrics,
                "per_case": rows,
                "case_bootstrap": case_bootstrap,
            }
            st.warning("Resultado exploratório out-of-dataset; não é validação clínica.")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Janelas avaliadas", f"{report['n_windows']:,}")
            metric_columns[1].metric("MAE BIS", f"{aggregate_metrics['mae']:.2f}")
            metric_columns[2].metric("Pearson", f"{aggregate_metrics['pearson_r']:.3f}")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            if any(item["nonfinite_count"] for item in vital_diagnostics):
                st.info(
                    "A avaliação offline encontrou pontos NaN/Inf em alguns arquivos; "
                    "eles podem ser imputados somente na preparação das janelas e não "
                    "são aceitos pelo caminho ao vivo."
                )
                st.dataframe(pd.DataFrame(vital_diagnostics), hide_index=True)
            if case_bootstrap:
                st.caption(
                    "Intervalos exploratórios por bootstrap de cirurgias inteiras "
                    f"({vital_bootstrap_samples:,} reamostragens; seed {vital_bootstrap_seed})"
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"métrica": name, **values}
                            for name, values in case_bootstrap.items()
                        ]
                    ),
                    hide_index=True,
                )
            st.download_button(
                "Baixar relatório JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="brainsniffer_vitaldb_evaluation.json",
                mime="application/json",
                key="download_vitaldb_report",
            )
        except Exception as error:
            st.error(f"Não foi possível avaliar o VitalDB: {error}")

with tab_explore:
    st.subheader("Inspeção de um caso")
    local = sorted(data_dir.glob("case*.mat"))
    if not local:
        st.info("Baixe pelo menos um caso na aba Dados.")
    else:
        selected_path = st.selectbox("Caso", local, format_func=lambda path: path.stem)
        case = load_case(selected_path)
        seconds = st.slider(
            "Segundos exibidos",
            min_value=5,
            max_value=min(120, int(case.duration_seconds)),
            value=min(30, int(case.duration_seconds)),
        )
        samples = int(seconds * case.sampling_rate)
        eeg_frame = pd.DataFrame({"EEG (unidade do arquivo)": case.eeg[:samples]})
        st.line_chart(eeg_frame, height=260)
        bis_frame = pd.DataFrame(
            {"BIS": case.bis[: int(seconds / case.label_interval_seconds) + 1]}
        )
        st.line_chart(bis_frame, height=180)
        c1, c2, c3 = st.columns(3)
        c1.metric("Duração", f"{case.duration_seconds / 60:.1f} min")
        c2.metric("Amostras EEG", f"{case.eeg.size:,}")
        c3.metric("Pontos BIS", f"{case.bis.size:,}")

with tab_train:
    st.subheader("Treinar a CNN")
    local = sorted(data_dir.glob("case*.mat"))
    epochs = st.number_input(
        "Épocas",
        min_value=1,
        max_value=100,
        value=TrainingConfig.epochs,
        help="O padrão de 10 épocas corresponde ao checkpoint e ao artigo documentados.",
    )
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=128, step=8)
    min_quality = st.slider(
        "Qualidade mínima heurística",
        0.0,
        1.0,
        DEFAULT_MIN_SIGNAL_QUALITY,
        0.05,
    )
    st.write(
        "A separação é feita por caso para que janelas do mesmo paciente não caiam "
        "em conjuntos diferentes."
    )
    label_offset_seconds = st.number_input(
        "Deslocamento do rótulo BIS (s)",
        min_value=-60.0,
        max_value=60.0,
        value=0.0,
        step=0.5,
        help=(
            "Positivo associa a janela a um BIS posterior; use somente após definir "
            "o atraso do monitor no protocolo."
        ),
    )
    st.caption(
        "O padrão 0,20 remove janelas saturadas/planas. Use 0,00 somente para auditoria "
        "exploratória do arquivo bruto."
    )
    if not local:
        st.info("Baixe os casos antes de treinar.")
    elif st.button("Treinar e salvar checkpoint", type="primary"):
        with st.spinner("Preparando janelas e treinando — o primeiro ciclo pode demorar no CPU..."):
            preprocess_config = PreprocessConfig(label_offset_seconds=float(label_offset_seconds))
            windows = load_windows(local, preprocess_config, min_quality=min_quality)
            result = train_model(
                windows,
                preprocess_config=preprocess_config,
                training_config=TrainingConfig(epochs=int(epochs), batch_size=int(batch_size)),
                checkpoint_path=model_path,
                min_quality=float(min_quality),
                input_files=local,
            )
        st.success(f"Checkpoint salvo em {model_path}")
        st.json(
            {
                "dataset": result.dataset_summary,
                "split": result.split.__dict__,
                "min_quality": result.quality_threshold,
                "checkpoint_sha256": sha256_file(model_path),
                "validation": result.validation_metrics,
                "test": result.test_metrics,
                "device": result.device,
                "input_files": result.input_files,
            }
        )
        st.line_chart(pd.DataFrame(result.history).set_index("epoch"), height=220)

with tab_replay:
    st.subheader("Replay como fluxo de EEG (simulação)")
    st.caption(
        "O valor exibido é uma referência BIS estimada para pesquisa; não é diagnóstico "
        "nem instrução de anestesia."
    )
    replay_sources = sorted(data_dir.glob("case*.mat")) + sorted(
        vital_data_dir.glob("vitaldb_case*.npz")
    )
    if not model_path.exists():
        st.info("Treine um checkpoint antes do replay.")
    elif not replay_sources:
        st.info("Baixe pelo menos um caso Figshare ou VitalDB antes do replay.")
    else:
        selected_path = st.selectbox(
            "Caso/arquivo para replay",
            replay_sources,
            format_func=lambda path: path.stem,
            key="replay_case",
        )
        stride = st.slider("Passo de emissão (s)", 0.5, 5.0, 1.0, 0.5)
        replay_min_quality = st.slider(
            "Qualidade mínima para emitir BIS", 0.0, 1.0, 0.2, 0.05, key="replay_min_quality"
        )
        max_seconds = st.slider("Duração máxima do replay (s)", 10, 300, 60, 10)
        if st.button("Iniciar replay"):
            model, preprocess, _ = load_checkpoint(model_path)
            estimator = RealtimeEstimator(
                model,
                preprocess,
                stride_seconds=stride,
                min_quality=replay_min_quality,
            )
            case = load_case(selected_path)
            chunk_size = max(1, int(stride * case.sampling_rate))
            latest = st.empty()
            chart = st.empty()
            history: list[dict[str, float | str]] = []
            max_samples = min(case.eeg.size, int(max_seconds * case.sampling_rate))
            for start in range(0, max_samples, chunk_size):
                outputs = estimator.push(case.eeg[start : start + chunk_size])
                if outputs:
                    output = outputs[-1]
                    history.append(
                        {
                            "tempo_s": output.elapsed_seconds,
                            "BIS suavizado": output.smoothed_bis,
                            "qualidade": output.quality,
                        }
                    )
                    value = (
                        "ABSTAIN" if output.smoothed_bis is None else f"{output.smoothed_bis:.1f}"
                    )
                    latest.metric(
                        "Referência BIS estimada",
                        value,
                        STAGE_LABELS.get(output.stage, output.stage),
                    )
                    chart.line_chart(pd.DataFrame(history).set_index("tempo_s"), height=240)
                time.sleep(0.01)

with tab_live:
    st.subheader("Aquisição ao vivo via Lab Streaming Layer")
    st.caption(
        "O LSL é uma ponte de pesquisa; o monitor EEG precisa ter um outlet LSL "
        "ou um bridge fornecido pelo fabricante. O resultado é uma referência BIS "
        "estimada e não comanda nem recomenda anestésicos."
    )
    stream_name = st.text_input("Nome do stream (opcional)", value="")
    stream_type = st.text_input("Tipo do stream", value="EEG")
    channel_index = st.number_input("Canal", min_value=0, value=0, step=1)
    st.markdown("**Manifesto do sinal (conforme a documentação do equipamento)**")
    live_unit = st.text_input("Unidade do canal", value="", key="live_unit")
    live_channel_name = st.text_input(
        "Nome/posição do canal", value="", key="live_channel_name"
    )
    live_reference = st.text_input("Referência elétrica", value="", key="live_reference")
    live_montage = st.text_input("Montagem", value="", key="live_montage")
    with st.expander("Ficha técnica do equipamento", expanded=True):
        live_device_manufacturer = st.text_input(
            "Fabricante", value="", key="live_device_manufacturer"
        )
        live_device_model = st.text_input("Modelo", value="", key="live_device_model")
        live_firmware = st.text_input(
            "Firmware/software", value="", key="live_firmware"
        )
        live_bridge = st.text_input(
            "SDK, protocolo ou bridge", value="", key="live_bridge"
        )
        live_sampling_rate = st.number_input(
            "Taxa nominal documentada (Hz; obrigatória no gate)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="live_sampling_rate",
        )
        live_nominal_range = st.text_input(
            "Faixa nominal / saturação documentada", value="", key="live_nominal_range"
        )
        live_processing_applied = st.text_input(
            "Processamento/ganho aplicado pelo equipamento ou bridge",
            value="",
            key="live_processing_applied",
        )
    live_manifest: dict[str, object] = {
        "unit": live_unit,
        "channel_name": live_channel_name,
        "reference": live_reference,
        "montage": live_montage,
        "device_manufacturer": live_device_manufacturer,
        "device_model": live_device_model,
        "firmware": live_firmware,
        "bridge": live_bridge,
        "nominal_range": live_nominal_range,
        "processing_applied": live_processing_applied,
        "source_name": stream_name,
    }
    if live_sampling_rate > 0:
        live_manifest["sampling_rate"] = float(live_sampling_rate)
    manifest_json = json.dumps(live_manifest, ensure_ascii=False, indent=2)
    manifest_columns = st.columns(2)
    with manifest_columns[0]:
        if st.button("Validar ficha para bancada", key="validate_live_manifest"):
            intake_preview = validate_intake_metadata(live_manifest)
            if intake_preview["ready_for_bench"]:
                st.success("Ficha completa para o preflight técnico de bancada.")
            else:
                st.error(
                    "Ficha não liberada: "
                    + ", ".join(
                        [
                            *intake_preview["missing_fields"],
                            *intake_preview["compatibility_issues"],
                        ]
                    )
                )
    with manifest_columns[1]:
        st.download_button(
            "Baixar manifesto JSON",
            data=manifest_json,
            file_name="brainsniffer_stream_metadata.json",
            mime="application/json",
            key="download_live_manifest",
        )
    live_require_intake = st.checkbox(
        "Exigir ficha técnica completa antes de emitir",
        value=True,
        help="Garante que o stream foi identificado para a bancada de pesquisa.",
    )
    live_require_metadata = st.checkbox(
        "Exigir metadados completos antes de emitir",
        value=True,
        help="Impede a inferência se unidade, canal, referência ou montagem estiverem vazios.",
    )
    live_fail_on_audit = st.checkbox(
        "Rejeitar a sessão assim que a auditoria falhar",
        value=True,
        help=(
            "Em modo de bancada, transforma baixa qualidade ou timestamps inválidos "
            "em erro de sessão."
        ),
    )
    live_stale_timeout = st.number_input(
        "Timeout sem dados antes de invalidar/rejeitar (s)",
        min_value=0.0,
        value=2.0,
        step=0.5,
        help=(
            "Se o stream ficar silencioso, a última estimativa não permanece válida. "
            "Com fail-on-audit, a sessão termina com erro."
        ),
    )
    live_seconds = st.slider("Duração da captura (s)", 10, 300, 60, 10)
    live_min_quality = st.slider(
        "Qualidade mínima para emitir BIS", 0.0, 1.0, 0.2, 0.05, key="live_min_quality"
    )
    if not model_path.exists():
        st.info("Treine um checkpoint antes de conectar ao stream.")
    elif st.button("Conectar e estimar", type="primary"):
        st.session_state.pop("live_report", None)
        audit: StreamAudit | None = None
        checkpoint_sha256: str | None = None
        prediction_count = 0
        abstention_count = 0
        stale_abstention_count = 0
        prediction_qualities: list[float] = []
        intake_report: dict[str, object] = validate_intake_metadata({})
        try:
            model, preprocess, _ = load_checkpoint(model_path)
            checkpoint_sha256 = sha256_file(model_path)
            audit = StreamAudit(
                preprocess,
                min_quality=live_min_quality,
                require_metadata=live_require_metadata,
                require_timestamps=True,
            )
            source = LSLSource.connect(
                stream_name=stream_name or None,
                stream_type=stream_type,
                channel_index=int(channel_index),
            )
            audit.set_metadata(source.metadata)
            audit.set_metadata(live_manifest)
            intake_report = validate_intake_metadata(audit.report().metadata)
            if live_require_metadata and not audit.metadata_complete():
                missing = ", ".join(audit.report().metadata_missing)
                raise RuntimeError(f"metadata obrigatório incompleto: {missing}")
            if live_require_intake and not intake_report["ready_for_bench"]:
                missing = ", ".join(intake_report["missing_fields"])
                raise RuntimeError(f"ficha do equipamento incompleta: {missing}")
            estimator = RealtimeEstimator(
                model,
                preprocess,
                stride_seconds=1.0,
                min_quality=live_min_quality,
            )
            resampler = StreamingResampler(
                source.sampling_rate,
                preprocess.sampling_rate,
            )
            latest = st.empty()
            chart = st.empty()
            history: list[dict[str, float]] = []
            deadline = time.monotonic() + live_seconds
            last_data_wall = time.monotonic()
            stale_notified = False
            while time.monotonic() < deadline:
                chunk = source.read_chunk(timeout_seconds=0.2, max_samples=256)
                if chunk.samples.size == 0:
                    silence_seconds = time.monotonic() - last_data_wall
                    if silence_seconds >= live_stale_timeout and not stale_notified:
                        stale_output = estimator.mark_stale()
                        prediction_count += 1
                        abstention_count += 1
                        stale_abstention_count += 1
                        prediction_qualities.append(stale_output.quality)
                        latest.metric(
                            "Referência BIS estimada",
                            "ABSTAIN",
                            "ABSTAIN — stream sem dados",
                        )
                        if live_fail_on_audit:
                            raise RuntimeError(
                                "sem dados EEG no stream LSL por "
                                f"{silence_seconds:.2f} s; sessão rejeitada"
                            )
                        stale_notified = True
                    continue
                last_data_wall = time.monotonic()
                stale_notified = False
                audit.push(
                    chunk.samples,
                    source_rate=chunk.sampling_rate,
                    timestamps=chunk.timestamps,
                )
                if audit.report().timestamps_present is not True:
                    raise RuntimeError("timestamps obrigatórios ausentes no stream LSL")
                if live_fail_on_audit and not audit.report().ok:
                    raise RuntimeError("auditoria da sessão rejeitou o stream LSL")
                converted = resampler.process(
                    chunk.samples,
                    timestamps=chunk.timestamps,
                )
                for output in estimator.push(converted.samples, timestamps=converted.timestamps):
                    prediction_count += 1
                    abstention_count += int(output.stage == "abstain")
                    prediction_qualities.append(output.quality)
                    history.append(
                        {
                            "tempo_s": output.elapsed_seconds,
                            "BIS suavizado": output.smoothed_bis,
                            "qualidade": output.quality,
                        }
                    )
                    value = (
                        "ABSTAIN" if output.smoothed_bis is None else f"{output.smoothed_bis:.1f}"
                    )
                    latest.metric(
                        "Referência BIS estimada",
                        value,
                        STAGE_LABELS.get(output.stage, output.stage),
                    )
                    chart.line_chart(pd.DataFrame(history).set_index("tempo_s"), height=240)
            if audit.report().sample_count == 0:
                raise RuntimeError(
                    "Nenhuma amostra EEG recebida durante a captura LSL; "
                    "verifique o outlet, o nome/tipo do stream e a conexão"
                )
            if live_fail_on_audit and not audit.report().ok:
                raise RuntimeError("auditoria da sessão rejeitou o stream LSL")
            st.session_state["live_report"] = _live_session_report(
                checkpoint=model_path,
                checkpoint_sha256=checkpoint_sha256,
                audit=audit,
                preprocess_config=preprocess,
                stride_seconds=1.0,
                prediction_count=prediction_count,
                abstention_count=abstention_count,
                stale_abstention_count=stale_abstention_count,
                prediction_qualities=prediction_qualities,
                fail_on_audit=live_fail_on_audit,
                require_intake=live_require_intake,
                stale_timeout_seconds=float(live_stale_timeout),
                intake_report=intake_report,
            )
            st.success("Captura encerrada; relatório de sessão disponível abaixo.")
        except Exception as error:
            if audit is not None and checkpoint_sha256 is not None:
                st.session_state["live_report"] = _live_session_report(
                    checkpoint=model_path,
                    checkpoint_sha256=checkpoint_sha256,
                    audit=audit,
                    preprocess_config=preprocess,
                    stride_seconds=1.0,
                    prediction_count=prediction_count,
                    abstention_count=abstention_count,
                    stale_abstention_count=stale_abstention_count,
                    prediction_qualities=prediction_qualities,
                    fail_on_audit=live_fail_on_audit,
                    require_intake=live_require_intake,
                    stale_timeout_seconds=float(live_stale_timeout),
                    intake_report=intake_report,
                    error=str(error),
                )
            st.error(f"Não foi possível iniciar o stream LSL: {error}")

    live_report = st.session_state.get("live_report")
    if isinstance(live_report, dict):
        st.divider()
        st.subheader("Relatório da última sessão LSL")
        report_predictions = live_report.get("predictions", {})
        report_audit = live_report.get("audit", {})
        if isinstance(report_predictions, dict) and isinstance(report_audit, dict):
            report_columns = st.columns(4)
            report_columns[0].metric(
                "Amostras auditadas", f"{int(report_audit.get('sample_count', 0)):,}"
            )
            report_columns[1].metric(
                "Predições", f"{int(report_predictions.get('count', 0)):,}"
            )
            report_columns[2].metric(
                "Abstenções", f"{int(report_predictions.get('abstentions', 0)):,}"
            )
            report_columns[3].metric("Auditoria", "OK" if report_audit.get("ok") else "REVISAR")
        if live_report.get("status") == "error":
            st.warning(f"Sessão encerrada com erro: {live_report.get('error')}")
        st.caption(
            "O relatório contém somente metadados, métricas de qualidade e o hash do "
            "checkpoint; não contém amostras EEG."
        )
        st.download_button(
            "Baixar relatório da sessão LSL",
            data=json.dumps(live_report, ensure_ascii=False, indent=2),
            file_name="brainsniffer_lsl_session.json",
            mime="application/json",
            key="download_lsl_session_report",
        )
        with st.expander("Diagnóstico do stream"):
            st.json(report_audit)
            intake = live_report.get("intake")
            if isinstance(intake, dict):
                st.write(
                    "Ficha do equipamento:",
                    "pronta para bancada"
                    if intake.get("ready_for_bench")
                    else "incompleta",
                )
                if intake.get("missing_fields"):
                    st.write("Campos ausentes:", ", ".join(intake["missing_fields"]))

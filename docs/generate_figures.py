"""Render audited historical report snapshots as conventional Matplotlib article figures.

Historical metrics are not recomputed. Explicit --infer-trajectory runs bounded
CPU inference on the complete, preselected case19; default runs never infer,
download or train. --trajectory redraws its saved audit arrays without inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = Path(__file__).resolve().parent / "figures"

REPORT_FILES = (
    "corpus_manifest.json",
    "figshare_holdout_evaluation.json",
    "lsl_synthetic_intake_session.json",
    "lsl_synthetic_session.json",
    "mixed_fixed_figshare_holdout.json",
    "mixed_vitaldb_external.json",
    "offset_sensitivity.json",
    "vitaldb_external_validation.json",
)

PK_FILES = (
    "pk_figshare_active.json",
    "pk_figshare_mixed.json",
    "pk_vitaldb_active.json",
    "pk_vitaldb_mixed.json",
)
PK_HOLDOUTS = (
    ("Figshare", "pk_figshare_active.json", "pk_figshare_mixed.json"),
    ("VitalDB", "pk_vitaldb_active.json", "pk_vitaldb_mixed.json"),
)

MODEL_FILES = ("brainsniffer_cnn.json", "brainsniffer_corpus_fixed.json")

CALIBRATION_FILE = "calibration_analysis.json"
# Ordem de exibição (Figshare em cima, VitalDB embaixo): braço, rótulo, marcador,
# cor, estilo do IC e preenchimento do marcador. Marcadores separam CNN ativo
# (círculo), CNN misto (quadrado) e baseline espectral (diamante) em preto e branco.
CALIBRATION_ARMS = (
    ("cnn_active_figshare", "Figshare (ativo)", "o", "#000000", "-", True),
    ("cnn_mixed_figshare", "Figshare (misto)", "s", "#555555", "--", False),
    ("rf_spectral_figshare", "Figshare (RF espectral)", "D", "#666666", ":", True),
    ("cnn_active_vitaldb", "VitalDB (ativo)", "o", "#000000", "-", True),
    ("cnn_mixed_vitaldb", "VitalDB (misto)", "s", "#555555", "--", False),
    ("rf_spectral_vitaldb", "VitalDB (RF espectral)", "D", "#666666", ":", True),
)
CALIBRATION_CROSSCHECKS = (
    ("cnn_active_figshare", "figshare_holdout_evaluation.json"),
    ("cnn_mixed_figshare", "mixed_fixed_figshare_holdout.json"),
    ("cnn_active_vitaldb", "vitaldb_external_validation.json"),
    ("cnn_mixed_vitaldb", "mixed_vitaldb_external.json"),
)


def report_value(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing report field: {'.'.join(keys)}")
        value = value[key]
    return value


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_reports() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in REPORT_FILES:
        path = REPORTS / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, dict), f"{name} must contain a JSON object")
        loaded[name] = payload
    return loaded


def load_pk_reports() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in PK_FILES:
        path = REPORTS / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, dict), f"{name} must contain a JSON object")
        loaded[name] = payload
    return loaded


def audit_pk_reports(
    reports: dict[str, dict[str, Any]], pk_reports: dict[str, dict[str, Any]]
) -> None:
    """Check Pk snapshots against the audited historical aggregates."""
    pairs = (
        ("figshare_holdout_evaluation.json", "pk_figshare_active.json"),
        ("mixed_fixed_figshare_holdout.json", "pk_figshare_mixed.json"),
        ("vitaldb_external_validation.json", "pk_vitaldb_active.json"),
        ("mixed_vitaldb_external.json", "pk_vitaldb_mixed.json"),
    )
    for hist_name, pk_name in pairs:
        hist = reports[hist_name]
        pk_report = pk_reports[pk_name]
        ensure(pk_report.get("scope") == "research_only", f"unsafe scope in {pk_name}")
        ensure(pk_report.get("retrained") is False, f"retrain flag in {pk_name}")
        hist_metrics = metrics(hist)
        overall = pk_report.get("metrics", {})
        hist_n = float(hist_metrics.get("n", 0))
        ensure(float(pk_report.get("n_windows", 0)) == hist_n, f"n mismatch in {pk_name}")
        ensure(float(overall.get("n", 0)) == hist_n, f"overall n mismatch in {pk_name}")
        for key in ("mae", "rmse", "bias", "pearson_r"):
            a = float(hist_metrics[key])
            b = float(overall[key])
            ensure(abs(a - b) < 1e-4, f"{key} drift in {pk_name}: {a} vs {b}")
        pk_value = float(pk_report["pk"])
        ensure(0.0 <= pk_value <= 1.0, f"Pk outside [0, 1] in {pk_name}")
        interval = pk_report.get("pk_bootstrap", {})
        ensure(bool(interval), f"missing Pk bootstrap in {pk_name}")
        lower = float(interval["lower_95"])
        upper = float(interval["upper_95"])
        ensure(lower <= pk_value <= upper, f"Pk outside its interval in {pk_name}")
        ensure(
            int(pk_report.get("bootstrap_samples", 0)) == 1000,
            f"unexpected Pk bootstrap count in {pk_name}",
        )
        ensure(int(pk_report.get("bootstrap_seed", -1)) == 42, f"seed drift in {pk_name}")


def audit_calibration(calibration: dict[str, Any], reports: dict[str, dict[str, Any]]) -> None:
    """Check the calibration snapshot against the audited historical aggregates."""
    ensure(calibration.get("scope") == "research_only", "unsafe calibration scope")
    ensure(calibration.get("retrained") is False, "calibration unexpectedly retrained")
    ensure(
        int(calibration.get("bootstrap_samples", 0)) == 1000,
        "unexpected calibration bootstrap count",
    )
    ensure(int(calibration.get("bootstrap_seed", -1)) == 42, "calibration seed drift")
    arms = calibration.get("arms")
    ensure(isinstance(arms, dict), "calibration arms missing")
    ensure(
        set(arms) == {key for key, *_ in CALIBRATION_ARMS},
        "calibration arms changed",
    )
    for key, *_ in CALIBRATION_ARMS:
        arm = arms[key]
        line = arm["calibration_line"]
        ensure(
            line.get("x") == "referência BIS" and line.get("y") == "predição",
            f"calibration axes changed in {key}",
        )
        ensure(int(line.get("bootstrap_samples", 0)) == 1000,
               f"unexpected slope bootstrap count in {key}")
        ensure(int(line.get("bootstrap_seed", -1)) == 42, f"slope seed drift in {key}")
        slope = float(line["slope"])
        ensure(
            float(line["slope_ci95_lower"]) <= slope <= float(line["slope_ci95_upper"]),
            f"slope outside its interval in {key}",
        )
        intercept = float(line["intercept"])
        ensure(
            float(line["intercept_ci95_lower"]) <= intercept
            <= float(line["intercept_ci95_upper"]),
            f"intercept outside its interval in {key}",
        )
        icc = float(arm["icc_2_1_absolute_agreement"])
        ensure(-1.0 <= icc <= 1.0, f"ICC outside [-1, 1] in {key}")
        fraction = float(arm["fraction_abs_error_le_10"])
        ensure(0.0 <= fraction <= 1.0, f"error fraction outside [0, 1] in {key}")
        ensure(
            abs(float(arm["bias"]) - float(arm["bland_altman"]["bias"])) < 1e-9,
            f"bias drift in {key}",
        )
        ensure(int(arm["n"]) > 0 and int(arm["n_cases"]) > 0, f"missing sample size in {key}")
    for key, name in CALIBRATION_CROSSCHECKS:
        arm = arms[key]
        historical = metrics(reports[name])
        ensure(int(arm["n"]) == int(historical["n"]), f"calibration n drift in {key}")
        ensure(
            abs(float(arm["mae"]) - float(historical["mae"])) < 1e-4,
            f"calibration mae drift in {key}",
        )


def load_calibration(
    reports: dict[str, dict[str, Any]], attempts: int = 10, delay_seconds: float = 3.0
) -> dict[str, Any]:
    """Load the calibration snapshot, tolerating an in-flight rewrite."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            path = REPORTS / CALIBRATION_FILE
            payload = json.loads(path.read_text(encoding="utf-8"))
            ensure(isinstance(payload, dict), f"{CALIBRATION_FILE} must contain a JSON object")
            audit_calibration(payload, reports)
            return payload
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                print(f"retrying {CALIBRATION_FILE} in {delay_seconds:g}s ({error})")
                time.sleep(delay_seconds)
    raise ValueError(f"invalid {CALIBRATION_FILE} after {attempts} attempts: {last_error}")


def audit_reports(reports: dict[str, dict[str, Any]]) -> None:
    """Reject inconsistent snapshots before any chart is rendered."""

    evaluation_names = (
        "corpus_manifest.json",
        "figshare_holdout_evaluation.json",
        "mixed_fixed_figshare_holdout.json",
        "mixed_vitaldb_external.json",
        "offset_sensitivity.json",
        "vitaldb_external_validation.json",
    )
    for name in evaluation_names:
        ensure(reports[name].get("scope") == "research_only", f"unsafe scope in {name}")

    for name in ("lsl_synthetic_intake_session.json", "lsl_synthetic_session.json"):
        scope = report_value(reports[name], "scope")
        ensure(scope.get("intended_use") == "research_only", f"unsafe LSL scope in {name}")
        ensure(scope.get("clinical_decision_support") is False, f"clinical flag set in {name}")
        ensure(scope.get("controls_anesthetic_delivery") is False, f"control flag set in {name}")

    figshare = reports["figshare_holdout_evaluation.json"]
    mixed_figshare = reports["mixed_fixed_figshare_holdout.json"]
    offset = reports["offset_sensitivity.json"]
    vitaldb = reports["vitaldb_external_validation.json"]
    mixed_vitaldb = reports["mixed_vitaldb_external.json"]
    corpus = reports["corpus_manifest.json"]

    figshare_cases = report_value(figshare, "test_cases")
    ensure(figshare_cases == report_value(mixed_figshare, "case_ids"), "Figshare holdouts differ")
    ensure(figshare_cases == report_value(offset, "test_cases"), "offset split differs")
    ensure(
        report_value(figshare, "n_test_windows") == report_value(mixed_figshare, "n_windows"),
        "Figshare window counts differ",
    )
    ensure(
        report_value(vitaldb, "case_ids") == report_value(mixed_vitaldb, "case_ids"),
        "VitalDB holdouts differ",
    )
    ensure(
        report_value(vitaldb, "n_windows") == report_value(mixed_vitaldb, "n_windows"),
        "VitalDB window counts differ",
    )

    active_per_case = {row["case_id"]: row for row in report_value(vitaldb, "per_case")}
    mixed_per_case = {row["case_id"]: row for row in report_value(mixed_vitaldb, "per_case")}
    ensure(active_per_case.keys() == mixed_per_case.keys(), "VitalDB per-case IDs differ")
    for case_id in active_per_case:
        ensure(
            active_per_case[case_id]["n_windows"] == mixed_per_case[case_id]["n_windows"],
            f"VitalDB per-case grain differs for {case_id}",
        )

    frozen_ids = [case["case_id"] for case in report_value(corpus, "frozen_external_cases")]
    ensure(set(frozen_ids) == set(report_value(vitaldb, "case_ids")), "frozen holdout IDs differ")

    preprocessing = report_value(corpus, "preprocess_config")
    keys = (
        "sampling_rate",
        "window_seconds",
        "lowcut_hz",
        "highcut_hz",
        "notch_hz",
        "causal",
        "label_offset_seconds",
    )
    for name in evaluation_names[1:]:
        candidate = report_value(reports[name], "preprocess_config")
        ensure(
            all(candidate[key] == preprocessing[key] for key in keys),
            f"preprocessing mismatch in {name}",
        )

    bootstrap_names = (
        "figshare_holdout_evaluation.json",
        "mixed_fixed_figshare_holdout.json",
        "vitaldb_external_validation.json",
        "mixed_vitaldb_external.json",
    )
    settings = {
        (
            report_value(reports[name], "bootstrap_samples"),
            report_value(reports[name], "bootstrap_seed"),
        )
        for name in bootstrap_names
    }
    ensure(len(settings) == 1, "bootstrap settings differ between reports")
    ensure(offset.get("retained_model_and_weights") is True, "offset model was not retained")
    ensure(offset.get("retained_split_by_case") is True, "offset split was not retained")
    ensure(offset.get("retrained") is False, "offset analysis unexpectedly retrained the model")

    for name, count_key, metrics_key in (
        ("mixed_fixed_figshare_holdout.json", "n_windows", "metrics"),
        ("vitaldb_external_validation.json", "n_windows", "metrics"),
        ("mixed_vitaldb_external.json", "n_windows", "metrics"),
    ):
        expected = float(report_value(reports[name], count_key))
        observed = float(report_value(reports[name], metrics_key, "n"))
        ensure(expected == observed, f"metric n differs from n_windows in {name}")


# Physical width matches the SBC text block exactly (A4 21 cm - 3 cm - 3 cm = 15 cm),
# so TeX inserts the figures at scale 1 and the point sizes below survive unscaled.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
WIDTH = 15 / 2.54
COLORS = ("#000000", "#555555")
MARKERS = ("o", "s")
LABELS = ("Ativo", "Misto fixo")
HOLDOUTS = (
    "figshare_holdout_evaluation.json",
    "mixed_fixed_figshare_holdout.json",
    "vitaldb_external_validation.json",
    "mixed_vitaldb_external.json",
)


def save_figure(fig, name, sources):
    source_text = "; ".join(sources)
    fig.savefig(
        OUTPUT / f"{name}.png", dpi=300, metadata={"Sources": source_text, "Scope": "research_only"}
    )
    fig.savefig(
        OUTPUT / f"{name}.pdf",
        metadata={
            "Subject": source_text,
            "CreationDate": None,
            "ModDate": None,
            "Creator": "BrainSniffer / Matplotlib",
        },
    )
    plt.close(fig)


def metrics(report):
    return report.get("recomputed_test_metrics", report.get("metrics"))


def point(ax, x, y, model, **kwargs):
    return ax.plot(
        x,
        y,
        marker=MARKERS[model],
        color=COLORS[model],
        markerfacecolor=COLORS[model] if model == 0 else "white",
        markersize=5,
        linestyle="none",
        **kwargs,
    )


def figure_comparison(reports):
    fig = plt.figure(figsize=(WIDTH, 4.4), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.35))
    for i, (metric, title, limits) in enumerate(
        (
            ("mae", "(a) MAE agregado (pontos BIS)", (0, 16)),
            ("pearson_r", "(b) Pearson r agregado", (-1, 1)),
        )
    ):
        ax = fig.add_subplot(grid[i, 0])
        for source in range(2):
            vals = [metrics(reports[HOLDOUTS[2 * source + m]])[metric] for m in range(2)]
            ax.plot(vals, [source, source], color=".55", linewidth=0.8)
            for model, val in enumerate(vals):
                point(ax, val, source, model)
        ax.set(
            yticks=[0, 1],
            yticklabels=["Figshare", "VitalDB"],
            xlim=limits,
            ylim=(1.5, -0.5),
            title=title,
        )
        ax.grid(axis="x", color=".9", linewidth=0.5)
    ax = fig.add_subplot(grid[:, 1])
    active = {r["case_id"]: r for r in reports[HOLDOUTS[2]]["per_case"]}
    mixed = {r["case_id"]: r for r in reports[HOLDOUTS[3]]["per_case"]}
    ids = sorted(active, key=lambda k: mixed[k]["mae"] - active[k]["mae"])
    handles = []
    for y, key in enumerate(ids):
        vals = [active[key]["mae"], mixed[key]["mae"]]
        ax.plot(vals, [y, y], color=".55", linewidth=0.8)
        for model, val in enumerate(vals):
            drawn = point(ax, val, y, model, label=LABELS[model] if y == 0 else None)
            if y == 0:
                handles.extend(drawn)
    ax.set(
        yticks=range(len(ids)),
        yticklabels=[k.removeprefix("vitaldb_") for k in ids],
        ylim=(len(ids) - 0.4, -0.6),
        xlim=(0, 24),
        xticks=[0, 6, 12, 18, 24],
        xlabel="MAE (pontos BIS)",
        title="(c) VitalDB: 15 casos pareados",
    )
    ax.grid(axis="x", color=".9", linewidth=0.5)
    fig.legend(
        handles=handles,
        labels=list(LABELS),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save_figure(fig, "comparison", [f"reports/{n}" for n in HOLDOUTS])


def figure_bootstrap(reports):
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.8), layout="constrained")
    labels = ["Figshare · ativo", "Figshare · misto", "VitalDB · ativo", "VitalDB · misto"]
    for ax, metric, title, limits in zip(
        axes,
        ("pearson_r", "mae"),
        ("(a) Pearson r", "(b) MAE (pontos BIS)"),
        ((-1, 1), (0, 16)),
        strict=True,
    ):
        for y, name in enumerate(HOLDOUTS):
            report = reports[name]
            ci = report["case_bootstrap"][metric]
            lo, hi = ci["lower_95"], ci["upper_95"]
            observed = metrics(report)[metric]
            ensure(limits[0] <= lo <= hi <= limits[1], f"interval outside axis: {name}")
            ax.hlines(
                y, lo, hi, color=COLORS[y % 2], linestyles="solid" if y % 2 == 0 else "dashed"
            )
            ax.plot([lo, hi], [y, y], "|", color=COLORS[y % 2], markersize=6)
            point(ax, observed, y, y % 2)
        ax.set(yticks=range(4), yticklabels=labels, ylim=(3.6, -0.6), xlim=limits, title=title)
        ax.grid(axis="x", color=".9", linewidth=0.5)
        if metric == "pearson_r":
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.axvline(0, color=".5", linewidth=0.7)
        else:
            ax.set_xticks([0, 4, 8, 12, 16])
    save_figure(fig, "bootstrap_intervals", [f"reports/{n}" for n in HOLDOUTS])


def figure_offset(reports):
    rows = reports["offset_sensitivity.json"]["results"]
    offsets = [r["offset_seconds"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), layout="constrained")
    for i, (ax, metric, title, limits) in enumerate(
        zip(
            axes,
            ("mae", "pearson_r"),
            ("(a) MAE (pontos BIS)", "(b) Pearson r"),
            ((6.7, 7.3), (0.76, 0.805)),
            strict=True,
        )
    ):
        ax.plot(
            offsets,
            [r["metrics"][metric] for r in rows],
            marker=MARKERS[i],
            color=COLORS[i],
            linestyle="-" if i == 0 else "--",
            markerfacecolor=COLORS[i] if i == 0 else "white",
            markersize=4,
        )
        ax.axvline(0, color=".5", linewidth=0.7, linestyle=":")
        ax.set(
            xlim=(-22, 22),
            ylim=limits,
            xticks=[-20, -10, 0, 10, 20],
            xlabel="Offset do rótulo BIS (s)",
            title=title,
        )
        ax.grid(axis="y", color=".9", linewidth=0.5)
    save_figure(fig, "offset_sensitivity", ["reports/offset_sensitivity.json"])


def figure_pipeline(reports):
    corpus = reports["corpus_manifest.json"]
    models = [json.loads((ROOT / "models" / n).read_text()) for n in MODEL_FILES]
    active, fixed = models
    for model, report in zip(models, (reports[HOLDOUTS[0]], reports[HOLDOUTS[1]]), strict=True):
        ensure(model["checkpoint_sha256"] == report["checkpoint_sha256"], "checkpoint mismatch")
    splits = [
        [len(m["split"][k]) for k in ("train_cases", "validation_cases", "test_cases")]
        for m in models
    ]
    ensure(splits == [[13, 5, 5], [16, 6, 6]], "historical split changed")
    ensure(fixed["dataset_summary"]["n_groups"] == 28, "fixed group count changed")
    ensure(corpus["summary"]["eligible_training_cases"] == 33, "eligible pool changed")
    ensure(corpus["summary"]["eligible_training_windows"] == 55471, "pool windows changed")
    ensure(active["dataset_summary"]["n_cases"] == 23, "active case count changed")
    ensure(len(reports[HOLDOUTS[0]]["test_cases"]) == 5, "Figshare holdout count changed")
    ensure(len(reports[HOLDOUTS[2]]["case_ids"]) == 15, "VitalDB holdout count changed")
    fixed_groups = set().union(*(set(v) for v in fixed["split"].values()))
    ensure(len(fixed_groups) == 28, "fixed splits overlap")
    ensure(sum(g.startswith("figshare:") for g in fixed_groups) == 18, "fixed sources changed")
    ensure(sum(g.startswith("vitaldb:") for g in fixed_groups) == 10, "fixed sources changed")
    ensure(
        sum(g.startswith("figshare:") for g in fixed["split"]["train_cases"]) == 12,
        "fixed Figshare training count changed",
    )
    ensure(
        sum(g.startswith("vitaldb:") for g in fixed["split"]["train_cases"]) == 4,
        "fixed VitalDB training count changed",
    )
    ensure(
        not fixed_groups.intersection(
            f"figshare:case:{k}" for k in reports[HOLDOUTS[0]]["test_cases"]
        ),
        "historical Figshare holdout entered fixed development",
    )
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, text, width=0.43, height=0.17):
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor="white",
                edgecolor=".2",
                linewidth=0.7,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=9, linespacing=1.3)

    def arrow(start, end):
        ax.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": ".25", "lw": 0.8}
        )

    box(
        0.5,
        0.90,
        "Arquivos retrospectivos: 33 casos / 55.471 janelas\n23 Figshare + 10 VitalDB elegíveis",
        width=0.65,
        height=0.15,
    )
    box(0.25, 0.65, "Ativo: 23 casos Figshare\nTreino / validação / teste: 13 / 5 / 5")
    box(0.75, 0.65, "Misto fixo: 28 grupos (18 + 10)\nTreino / validação / teste: 16 / 6 / 6")
    arrow((0.36, 0.825), (0.25, 0.735))
    arrow((0.64, 0.825), (0.75, 0.735))
    box(0.25, 0.39, "CNN ativa\nAjuste: 13 Figshare", height=0.15)
    box(0.75, 0.39, "CNN mista fixa\nAjuste: 12 Figshare + 4 VitalDB", height=0.15)
    arrow((0.25, 0.565), (0.25, 0.465))
    arrow((0.75, 0.565), (0.75, 0.465))
    box(
        0.5,
        0.12,
        "Avaliação dos dois checkpoints (sem ajuste)\n"
        "5 casos Figshare históricos + 15 VitalDB congelados",
        width=0.86,
        height=0.17,
    )
    arrow((0.25, 0.315), (0.35, 0.205))
    arrow((0.75, 0.315), (0.65, 0.205))
    save_figure(
        fig, "pipeline", ["reports/corpus_manifest.json"] + [f"models/{n}" for n in MODEL_FILES]
    )


TRAJECTORY = ROOT / "tmp/pdfs/trajectory-audit"


def infer_trajectory():
    """No case/segment search: case19 fixed before observing predictions."""
    from dataclasses import asdict

    import numpy as np
    import torch

    from brainsniffer.config import DEFAULT_MIN_SIGNAL_QUALITY
    from brainsniffer.data.mat_reader import load_case
    from brainsniffer.data.preprocess import make_windows
    from brainsniffer.pipeline.training import load_checkpoint, predict_model, sha256_file

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source = ROOT / "data/raw/case19.mat"
    checkpoint = ROOT / "models/brainsniffer_cnn.pt"
    model, config, payload = load_checkpoint(checkpoint, device="cpu")
    ensure("case19" in payload["split"]["test_cases"], "case19 is not holdout")
    expected = next(r for r in payload["input_files"] if Path(r["path"]).name == source.name)
    ensure(sha256_file(source) == expected["sha256"], "case19 input hash mismatch")
    case = load_case(source)
    ensure(config.label_offset_seconds == 0, "illustration requires unchanged zero offset")
    windows = make_windows(case, config, min_quality=DEFAULT_MIN_SIGNAL_QUALITY)
    ensure(len(windows.bis) > 0, "no accepted windows")
    prediction = predict_model(model, windows.signals, device="cpu", batch_size=64)
    # Full reference grid, including invalid labels and windows rejected by quality.
    # NaNs remain NaNs: never connect across omitted windows by compressing time.
    times = np.arange(case.bis.size) * case.label_interval_seconds
    reference = case.bis.astype(float).copy()
    reference[~np.isfinite(reference) | (reference < 0) | (reference > 100)] = np.nan
    raw = np.full(times.shape, np.nan)
    indices = np.rint((windows.start_seconds + config.label_offset_seconds)
                      / case.label_interval_seconds).astype(int)
    raw[indices] = prediction
    ensure(np.allclose(reference[indices], windows.bis), "target alignment mismatch")
    error = prediction.astype(float) - windows.bis
    TRAJECTORY.mkdir(parents=True, exist_ok=True)
    np.savez(TRAJECTORY / "case19.npz", reference_seconds=times, reference_bis=reference,
             cnn_raw_bis=raw, accepted_start_seconds=windows.start_seconds,
             available_after_seconds=windows.start_seconds + config.window_seconds,
             quality=windows.quality)
    np.savetxt(TRAJECTORY / "case19.csv", np.column_stack((times, reference, raw)),
               delimiter=",", header="reference_seconds,reference_bis,cnn_raw_bis", comments="")
    audit = {
        "scope": "research_only", "case": "case19", "selection":
        "Fixed a priori for consistency with the coordinator's existing smoke demo; "
        "complete recording, no performance-based case or segment search.",
        "source_sha256": sha256_file(source), "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess_config": asdict(config), "min_quality": DEFAULT_MIN_SIGNAL_QUALITY,
        "device": "cpu", "threads": 2, "batch_size": 64, "torch": str(torch.__version__),
        "numpy": np.__version__, "n_windows": len(prediction),
        "duration_seconds": case.duration_seconds, "reference_points": len(times),
        "missing_reference_points": int(np.isnan(reference).sum()),
        "missing_prediction_points": int(np.isnan(raw).sum()),
        "complete_eeg_windows": (case.eeg.size - config.window_samples)
        // config.window_samples + 1,
        "reference_without_complete_eeg_window": int(np.sum(
            times + config.window_seconds > case.duration_seconds)),
        "rejected_quality_windows": int(np.sum(np.isnan(raw[
            times + config.window_seconds <= case.duration_seconds]))),
        "raw_eeg_nonfinite": int((~np.isfinite(case.eeg)).sum()),
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()), "pearson_r": float(np.corrcoef(windows.bis, prediction)[0, 1]),
        "alignment": "Offline target=start+offset rounded to BIS grid; availability=start+5s "
        "plus processing, not measured replay latency. No shifting, no EWMA.",
        "source_code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            ROOT / "src/brainsniffer/data/preprocess.py",
            ROOT / "src/brainsniffer/pipeline/training.py",
            ROOT / "src/brainsniffer/data/mat_reader.py")},
    }
    (TRAJECTORY / "case19.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def figure_trajectory():
    import numpy as np

    with np.load(TRAJECTORY / "case19.npz", allow_pickle=False) as saved:
        time = saved["reference_seconds"] / 60
        reference, raw = saved["reference_bis"], saved["cnn_raw_bis"]
    audit = json.loads((TRAJECTORY / "case19.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    axes[0].plot(time, reference, color="black", linestyle="-",
                 label="BIS referência (linha contínua)", linewidth=1.0)
    axes[0].plot(time, raw, color="black", linestyle="--", dashes=(4, 2),
                 label="CNN ativa bruta (linha tracejada)", linewidth=1.0)
    axes[0].set(ylabel="Índice (pontos BIS)", ylim=(0, 100),
                title="Figshare case19 · gravação completa · comparação offline")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(time, raw - reference, color="black", linestyle="-", linewidth=0.7)
    axes[1].axhline(0, color=".4", linewidth=0.7)
    axes[1].set(ylabel="Erro (pontos BIS)", xlabel="Tempo da referência desde o início (min)")
    for ax in axes:
        ax.set_xlim(0, max(audit["duration_seconds"] / 60, time[-1]))
        ax.grid(color=".9", linewidth=0.5)
    save_figure(fig, "bis_trajectory", ["data/raw/case19.mat", "models/brainsniffer_cnn.pt",
                                      "tmp/pdfs/trajectory-audit/case19.json"])


def figure_pk(pk_reports: dict[str, dict[str, Any]]) -> None:
    """Prediction probability Pk with case-cluster intervals, ativo vs misto."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2), layout="constrained")
    benchmarks = [name for name, _, _ in PK_HOLDOUTS]
    for model in range(2):
        values: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for _, active_name, mixed_name in PK_HOLDOUTS:
            report = pk_reports[(active_name, mixed_name)[model]]
            interval = report["pk_bootstrap"]
            values.append(float(report["pk"]))
            lowers.append(float(interval["lower_95"]))
            uppers.append(float(interval["upper_95"]))
        for row, (value, low, high) in enumerate(
            zip(values, lowers, uppers, strict=True)
        ):
            ensure(0.0 <= low <= value <= high <= 1.0, f"Pk interval outside [0, 1]: {value}")
            y = row + (model - 0.5) * 0.17
            ax.hlines(
                y,
                low,
                high,
                color=COLORS[model],
                linewidth=1.4,
                linestyles="solid" if model == 0 else "dashed",
            )
            point(ax, value, y, model, label=LABELS[model] if row == 0 else None)
    ax.axvline(0.5, color=".45", linewidth=0.7, linestyle=":")
    ax.text(
        0.505,
        -0.5,
        "chance 0,5",
        fontsize=7,
        color=".35",
        ha="left",
        va="center",
    )
    ax.set(
        yticks=range(len(benchmarks)),
        yticklabels=benchmarks,
        ylim=(len(benchmarks) - 0.55, -0.6),
        xlim=(0.4, 1.0),
        xticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        xlabel="$P_K$ (1 = ordem perfeita · 0,5 = chance)",
    )
    ax.grid(axis="x", color=".9", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "pk_prediction", [f"reports/{n}" for n in PK_FILES])


def figure_corpus_panels(reports):
    """Composição do corpus e mapa de qualidade, em duas colunas legíveis."""
    corpus = reports["corpus_manifest.json"]
    summary = corpus.get("summary", {})
    ensure(isinstance(summary, dict), "corpus summary missing")
    cases = [record for record in corpus.get("cases", []) if isinstance(record, dict)]
    ensure(bool(cases), "corpus case list missing")
    ensure(
        len(cases) == int(summary.get("total_files", -1)),
        "corpus case count differs from total_files",
    )
    roles = {record.get("role") for record in cases}
    ensure(
        roles <= {"development_pool", "frozen_external"},
        f"unexpected corpus roles: {sorted(roles - {'development_pool', 'frozen_external'})}",
    )
    statuses = {record.get("quality_status") for record in cases}
    ensure(
        statuses <= {"include", "quarantine"},
        f"unexpected quality statuses: {sorted(statuses - {'include', 'quarantine'})}",
    )

    sources = ("figshare", "vitaldb")
    names = {"figshare": "Figshare", "vitaldb": "VitalDB"}
    eligible: list[int] = []
    quarantined: list[int] = []
    frozen: list[int] = []
    frozen_failing: list[int] = []
    for source in sources:
        selected = [record for record in cases if record.get("source_key") == source]
        ensure(bool(selected), f"corpus source missing: {source}")
        external = [record for record in selected if record.get("role") == "frozen_external"]
        eligible.append(
            sum(
                1
                for record in selected
                if record.get("role") == "development_pool"
                and record.get("quality_status") == "include"
            )
        )
        quarantined.append(
            sum(
                1
                for record in selected
                if record.get("role") == "development_pool"
                and record.get("quality_status") == "quarantine"
            )
        )
        frozen.append(len(external))
        frozen_failing.append(
            sum(1 for record in external if record.get("quality_status") != "include")
        )
    totals = [
        part_eligible + part_quarantine + part_frozen
        for part_eligible, part_quarantine, part_frozen in zip(
            eligible, quarantined, frozen, strict=True
        )
    ]
    # Categorias mutuamente exclusivas derivadas de role/quality_status, pois o
    # source_summary do manifesto soma os congelados em quarantined_cases.
    ensure(
        sum(totals) == int(summary.get("total_files", -1)),
        "corpus composition differs from total_files",
    )
    ensure(
        sum(eligible) == int(summary.get("eligible_training_cases", -1)),
        "eligible training cases differ",
    )
    ensure(
        sum(quarantined) == int(summary.get("quarantined_development_cases", -1)),
        "development quarantine differs",
    )
    ensure(
        sum(frozen) == int(summary.get("frozen_external_cases", -1)),
        "frozen external cases differ",
    )
    ensure(
        sum(eligible) + sum(quarantined) == int(summary.get("development_cases", -1)),
        "development pool size differs",
    )
    source_summary = summary.get("source_summary", {})
    ensure(
        isinstance(source_summary, dict) and bool(source_summary),
        "corpus source summary missing",
    )
    for index, source in enumerate(sources):
        expected = source_summary.get(source, {})
        ensure(isinstance(expected, dict), f"missing source summary: {source}")
        ensure(
            int(expected.get("cases", -1)) == totals[index],
            f"case total differs for {source}",
        )
        ensure(
            int(expected.get("eligible_cases", -1)) == eligible[index],
            f"eligible cases differ for {source}",
        )
        ensure(
            int(expected.get("frozen_external_cases", -1)) == frozen[index],
            f"frozen external cases differ for {source}",
        )
        ensure(
            int(expected.get("quarantined_cases", -1))
            == quarantined[index] + frozen_failing[index],
            f"quarantine accounting differs for {source}",
        )

    fig, grid = plt.subplots(1, 2, figsize=(WIDTH, 4.0), layout="constrained")
    ax = grid[0]
    positions = list(range(len(sources)))
    ax.bar(
        positions,
        eligible,
        0.5,
        label="Elegíveis para treino (sólido)",
        color="black",
        edgecolor="black",
    )
    ax.bar(
        positions,
        quarantined,
        0.5,
        label="Quarentena de desenvolvimento (hachurado)",
        color="white",
        edgecolor="black",
        hatch="///",
        bottom=eligible,
    )
    frozen_bottom = [
        part_eligible + part_quarantine
        for part_eligible, part_quarantine in zip(eligible, quarantined, strict=True)
    ]
    ax.bar(
        positions,
        frozen,
        0.5,
        label="Congelados, fora do pool (quadriculado)",
        color="#BBBBBB",
        edgecolor="black",
        hatch="xx",
        bottom=frozen_bottom,
    )
    for position, total in enumerate(totals):
        ax.text(position, total + 1.2, f"{total} casos", ha="center", fontsize=8, color=".3")
    frozen_total = sum(frozen)
    frozen_failing_total = sum(frozen_failing)
    if frozen_total > 0 and frozen_failing_total > 0:
        anchor = max(range(len(sources)), key=lambda index: frozen[index])
        ensure(frozen[anchor] > 0, "frozen annotation without frozen cases")
        ax.text(
            anchor,
            frozen_bottom[anchor] + frozen[anchor] / 2,
            f"{frozen_failing[anchor]} de {frozen[anchor]}\nreprovam\nnos gates",
            ha="center",
            va="center",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85},
        )
    ax.set(
        xticks=positions,
        xticklabels=[names.get(source, source) for source in sources],
        title=f"(a) Composição do corpus ({sum(totals)} casos)",
        ylabel="Casos",
        ylim=(0, max(totals) * 1.3),
    )
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", borderaxespad=0.6)

    ax = grid[1]
    records = [record for record in cases if isinstance(record.get("signal"), dict)]
    gate = corpus.get("quality_config", {})
    min_finite = 90.0
    if isinstance(gate, dict):
        min_finite = float(gate.get("min_finite_fraction", 0.9)) * 100
    styles = (
        ("include", "black", "o", "Elegível (círculo preenchido)", True),
        ("quarantine", "black", "x", "Quarentena (x)", False),
        ("exclude", ".55", "d", "Excluído (losango aberto)", False),
    )
    for status, color, marker, label, filled in styles:
        selected = [record for record in records if record.get("quality_status") == status]
        if not selected:
            continue
        xs = [
            float(record["signal"].get("finite_fraction", float("nan"))) * 100
            for record in selected
        ]
        ys = [
            float(record.get("windows", {}).get("accepted_fraction", float("nan"))) * 100
            for record in selected
        ]
        ax.plot(
            xs,
            ys,
            marker=marker,
            linestyle="none",
            color=color,
            markersize=5,
            label=label,
            markerfacecolor=color if filled else "none",
        )
    ax.axvline(min_finite, color="black", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(
        min_finite - 2,
        6,
        f"gate {min_finite:.0f}%",
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
    )
    ax.set(
        title="(b) Qualidade: finitude x janelas",
        xlabel="Amostras EEG finitas (%)",
        ylabel="Janelas aceitas (%)",
        xlim=(0, 104),
        ylim=(0, 104),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left", borderaxespad=0.6)

    for axis in grid:
        axis.grid(color=".9", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure(fig, "corpus_panels", ["reports/corpus_manifest.json"])


def figure_training_panels(reports, calibration):
    """Histórico de treino do checkpoint ativo e calibração medida por braço."""
    model = json.loads((ROOT / "models" / MODEL_FILES[0]).read_text())
    history = model.get("history", [])
    ensure(isinstance(history, list) and bool(history), "training history missing")
    arms = calibration["arms"]
    ensure(set(arms) == {key for key, *_ in CALIBRATION_ARMS}, "calibration arms changed")

    fig, grid = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 3.8),
        layout="constrained",
        gridspec_kw={"width_ratios": (1, 1.25)},
    )

    ax = grid[0]
    rows = [row for row in history if isinstance(row, dict)]
    epochs = [float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]
    for key, label, marker, color, style in (
        ("train_loss", "Loss treino", "o", COLORS[0], "-"),
        ("validation_mae", "MAE validação", "s", COLORS[1], "--"),
        ("validation_rmse", "RMSE validação", "^", ".35", ":"),
    ):
        values = [float(row[key]) for row in rows if key in row]
        if len(values) != len(epochs):
            continue
        ax.plot(
            epochs,
            values,
            marker=marker,
            color=color,
            linewidth=1.2,
            label=label,
            linestyle=style,
        )
    ax.set(
        title="(a) Histórico de treino e validação",
        xlabel="Época",
        ylabel="Loss / pontos BIS",
        xticks=epochs,
        ylim=(0, max(float(row.get("validation_rmse", 0)) for row in rows) * 1.25),
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right", borderaxespad=0.6)

    # Dot-and-whisker da inclinação de calibração (IC 95% por bootstrap por caso).
    # x = 1 é identidade e x = 0 é ausência de calibração; à direita, ICC(2,1)
    # absoluto e a fração de janelas com |erro| <= 10 pontos BIS.
    ax = grid[1]
    annotation_x = 1.07
    for row, (key, label, marker, color, style, filled) in enumerate(CALIBRATION_ARMS):
        arm = arms[key]
        line = arm["calibration_line"]
        slope = float(line["slope"])
        low = float(line["slope_ci95_lower"])
        upper = float(line["slope_ci95_upper"])
        ensure(low <= slope <= upper, f"slope outside its interval in {key}")
        ax.hlines(row, low, upper, color=color, linewidth=1.3, linestyles=style)
        ax.plot(
            [low, upper],
            [row, row],
            marker="|",
            color=color,
            linestyle="none",
            markersize=5,
        )
        ax.plot(
            slope,
            row,
            marker=marker,
            color=color,
            markerfacecolor=color if filled else "white",
            markersize=5.5,
            linestyle="none",
        )
        icc = float(arm["icc_2_1_absolute_agreement"])
        fraction = float(arm["fraction_abs_error_le_10"]) * 100.0
        ax.text(
            annotation_x,
            row,
            f"ICC {icc:.2f}".replace(".", ",")
            + "\n"
            + f"{fraction:.1f}".replace(".", ",")
            + "% ≤10",
            fontsize=7,
            color=".25",
            ha="left",
            va="center",
            linespacing=1.25,
        )
    separator = len(CALIBRATION_ARMS) / 2 - 0.5
    ax.axhline(separator, color=".85", linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=".45", linewidth=0.8, linestyle=":", zorder=1)
    ax.axvline(1.0, color=".45", linewidth=0.8, linestyle="--", zorder=1)
    handles = []
    for index, legend_label in enumerate(("CNN ativo", "CNN misto", "RF espectral")):
        _, _, marker, color, _, filled = CALIBRATION_ARMS[index]
        drawn = ax.plot(
            [],
            [],
            marker=marker,
            color=color,
            markerfacecolor=color if filled else "white",
            markersize=5.5,
            linestyle="none",
            label=legend_label,
        )
        handles.extend(drawn)
    ax.set(
        title="(b) Calibração: inclinação\n(IC 95% por caso)",
        xlabel="Predito ~ referência\nx = 0: sem calibração\nx = 1: identidade\n"
        "rótulos: ICC e % |erro| ≤ 10",
        yticks=range(len(CALIBRATION_ARMS)),
        yticklabels=[label for _, label, *_ in CALIBRATION_ARMS],
        ylim=(len(CALIBRATION_ARMS) - 0.55, -0.6),
        xlim=(-0.25, 1.62),
        xticks=[0, 0.5, 1],
    )
    ax.grid(axis="x", color=".9", linewidth=0.5)
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        frameon=True,
        facecolor="white",
        edgecolor=".85",
        framealpha=1.0,
        borderpad=0.5,
        labelspacing=0.4,
        handletextpad=0.5,
    )

    for axis in grid:
        axis.grid(color=".9", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure(
        fig,
        "training_panels",
        [f"models/{MODEL_FILES[0]}", f"reports/{CALIBRATION_FILE}"],
    )


def main():



    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-trajectory", action="store_true",
                        help="CPU inference on full preselected case19; no training/download")
    parser.add_argument("--trajectory", action="store_true", help="plot existing trajectory audit")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = load_reports()
    audit_reports(reports)
    pk_reports = load_pk_reports()
    audit_pk_reports(reports, pk_reports)
    calibration = load_calibration(reports)
    figure_pipeline(reports)
    figure_comparison(reports)
    figure_offset(reports)
    figure_bootstrap(reports)
    figure_pk(pk_reports)
    figure_corpus_panels(reports)
    figure_training_panels(reports, calibration)
    if args.infer_trajectory:
        infer_trajectory()
    if args.infer_trajectory or args.trajectory:
        figure_trajectory()
    print(
        f"audited {len(reports)} report snapshots + {len(pk_reports)} Pk snapshots "
        f"+ {CALIBRATION_FILE}; generated historical figures"
    )


if __name__ == "__main__":
    main()

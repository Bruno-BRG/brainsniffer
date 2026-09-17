"""Render audited historical report snapshots as conventional Matplotlib article figures.

Historical metrics are not recomputed. Explicit --infer-trajectory runs bounded
CPU inference on the complete, preselected case19; default runs never infer,
download or train. --trajectory redraws its saved audit arrays without inference.
"""

from __future__ import annotations

import argparse
import json
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

ZONE_FILES = (
    "zone_figshare_active.json",
    "zone_figshare_mixed.json",
    "zone_vitaldb_active.json",
    "zone_vitaldb_mixed.json",
)
ZONE_ORDER = ("deep", "general", "light", "awake")
ZONE_PT = {
    "deep": "profunda\n0-40",
    "general": "geral\n40-60",
    "light": "leve\n60-80",
    "awake": "acordado\n80-100",
}

MODEL_FILES = ("brainsniffer_cnn.json", "brainsniffer_corpus_fixed.json")


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


def load_zone_reports() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in ZONE_FILES:
        path = REPORTS / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        ensure(isinstance(payload, dict), f"{name} must contain a JSON object")
        loaded[name] = payload
    return loaded


def audit_zone_reports(
    reports: dict[str, dict[str, Any]], zones: dict[str, dict[str, Any]]
) -> None:
    """Check zone snapshots against the audited historical aggregates."""
    pairs = (
        ("figshare_holdout_evaluation.json", "zone_figshare_active.json"),
        ("mixed_fixed_figshare_holdout.json", "zone_figshare_mixed.json"),
        ("vitaldb_external_validation.json", "zone_vitaldb_active.json"),
        ("mixed_vitaldb_external.json", "zone_vitaldb_mixed.json"),
    )
    for hist_name, zone_name in pairs:
        hist = reports[hist_name]
        zone = zones[zone_name]
        ensure(zone.get("scope") == "research_only", f"unsafe scope in {zone_name}")
        ensure(zone.get("retrained") is False, f"retrain flag in {zone_name}")
        hist_metrics = metrics(hist)
        overall = zone.get("overall", {})
        hist_n = float(hist_metrics.get("n", 0))
        ensure(float(zone.get("n_windows", 0)) == hist_n, f"n mismatch in {zone_name}")
        ensure(float(overall.get("n", 0)) == hist_n, f"overall n mismatch in {zone_name}")
        for key in ("mae", "rmse", "bias", "pearson_r"):
            a = float(hist_metrics[key])
            b = float(overall[key])
            ensure(abs(a - b) < 1e-4, f"{key} drift in {zone_name}: {a} vs {b}")
        by_zone = zone.get("by_zone", {})
        ensure(set(by_zone) == set(ZONE_ORDER), f"zone keys differ in {zone_name}")
        total = sum(int(by_zone[z]["n"]) for z in ZONE_ORDER)
        ensure(total == int(hist_n), f"zone n sum differs in {zone_name}")
        conf = zone.get("confusion", {})
        ensure(conf.get("order") == list(ZONE_ORDER), f"confusion order in {zone_name}")
        counts = conf.get("counts", [])
        ensure(len(counts) == 4 and all(len(r) == 4 for r in counts), f"confusion shape {zone_name}")
        flat = sum(sum(r) for r in counts)
        ensure(flat == int(hist_n), f"confusion sum differs in {zone_name}")


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


# Physical width matches the SBC text block (16 cm); no downscaling in TeX.
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
WIDTH = 16 / 2.54
COLORS = ("#0072B2", "#D55E00")
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
    for y, key in enumerate(ids):
        vals = [active[key]["mae"], mixed[key]["mae"]]
        ax.plot(vals, [y, y], color=".55", linewidth=0.8)
        for model, val in enumerate(vals):
            point(ax, val, y, model, label=LABELS[model] if y == 0 else None)
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
    ax.legend(loc="lower right", frameon=False)
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
    axes[0].plot(time, reference, color=COLORS[0], label="BIS referência", linewidth=0.9)
    axes[0].plot(time, raw, color=COLORS[1], label="CNN ativa bruta", linewidth=0.8)
    axes[0].set(ylabel="Índice (pontos BIS)", ylim=(0, 100),
                title="Figshare case19 · gravação completa · comparação offline")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(time, raw - reference, color=COLORS[1], linewidth=0.7)
    axes[1].axhline(0, color=".4", linewidth=0.7)
    axes[1].set(ylabel="Erro (pontos BIS)", xlabel="Tempo da referência desde o início (min)")
    for ax in axes:
        ax.set_xlim(0, max(audit["duration_seconds"] / 60, time[-1]))
        ax.grid(color=".9", linewidth=0.5)
    save_figure(fig, "bis_trajectory", ["data/raw/case19.mat", "models/brainsniffer_cnn.pt",
                                      "tmp/pdfs/trajectory-audit/case19.json"])


def figure_zones(zones: dict[str, dict[str, Any]]) -> None:
    """MAE por zona BIS verdadeira, ativo vs misto, nos dois benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4), layout="constrained", sharey=True)
    panels = (
        ("Figshare · 5 casos / 5.523 janelas", "zone_figshare_active.json", "zone_figshare_mixed.json", False),
        ("VitalDB · 15 casos / 38.730 janelas", "zone_vitaldb_active.json", "zone_vitaldb_mixed.json", True),
    )
    x = list(range(len(ZONE_ORDER)))
    width = 0.36
    for ax, (title, active_name, mixed_name, show_iso) in zip(axes, panels, strict=True):
        active = zones[active_name]["by_zone"]
        mixed = zones[mixed_name]["by_zone"]
        active_mae = [float(active[z]["mae"]) for z in ZONE_ORDER]
        mixed_mae = [float(mixed[z]["mae"]) for z in ZONE_ORDER]
        active_n = [int(active[z]["n"]) for z in ZONE_ORDER]
        mixed_n = [int(mixed[z]["n"]) for z in ZONE_ORDER]
        ensure(active_n == mixed_n, f"zone n differs: {active_name} vs {mixed_name}")
        ax.bar([i - width / 2 for i in x], active_mae, width, label=LABELS[0], color=COLORS[0])
        ax.bar([i + width / 2 for i in x], mixed_mae, width, label=LABELS[1], color="white",
               edgecolor=COLORS[1], linewidth=1.2, hatch="//")
        for i, n in enumerate(active_n):
            top = max(active_mae[i], mixed_mae[i])
            ax.text(i, top + 0.55, f"n={n}", ha="center", va="bottom", fontsize=7, color=".3")
        if show_iso:
            iso_active = zones[active_name]["isoelectric_subset_lt20"]
            iso_mixed = zones[mixed_name]["isoelectric_subset_lt20"]
            ensure(int(iso_active["n"]) == int(iso_mixed["n"]), "iso n differs")
            ax.text(
                0.98, 0.96,
                f"Subset isoelétrico BIS<20 (n={int(iso_active['n'])}): "
                f"MAE ativo {float(iso_active['mae']):.1f} vs misto {float(iso_mixed['mae']):.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": ".7", "boxstyle": "round,pad=0.3"},
            )
        else:
            ax.text(
                0.98, 0.96, "BIS<20 ausente no holdout Figshare (n=0)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": ".7", "boxstyle": "round,pad=0.3"},
            )
        ax.set(
            xticks=x,
            xticklabels=[ZONE_PT[z] for z in ZONE_ORDER],
            title=title,
            ylim=(0, 22),
        )
        ax.grid(axis="y", color=".9", linewidth=0.5)
    axes[0].set_ylabel("MAE (pontos BIS · menor é melhor)")
    axes[1].legend(frameon=False, loc="lower right")
    save_figure(fig, "zone_accuracy", [f"reports/{n}" for n in ZONE_FILES])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-trajectory", action="store_true",
                        help="CPU inference on full preselected case19; no training/download")
    parser.add_argument("--trajectory", action="store_true", help="plot existing trajectory audit")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = load_reports()
    audit_reports(reports)
    zones = load_zone_reports()
    audit_zone_reports(reports, zones)
    figure_pipeline(reports)
    figure_comparison(reports)
    figure_offset(reports)
    figure_bootstrap(reports)
    figure_zones(zones)
    if args.infer_trajectory:
        infer_trajectory()
    if args.infer_trajectory or args.trajectory:
        figure_trajectory()
    print(f"audited {len(reports)} report snapshots + {len(zones)} zone snapshots; generated historical figures")


if __name__ == "__main__":
    main()

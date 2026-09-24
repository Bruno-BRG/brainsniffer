"""Pós-processamento EWMA exploratório sobre predições congeladas (branch de experimento).

Este script NÃO retreina nada e NÃO altera os relatórios históricos. Ele lê as
predições por janela gravadas em ``tmp/reanalysis/*.npz`` (não versionadas;
gere-as com ``scripts/reviewer_reanalysis.py``) e aplica uma média móvel
exponencial (EWMA) causal, reiniciada por caso, para medir o efeito de
suavização temporal sobre MAE, RMSE, viés, Pearson e P_K, incluindo intervalos
bootstrap agrupados por caso e estatísticas de calibração para spans-chave.

A suavização é causal e por caso: no instante ``t`` usa apenas ``y_t`` e o
estado anterior, como no fluxo de replay. Sem ``span`` (1) o resultado é a
predição bruta. Como a varredura é pós-hoc e feita sobre os mesmos benchmarks
já inspecionados, os números são exploratórios e não substituem os resultados
históricos do artigo.

Saída: ``reports/ewma_postprocess.json`` (determinístico, sem timestamps).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from brainsniffer.pipeline.metrics import (
    bootstrap_case_metrics,
    compute_metrics,
    prediction_probability,
)

PREDICTION_DIR = Path("tmp/reanalysis")
REPORT_PATH = Path("reports/ewma_postprocess.json")

# (nome do braço, benchmark, arquivo de predições)
ARMS: tuple[tuple[str, str, str], ...] = (
    ("ativo", "figshare_holdout", "cnn_active_figshare"),
    ("misto", "figshare_holdout", "cnn_mixed_figshare"),
    ("rf_spectral", "figshare_holdout", "rf_spectral_figshare"),
    ("ativo", "vitaldb_external", "cnn_active_vitaldb"),
    ("misto", "vitaldb_external", "cnn_mixed_vitaldb"),
    ("rf_spectral", "vitaldb_external", "rf_spectral_vitaldb"),
)

# span em janelas de 5 s; span=1 corresponde à predição bruta
SPANS: tuple[int, ...] = (1, 2, 3, 5, 10, 15)
# spans-chave que recebem bootstrap agrupado por caso (métricas; sem bootstrap de Pk,
# cujo custo O(n log n) por réplica é proibitivo nos 38.730 pontos do VitalDB)
BOOTSTRAP_SPANS: tuple[int, ...] = (1, 10)
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
# bootstrap apenas nos braços CNN (o RF é referência e entra só com estimativas pontuais)
BOOTSTRAP_ARMS: frozenset[str] = frozenset({"ativo", "misto"})


def ewma_causal(prediction: np.ndarray, case_ids: np.ndarray, span: int) -> np.ndarray:
    """EWMA causal reiniciada por caso (alpha = 2 / (span + 1))."""

    if span <= 1:
        return prediction.astype(np.float64, copy=True)
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(prediction, dtype=np.float64)
    state: float | None = None
    previous_case: str | None = None
    for index, (value, case) in enumerate(zip(prediction, case_ids, strict=True)):
        case_key = str(case)
        if state is None or case_key != previous_case:
            state = float(value)
        else:
            state = alpha * float(value) + (1.0 - alpha) * state
        smoothed[index] = state
        previous_case = case_key
    return smoothed


def _icc_2_1_absolute(reference: np.ndarray, prediction: np.ndarray) -> float:
    """ICC(2,1) de concordância absoluta (duas vias, efeitos aleatórios, medida única)."""

    values = np.column_stack([reference, prediction]).astype(np.float64)
    n, k = values.shape
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_total = float(((values - grand) ** 2).sum())
    ss_rows = float(k * ((row_means - grand) ** 2).sum())
    ss_columns = float(n * ((column_means - grand) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1) if n > 1 else float("nan")
    ms_columns = ss_columns / (k - 1) if k > 1 else float("nan")
    ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else float("nan")
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def _calibration(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    differences = prediction - target
    slope, intercept = np.polyfit(target, prediction, 1)
    abs_error = np.abs(differences)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "bland_altman_bias": float(differences.mean()),
        "bland_altman_lower_loa": float(differences.mean() - 1.96 * differences.std(ddof=1)),
        "bland_altman_upper_loa": float(differences.mean() + 1.96 * differences.std(ddof=1)),
        "icc_2_1_absolute_agreement": _icc_2_1_absolute(target, prediction),
        "fraction_abs_error_le_10": float(np.mean(abs_error <= 10.0)),
        "abs_error_p95": float(np.percentile(abs_error, 95)),
    }


def _round(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = round(value, 12)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {str(key): _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value


def main() -> None:
    report: dict[str, object] = {
        "scope": "research_only",
        "analysis": "Pos-processamento EWMA causal exploratorio (pos-hoc; sem retreino).",
        "ewma_definition": "s_t = alpha*y_t + (1-alpha)*s_{t-1}; alpha = 2/(span+1); reinicio por caso; span em janelas de 5 s.",
        "raw_eeg_in_report": False,
        "retrained": False,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_spans": list(BOOTSTRAP_SPANS),
        "prediction_source": "tmp/reanalysis/*.npz (predicoes por janela dos mesmos benchmarks historicos)",
        "spans": list(SPANS),
        "arms": {},
    }
    header = (
        f"{'braço':<12} {'bench':<17} {'span':>4} {'MAE':>7} {'RMSE':>7} {'bias':>7} "
        f"{'r':>7} {'PK':>7} {'<=10':>6} {'P95':>7} {'ICC':>7}"
    )
    print(header)
    for arm, benchmark, filename in ARMS:
        path = PREDICTION_DIR / f"{filename}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"predicoes ausentes: {path}; rode scripts/reviewer_reanalysis.py primeiro"
            )
        data = np.load(path, allow_pickle=False)
        target = data["target"].astype(np.float64)
        prediction = data["prediction"].astype(np.float64)
        case_ids = data["case_ids"]
        by_span: dict[str, object] = {}
        for span in SPANS:
            smoothed = ewma_causal(prediction, case_ids, span)
            metrics = compute_metrics(target, smoothed)
            pk_value = prediction_probability(target, smoothed)
            calibration = _calibration(target, smoothed)
            block: dict[str, object] = {
                "span_windows": span,
                "span_seconds": 5 * span,
                "metrics": {key: _round(value) for key, value in metrics.items()},
                "pk": _round(float(pk_value)),
                "fraction_abs_error_le_10": _round(calibration["fraction_abs_error_le_10"]),
                "abs_error_p95": _round(calibration["abs_error_p95"]),
                "calibration": {key: _round(value) for key, value in calibration.items()},
            }
            if span in BOOTSTRAP_SPANS and arm in BOOTSTRAP_ARMS:
                case_bootstrap = bootstrap_case_metrics(
                    target,
                    smoothed,
                    case_ids,
                    n_bootstrap=BOOTSTRAP_SAMPLES,
                    seed=BOOTSTRAP_SEED,
                )
                block["case_bootstrap"] = _round(case_bootstrap)
            by_span[str(span)] = block
            print(
                f"{arm:<12} {benchmark:<17} {span:>4} {metrics['mae']:>7.3f} "
                f"{metrics['rmse']:>7.3f} {metrics['bias']:>7.3f} {metrics['pearson_r']:>7.3f} "
                f"{pk_value:>7.4f} {calibration['fraction_abs_error_le_10']:>6.3f} "
                f"{calibration['abs_error_p95']:>7.3f} "
                f"{calibration['icc_2_1_absolute_agreement']:>7.3f}"
            )
        report["arms"][f"{benchmark}__{arm}"] = {
            "benchmark": benchmark,
            "arm": arm,
            "prediction_file": str(path).replace("\\", "/"),
            "n_windows": int(target.size),
            "n_cases": int(np.unique(case_ids).size),
            "by_span": by_span,
        }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nrelatorio: {REPORT_PATH}")


if __name__ == "__main__":
    main()

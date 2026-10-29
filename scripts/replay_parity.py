"""Replay x offline parity on the Figshare holdout (active checkpoint, no training).

Each holdout case is pushed through ``replay_case`` in 5 s chunks, so the emission
windows coincide with the offline windows (stride = window = 5 s). Replay predictions
are aligned to the BIS label at window start + offset (the training target) and scored
with the same metrics as the offline report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from brainsniffer.config import DEFAULT_MIN_SIGNAL_QUALITY
from brainsniffer.data.mat_reader import load_case
from brainsniffer.data.preprocess import bis_stage, make_windows
from brainsniffer.pipeline.metrics import compute_metrics, prediction_probability
from brainsniffer.pipeline.realtime import replay_case
from brainsniffer.pipeline.training import load_checkpoint, predict_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_num_threads(2)
    model, config, payload = load_checkpoint(ROOT / "models/brainsniffer_cnn.pt", device="cpu")
    cases = sorted(payload["split"]["test_cases"])
    window = config.window_seconds
    ref, rep, off, per_case = [], [], [], {}
    for name in cases:
        case = load_case(ROOT / "data/raw" / f"{name}.mat")
        emissions = replay_case(
            model, case, config, stride_seconds=window, min_quality=DEFAULT_MIN_SIGNAL_QUALITY
        )
        windows = make_windows(case, config, min_quality=DEFAULT_MIN_SIGNAL_QUALITY)
        offline = dict(
            zip(
                np.rint(windows.start_seconds * case.sampling_rate).astype(int),
                predict_model(model, windows.signals, device="cpu", batch_size=64),
            )
        )
        r_ref, r_rep, r_off = [], [], []
        for p in emissions:
            if p.raw_bis is None:
                continue
            start = p.sample_index - config.window_samples
            index = int(round((start / case.sampling_rate + config.label_offset_seconds)
                              / case.label_interval_seconds))
            if not 0 <= index < case.bis.size or bis_stage(float(case.bis[index])) == "invalid":
                continue
            if start not in offline:
                continue
            r_ref.append(float(case.bis[index]))
            r_rep.append(p.raw_bis)
            r_off.append(float(offline[start]))
        ref += r_ref
        rep += r_rep
        off += r_off
        diff = np.abs(np.array(r_rep) - np.array(r_off))
        per_case[name] = {
            "replay_emissions": len(emissions),
            "offline_windows": len(windows.bis),
            "paired_windows": len(r_ref),
            "max_abs_diff": float(diff.max()) if diff.size else None,
            "mae_replay": float(np.mean(np.abs(np.array(r_rep) - np.array(r_ref)))),
            "mae_offline": float(np.mean(np.abs(np.array(r_off) - np.array(r_ref)))),
        }
    ref, rep, off = (np.asarray(a, dtype=float) for a in (ref, rep, off))
    report = {
        "scope": "research_only",
        "checkpoint": "models/brainsniffer_cnn.pt",
        "cases": cases,
        "min_quality": DEFAULT_MIN_SIGNAL_QUALITY,
        "n_paired_windows": int(ref.size),
        "replay": {**compute_metrics(ref, rep), "pk": prediction_probability(ref, rep)},
        "offline_same_windows": {**compute_metrics(ref, off), "pk": prediction_probability(ref, off)},
        "replay_vs_offline_max_abs_diff": float(np.abs(rep - off).max()),
        "per_case": per_case,
        "retrained": False,
    }
    out = ROOT / "reports/replay_parity_figshare_active.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

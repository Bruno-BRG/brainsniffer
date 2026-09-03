"""Publish a documented synthetic EEG stream for bench testing only.

Run this process first, then connect BrainSniffer with ``stream-lsl``. This is
not a patient simulator and must not be used as evidence of clinical safety.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pylsl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publisher LSL sintético para bancada")
    parser.add_argument("--name", default="BrainSnifferSyntheticEEG")
    parser.add_argument("--stream-type", default="EEG")
    parser.add_argument("--rate", type=float, default=128.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--amplitude-uv", type=float, default=20.0)
    parser.add_argument("--channel-name", default="Fpz")
    parser.add_argument("--unit", default="uV")
    parser.add_argument("--reference", default="linked ears")
    parser.add_argument("--montage", default="frontal referenced")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.rate) or args.rate <= 0:
        raise SystemExit("--rate deve ser finito e positivo")
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise SystemExit("--duration deve ser finito e positivo")
    if not np.isfinite(args.frequency) or args.frequency <= 0:
        raise SystemExit("--frequency deve ser finito e positivo")
    if not np.isfinite(args.amplitude_uv) or args.amplitude_uv < 0:
        raise SystemExit("--amplitude-uv deve ser finito e não negativo")

    info = pylsl.StreamInfo(
        args.name,
        args.stream_type,
        1,
        args.rate,
        pylsl.cf_float32,
        f"brainsniffer-synthetic-{args.name}",
    )
    channel = info.desc().append_child("channels").append_child("channel")
    channel.append_child_value("label", args.channel_name)
    channel.append_child_value("unit", args.unit)
    channel.append_child_value("reference", args.reference)
    channel.append_child_value("montage", args.montage)
    outlet = pylsl.StreamOutlet(info)
    print(
        f"Publicando {args.name!r}: {args.rate:g} Hz, canal {args.channel_name!r}; "
        f"duração {args.duration:g} s",
        file=sys.stderr,
    )

    sample_index = 0
    deadline = time.monotonic() + args.duration
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        value = args.amplitude_uv * np.sin(
            2.0 * np.pi * args.frequency * sample_index / args.rate
        )
        outlet.push_sample([float(value)], pylsl.local_clock())
        sample_index += 1
        next_tick += 1.0 / args.rate
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

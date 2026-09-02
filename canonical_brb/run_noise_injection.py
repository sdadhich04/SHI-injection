#!/usr/bin/env python3
"""Delegate reproducible software injection to the maintained noise pipeline.

Keeping injection materialized and separate avoids repeating it for every SHI
configuration.  This wrapper selects only top-level software ZIPs, excluding
the hardware_injection directory, and gives the SHI repository one documented
entry point for dataset preparation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve()
WORKSPACE_ROOT = HERE.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE_ROOT / "Data")
    parser.add_argument("--dataset-dir", type=Path, default=WORKSPACE_ROOT / "noise_injection/outputs/post_noise_injection_1")
    parser.add_argument("--noise-code", type=Path, default=WORKSPACE_ROOT / "noise_injection")
    parser.add_argument("--python", type=Path, default=WORKSPACE_ROOT / "noise_injection/venv/bin/python")
    parser.add_argument("--stage", choices=("ground-truth", "noise", "all"), default="all")
    parser.add_argument("--batch-hours", type=float)
    parser.add_argument("--batch", type=int, help="one-based noise-injection batch")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    archives = sorted(args.input_dir.resolve().glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no top-level software ZIPs in {args.input_dir}")
    python = str(args.python.resolve())
    dataset = args.dataset_dir.resolve()
    noise_code = args.noise_code.resolve()
    if args.plan_only:
        if args.batch_hours is None:
            raise ValueError("--plan-only requires --batch-hours")
        run([python, str(noise_code / "generate_noise_samples.py"), *map(str, archives),
             "--output-dir", str(dataset / "noise_injection"), "--scales", "0.25", "0.5",
             "--seed", str(args.seed), "--batch-hours", str(args.batch_hours)])
        return 0
    if args.stage in {"ground-truth", "all"}:
        run([python, str(noise_code / "extract_ground_truth.py"), *map(str, archives),
             "--output-dir", str(dataset / "ground_truth")])
    if args.stage in {"noise", "all"}:
        command = [python, str(noise_code / "generate_noise_samples.py"), *map(str, archives),
                   "--output-dir", str(dataset / "noise_injection"), "--scales", "0.25", "0.5",
                   "--seed", str(args.seed)]
        if args.batch_hours is not None:
            command.extend(["--batch-hours", str(args.batch_hours)])
        if args.batch is not None:
            if args.batch_hours is None:
                raise ValueError("--batch requires --batch-hours")
            command.extend(["--batch-index", str(args.batch)])
        run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

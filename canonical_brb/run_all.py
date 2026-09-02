#!/usr/bin/env python3
"""Run software and/or hardware canonical BRB-r batches with one command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("software", "hardware", "both"), default="both")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=int)
    group.add_argument("--all-batches", action="store_true")
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-adev", action="store_true")
    args = parser.parse_args()
    scripts = []
    if args.stage in {"software", "both"}:
        scripts.append(HERE / "run_software.py")
    if args.stage in {"hardware", "both"}:
        scripts.append(HERE / "run_hardware.py")
    okay = True
    for script in scripts:
        # abspath removes lexical ".." components without following the venv's
        # python symlink to the system interpreter (Path.resolve would do so).
        command = [os.path.abspath(sys.executable), str(script), "--total-batches", str(args.total_batches),
                   "--workers", str(args.workers)]
        command.extend(["--all-batches"] if args.all_batches else ["--batch", str(args.batch)])
        if args.no_adev:
            command.append("--no-adev")
        print("+", " ".join(command), flush=True)
        okay = subprocess.run(command).returncode == 0 and okay
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Capture or compare Go2-X5 low/high runtime parity snapshots.

Capture accepts a zero-argument factory so the diagnostic stays independent of
training entrypoints. The factory must return an initialized low/high env.

Examples:
  python3 scripts/check_go2x5_runtime_parity.py compare \
      --low /tmp/go2x5-low-runtime.json \
      --high /tmp/go2x5-high-runtime.json

  python3 scripts/check_go2x5_runtime_parity.py capture \
      --side low --kind runtime --factory my_debug_env:make_low_env \
      --output /tmp/go2x5-low-runtime.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.go2x5_runtime_parity import (  # noqa: E402
    collect_controller_snapshot,
    collect_runtime_snapshot,
    compare_snapshots,
    read_snapshot,
    write_snapshot,
)


def _factory(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    env = factory()
    return getattr(env, "_env", env)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--side", choices=("low", "high"), required=True)
    capture.add_argument("--kind", choices=("runtime", "controller"), required=True)
    capture.add_argument("--factory", required=True, help="Zero-argument module:function environment factory")
    capture.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--low", required=True)
    compare.add_argument("--high", required=True)
    compare.add_argument("--atol", type=float, default=1.0e-6)
    compare.add_argument("--report", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "capture":
        env = _factory(args.factory)
        collector = collect_runtime_snapshot if args.kind == "runtime" else collect_controller_snapshot
        snapshot = collector(env, args.side)
        write_snapshot(snapshot, args.output)
        print(f"wrote {args.kind} {args.side} snapshot: {args.output}")
        return 0

    low = read_snapshot(args.low)
    high = read_snapshot(args.high)
    mismatches = compare_snapshots(low, high, atol=args.atol)
    report = {
        "low": args.low,
        "high": args.high,
        "atol": args.atol,
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point.

    python -m osrepair.cli run --config configs/p4_smoke.json --out runs/
    python -m osrepair.cli env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from .runner import env_info, run_experiment


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="osrepair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="tune on dev seeds, then run paired branches on test seeds")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default="runs")
    r.add_argument("--run-id", default=None)
    sub.add_parser("env", help="print environment information")
    args = ap.parse_args(argv)

    if args.cmd == "env":
        print(json.dumps(env_info(), indent=2))
        return 0

    with open(args.config) as f:
        cfg = json.load(f)
    run_id = args.run_id or f"{cfg['name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = os.path.join(args.out, run_id)
    manifest = run_experiment(cfg, out_dir, argv=["python", "-m", "osrepair.cli", *argv])
    print(open(os.path.join(out_dir, "summary.md")).read())
    print(f"manifest: {os.path.join(out_dir, 'manifest.json')}")
    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

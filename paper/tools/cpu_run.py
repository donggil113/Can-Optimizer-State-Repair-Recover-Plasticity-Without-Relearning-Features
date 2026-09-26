"""Run one command, charge its CPU time to a budget ledger, refuse when the budget is spent.

    python3 paper/tools/cpu_run.py <bucket> -- <command ...>

Buckets and caps: analysis 900 s, build 600 s (child user+sys CPU, including
retries). RLIMIT_CPU of the child is set to the remaining budget, RLIMIT_AS to
3 GiB, and thread env vars to 1. Every call (including refusals and failures)
is appended to paper/cpu_ledger.tsv.
"""

import csv
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

CAPS = {"analysis": 900.0, "build": 600.0}
LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpu_ledger.tsv")
FIELDS = ["utc", "bucket", "status", "exit", "cpu_s", "wall_s", "peak_rss_kb", "bucket_total_cpu_s", "cmd"]


def spent(bucket):
    if not os.path.exists(LEDGER):
        return 0.0
    with open(LEDGER) as f:
        return sum(float(r["cpu_s"]) for r in csv.DictReader(f, delimiter="\t") if r["bucket"] == bucket)


def record(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    bucket, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
    remaining = CAPS[bucket] - spent(bucket)
    base = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bucket": bucket, "cmd": " ".join(cmd)}
    if remaining <= 1:
        record({**base, "status": "NOT_RUN_CAP", "exit": "", "cpu_s": 0, "wall_s": 0, "peak_rss_kb": 0,
                "bucket_total_cpu_s": round(CAPS[bucket] - remaining, 2)})
        print(f"[cpu_run] {bucket} budget exhausted", file=sys.stderr)
        return 3
    lim = int(remaining) + 1

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024 ** 3, 3 * 1024 ** 3))

    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    t = time.perf_counter()
    p = subprocess.run(cmd, preexec_fn=limits, env=env)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ru.ru_utime + ru.ru_stime
    status = "OK" if p.returncode == 0 else ("CAP_EXCEEDED" if p.returncode in (-24, 152) else "FAIL")
    record({**base, "status": status, "exit": p.returncode, "cpu_s": round(cpu, 3),
            "wall_s": round(time.perf_counter() - t, 3), "peak_rss_kb": ru.ru_maxrss,
            "bucket_total_cpu_s": round(CAPS[bucket] - remaining + cpu, 2)})
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())

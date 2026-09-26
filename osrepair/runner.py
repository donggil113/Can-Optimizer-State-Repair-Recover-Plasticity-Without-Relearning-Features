"""Paired checkpoint-branching experiment runner.

Protocol for one config:

1. Tuning (dev seeds only): from-scratch runs over the declared grids. The
   selection metric is computed on the post-branch window, the same window
   the reported comparison uses, so tuned baselines get the most favourable
   setting.
2. For each test seed: train the base optimizer from scratch up to the branch
   step (the "trunk"), take a full checkpoint, then run every branch arm from
   that checkpoint on the identical future stream. Each branch is checked to
   differ from the unmodified restore only in optimizer state (or, for
   declared hyperparameter controls, only in the declared field).
3. From-scratch arms (tuned Adam/SGD, periodic non-oracle reset, base Adam
   replay) run on the same stream seed and data RNG, so they see the same
   batches.
4. Integrity checks: identical batch and dropout-mask hashes across arms,
   keep_all branch == base-Adam-from-scratch bitwise, global ``random``
   untouched, per-arm status (PASS / FAIL / NOT_RUN).
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

from .interventions import NotApplicable, apply_intervention
from .rng import bit_hash, bitwise_equal
from .stream import TaskStream, check_seeds
from .trainer import (adam_state_diagnostics, branch, evaluate, feature_diagnostics, hash_parts, init_state,
                      restore, snapshot, train_step, verify_state_only)

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)


# ----------------------------------------------------------------- helpers
def _finite(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


class JsonlLog:
    def __init__(self, path: str):
        self._f = gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w")

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(_finite(rec), separators=(",", ":"), allow_nan=False) + "\n")

    def close(self) -> None:
        self._f.close()


def config_hash(cfg: dict) -> str:
    return bit_hash(json.loads(json.dumps(cfg)))


def code_hash() -> str:
    files = sorted(f for f in os.listdir(PKG_DIR) if f.endswith(".py"))
    contents = {}
    for fn in files:
        with open(os.path.join(PKG_DIR, fn)) as f:
            contents[fn] = f.read()
    return bit_hash(contents)


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            return f"ERROR: {e}"
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "status_porcelain": status.splitlines()}


def env_info() -> dict:
    def has(mod):
        try:
            __import__(mod)
            return True
        except ImportError:
            return False
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "third_party_required": [],
        "available": {m: has(m) for m in ("numpy", "torch", "pytest")},
    }


def make_stream(cfg: dict, seed: int) -> TaskStream:
    return TaskStream(seed=seed, **cfg["stream"])


def branch_step(cfg: dict) -> int:
    return cfg["branch"]["task_index"] * cfg["stream"]["steps_per_task"]


def window(cfg: dict) -> tuple[int, int]:
    b = branch_step(cfg)
    return b, b + cfg["post_branch_steps"]


def validate_config(cfg: dict) -> None:
    s = cfg["stream"]
    if cfg["model"]["in_dim"] != s["in_dim"] or cfg["model"]["out_dim"] != s["n_classes"]:
        raise ValueError("model in/out dims must match the stream")
    k = cfg["branch"]["task_index"]
    if not 1 <= k < s["n_tasks"]:
        raise ValueError("branch task_index must be a later task (1..n_tasks-1)")
    b, e = window(cfg)
    if e > s["n_tasks"] * s["steps_per_task"]:
        raise ValueError("post-branch window exceeds the stream")
    if b % cfg["eval_every"] != 0 or cfg["post_branch_steps"] < cfg["eval_every"]:
        raise ValueError("branch step must be a multiple of eval_every and the window must contain an eval")
    check_seeds(cfg["tuning"]["seeds"], cfg["tuning"]["split"])
    if cfg["tuning"]["split"] == "test":
        raise ValueError("tuning on test seeds is not allowed")
    check_seeds(cfg["test_seeds"], "test")
    names = [a["name"] for a in cfg["arms"]]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    seen = set()
    for a in cfg["arms"]:
        ref = (a.get("norm_match") or {}).get("reference_arm")
        if ref is not None and ref not in seen:
            raise ValueError(f"arm {a['name']} references {ref!r}, which must be listed earlier")
        seen.add(a["name"])
    if "keep_all" not in names:
        raise ValueError("a keep_all branch arm is required as the paired reference")


# ------------------------------------------------------------- run pieces
def run_segment(state, stream, start: int, end: int, *, cfg: dict, log: JsonlLog | None, tags: dict,
                eval_tasks: list[int], probe_x, ref_feats=None, ref_params=None, schedule=None,
                keep_records: bool = False, eval_from: int = 0) -> dict:
    """Train from ``start`` to ``end`` (exclusive), evaluating every ``eval_every``."""
    assert state.step == start, (state.step, start)
    every = cfg["eval_every"]
    steps, evals = [], []

    def do_eval():
        if state.step < eval_from:
            return
        ev = evaluate(state.model, stream, eval_tasks)
        fd = feature_diagnostics(state.model, probe_x, ref_feats, ref_params)
        rec = {**tags, "kind": "eval", "step": state.step, "eval": ev, "features": fd}
        evals.append(rec)
        if log:
            log.write(rec)

    if start % every == 0:
        do_eval()
    n_sched = 0
    while state.step < end:
        if schedule is not None and schedule["fires"](state.step):
            sd, _ = apply_intervention(state.optimizer.state_dict(), schedule["intervention"])
            state.optimizer.load_state_dict(sd)
            n_sched += 1
        rec = train_step(state, stream)
        rec = {**tags, "kind": "step", **rec}
        if log:
            log.write(rec)
        if keep_records:
            steps.append(rec)
        else:
            steps.append({k: rec[k] for k in ("step", "train_loss", "train_acc", "update_norm",
                                              "batch_hash", "dropout_hash", "update_norms")})
        if state.step % every == 0 or state.step == end:
            do_eval()
    return {"steps": steps, "evals": evals, "scheduled_interventions": n_sched}


def window_metrics(cfg: dict, seg: dict) -> dict:
    """Metrics on the post-branch window. See docs/PREREGISTRATION.md."""
    b, e = window(cfg)
    new_task = cfg["branch"]["task_index"]
    prev = list(range(new_task))
    ev = [r for r in seg["evals"] if b <= r["step"] <= e]
    at_b = [r for r in ev if r["step"] == b]
    after = [r for r in ev if r["step"] > b]
    if not at_b or not after:
        raise RuntimeError("window evaluation points missing")
    first, last = at_b[0], after[-1]
    st = [s for s in seg["steps"] if b <= s["step"] < e]
    prev_at_b = sum(first["eval"][k]["acc"] for k in prev) / len(prev)
    prev_end = sum(last["eval"][k]["acc"] for k in prev) / len(prev)
    early = st[:10]
    return {
        "new_task_auc_acc": sum(r["eval"][new_task]["acc"] for r in after) / len(after),
        "new_task_final_acc": last["eval"][new_task]["acc"],
        "new_task_final_loss": last["eval"][new_task]["loss"],
        "new_task_acc_at_branch": first["eval"][new_task]["acc"],
        "prev_task_acc_at_branch": first["eval"][new_task - 1]["acc"],
        "prev_task_acc_end": last["eval"][new_task - 1]["acc"],
        "forgetting_prev_task": first["eval"][new_task - 1]["acc"] - last["eval"][new_task - 1]["acc"],
        "forgetting_all_prev_mean": prev_at_b - prev_end,
        "online_train_acc_window": sum(s["train_acc"] for s in st) / len(st),
        "mean_update_norm_window": sum(s["update_norm"] for s in st) / len(st),
        "mean_update_norm_first10": sum(s["update_norm"] for s in early) / len(early),
        "cka_to_branch_end": last["features"].get("cka_to_branch"),
        "w1_rel_change_end": last["features"].get("w1_rel_change"),
        "dead_frac_end": last["features"]["dead_frac"],
        "diverged": any(not math.isfinite(s["train_loss"]) for s in st),
    }


def make_lr_schedule(spec: dict | None, start: int):
    if spec is None:
        return None
    if spec["kind"] == "bump":
        f, n = float(spec["factor"]), int(spec["steps"])
        return lambda step: f if start <= step < start + n else 1.0
    if spec["kind"] == "constant":
        f = float(spec["factor"])
        return lambda step: f if step >= start else 1.0
    raise ValueError(f"unknown lr schedule {spec['kind']!r}")


def make_norm_matcher(ref_steps: list[dict], counters: dict):
    target = {s["step"]: s["update_norms"] for s in ref_steps}

    def transform(step, deltas):
        tgt = target[step]
        out = {}
        for k, d in deltas.items():
            n = math.sqrt(sum(x * x for x in d))
            if n == 0.0:
                counters["zero_norm"] = counters.get("zero_norm", 0) + 1
                out[k] = d
            else:
                s = tgt[k] / n
                out[k] = [x * s for x in d]
        return out
    return transform


def make_periodic(spec: dict):
    period = int(spec["period"])
    return {"fires": lambda step: step > 0 and step % period == 0, "intervention": spec["intervention"]}


def resolve_optimizer(spec, tuned: dict) -> dict:
    if isinstance(spec, str):
        if not spec.startswith("tuned:"):
            raise ValueError(f"bad optimizer reference {spec!r}")
        return copy.deepcopy(tuned[spec.split(":", 1)[1]]["best"])
    return copy.deepcopy(spec)


# ------------------------------------------------------------------ tuning
def run_tuning(cfg: dict, tuning_log: JsonlLog, deadline: float) -> dict:
    tcfg = cfg["tuning"]
    b, e = window(cfg)
    results = {}
    for fam_name, fam in tcfg["families"].items():
        rows = []
        for hp in fam["grid"]:
            spec = {**fam["base"], **hp}
            per_seed = []
            for seed in tcfg["seeds"]:
                if time.time() > deadline:
                    raise TimeoutError("resource cap reached during tuning")
                stream = make_stream(cfg, seed)
                st = init_state(cfg["model"], spec, seed)
                probe = stream.probe_set(cfg["probe_size"])
                tasks = list(range(cfg["branch"]["task_index"] + 1))
                t0 = time.perf_counter()
                seg = run_segment(st, stream, 0, e, cfg=cfg, log=None, tags={}, eval_tasks=tasks, probe_x=probe,
                                  eval_from=b)
                m = window_metrics(cfg, seg)
                per_seed.append(m[tcfg["selection_metric"]])
                tuning_log.write({"family": fam_name, "optimizer": spec, "seed": seed, "metrics": m,
                                  "wall_s": time.perf_counter() - t0, "steps": e})
            score = sum(per_seed) / len(per_seed)
            if any(not math.isfinite(x) for x in per_seed):
                score = -math.inf
            rows.append({"optimizer": spec, "score": score, "per_seed": per_seed})
        # Deterministic rule: highest mean dev score, ties -> earliest grid entry.
        best = max(range(len(rows)), key=lambda i: (rows[i]["score"], -i))
        results[fam_name] = {"best": rows[best]["optimizer"], "rows": rows,
                             "n_runs": len(rows) * len(tcfg["seeds"]), "steps_per_run": e,
                             "best_at_grid_edge": grid_edges(fam["grid"], fam["grid"][best])}
    return results


def grid_edges(grid: list[dict], chosen: dict) -> dict:
    """Numeric hyperparameters whose selected value is the min or max of the grid.

    A selection at the edge means the optimum may lie outside the searched
    range, so the baseline is not demonstrably tuned.
    """
    out = {}
    for k, v in chosen.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        vals = sorted({g[k] for g in grid if k in g})
        if len(vals) < 2:
            continue
        if v == vals[0]:
            out[k] = "min"
        elif v == vals[-1]:
            out[k] = "max"
    return out


# ------------------------------------------------------------------- main
def run_experiment(cfg: dict, out_dir: str, argv: list[str] | None = None) -> dict:
    validate_config(cfg)
    os.makedirs(out_dir, exist_ok=False)
    t_start, c_start = time.perf_counter(), time.process_time()
    deadline = time.time() + cfg["resource_caps"]["max_wall_seconds"]
    global_rng_before = random.getstate()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    manifest = {
        "run_id": os.path.basename(out_dir.rstrip("/")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": argv if argv is not None else sys.argv,
        "config_sha256": config_hash(cfg),
        "code_sha256": code_hash(),
        "git": git_info(),
        "env": env_info(),
        "purpose": cfg.get("purpose"),
        "status": "RUNNING",
    }
    tuning_log = JsonlLog(os.path.join(out_dir, "tuning.jsonl"))
    log = JsonlLog(os.path.join(out_dir, "log.jsonl.gz"))
    summary = {"seeds": {}, "arms": {a["name"]: {k: v for k, v in a.items()} for a in cfg["arms"]}}
    arm_status: dict[str, dict[int, dict]] = {a["name"]: {} for a in cfg["arms"]}
    try:
        t0 = time.perf_counter()
        tuned = run_tuning(cfg, tuning_log, deadline)
        manifest["tuning"] = {"wall_s": time.perf_counter() - t0, "results": tuned}
        base_spec = resolve_optimizer(cfg["base_optimizer"], tuned)
        manifest["base_optimizer_resolved"] = base_spec
        b, e = window(cfg)
        new_task = cfg["branch"]["task_index"]
        eval_tasks = list(range(new_task + 1))
        for seed in cfg["test_seeds"]:
            summary["seeds"][seed] = seed_res = {"arms": {}, "checks": {}}
            stream = make_stream(cfg, seed)
            probe = stream.probe_set(cfg["probe_size"])
            seed_res["stream_fingerprint"] = stream.fingerprint()
            seed_res["oracle_boundaries"] = stream.oracle_boundaries()
            # ---- trunk
            t0 = time.perf_counter()
            trunk = init_state(cfg["model"], base_spec, seed)
            seed_res["init_hash"] = hash_parts(snapshot(trunk))
            run_segment(trunk, stream, 0, b, cfg=cfg, log=log, tags={"seed": seed, "arm": "_trunk"},
                        eval_tasks=eval_tasks, probe_x=probe)
            ckpt = snapshot(trunk)
            seed_res["trunk_wall_s"] = time.perf_counter() - t0
            seed_res["branch_checkpoint_hash"] = hash_parts(ckpt)
            ref_feats = restore(ckpt).model.features(probe)
            ref_params = copy.deepcopy(ckpt["model"]["params"])
            seed_res["branch_diagnostics"] = {"keep_all": adam_state_diagnostics(restore(ckpt), stream)}
            segs = {}
            for arm in cfg["arms"]:
                name = arm["name"]
                if time.time() > deadline:
                    arm_status[name][seed] = {"status": "NOT_RUN", "reason": "max_wall_seconds reached"}
                    continue
                t0 = time.perf_counter()
                try:
                    res = {"type": arm["type"]}
                    tags = {"seed": seed, "arm": name}
                    if arm["type"] == "branch":
                        st, record = branch(ckpt, arm["intervention"])
                        res["intervention_record"] = record
                        check = verify_state_only(restore(ckpt), st, probe)
                        res["state_only_check"] = check
                        expect_state_only = record["state_only"]
                        if expect_state_only and not check["state_only"]:
                            raise AssertionError(f"state-only check failed: {check}")
                        if expect_state_only != check["state_only"]:
                            raise AssertionError(f"declared state_only={expect_state_only} but check={check}")
                        if not check["outside_optimizer_unchanged"]:
                            raise AssertionError(f"intervention changed state outside the optimizer: {check}")
                        if st.optimizer.kind == "adam":
                            seed_res["branch_diagnostics"][name] = adam_state_diagnostics(st, stream)
                        st.lr_schedule = make_lr_schedule(arm.get("lr_schedule"), b)
                        nm = arm.get("norm_match")
                        counters = {}
                        if nm:
                            st.update_transform = make_norm_matcher(segs[nm["reference_arm"]]["steps"], counters)
                        seg = run_segment(st, stream, b, e, cfg=cfg, log=log, tags=tags, eval_tasks=eval_tasks,
                                          probe_x=probe, ref_feats=ref_feats, ref_params=ref_params)
                        res["norm_match_counters"] = counters
                    elif arm["type"] == "scratch":
                        spec = resolve_optimizer(arm["optimizer"], tuned)
                        res["optimizer_resolved"] = spec
                        st = init_state(cfg["model"], spec, seed)
                        sched = make_periodic(arm["schedule"]) if arm.get("schedule") else None
                        pre = run_segment(st, stream, 0, b, cfg=cfg, log=log, tags=tags, eval_tasks=eval_tasks,
                                          probe_x=probe, schedule=sched)
                        own_ck = snapshot(st)
                        seg = run_segment(st, stream, b, e, cfg=cfg, log=log, tags=tags, eval_tasks=eval_tasks,
                                          probe_x=probe, ref_feats=st.model.features(probe),
                                          ref_params=copy.deepcopy(own_ck["model"]["params"]), schedule=sched)
                        res["scheduled_interventions"] = pre["scheduled_interventions"] + seg["scheduled_interventions"]
                        res["own_branch_point_hash"] = hash_parts(own_ck)
                    else:
                        raise ValueError(f"unknown arm type {arm['type']!r}")
                    segs[name] = seg
                    res["metrics"] = window_metrics(cfg, seg)
                    res["final_hash"] = hash_parts(snapshot(st))
                    res["wall_s"] = time.perf_counter() - t0
                    res["train_steps"] = len(seg["steps"]) if arm["type"] == "branch" else e
                    res["status"] = "FAIL" if res["metrics"]["diverged"] else "PASS"
                    seed_res["arms"][name] = res
                    arm_status[name][seed] = {"status": res["status"]}
                except NotApplicable as ex:
                    arm_status[name][seed] = {"status": "NOT_APPLICABLE", "reason": str(ex)}
                except Exception as ex:  # noqa: BLE001 - failures are data, preserved in the manifest
                    arm_status[name][seed] = {"status": "FAIL", "error": repr(ex),
                                              "traceback": traceback.format_exc()}
            seed_res["checks"] = integrity_checks(cfg, segs, seed_res)
        manifest["status"] = "COMPLETED"
    except Exception as ex:  # noqa: BLE001
        manifest["status"] = "FAIL"
        manifest["error"] = repr(ex)
        manifest["traceback"] = traceback.format_exc()
    finally:
        tuning_log.close()
        log.close()
        manifest["global_random_untouched"] = random.getstate() == global_rng_before
        manifest["arm_status"] = arm_status
        manifest["wall_seconds"] = time.perf_counter() - t_start
        manifest["cpu_seconds"] = time.process_time() - c_start
        manifest["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["paired_deltas"] = paired_deltas(cfg, summary)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(_finite(summary), f, indent=1, sort_keys=True, allow_nan=False)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(_finite(manifest), f, indent=1, sort_keys=True, allow_nan=False)
        with open(os.path.join(out_dir, "summary.md"), "w") as f:
            f.write(render_summary(cfg, manifest, summary))
    return manifest


def integrity_checks(cfg: dict, segs: dict, seed_res: dict) -> dict:
    """Pairing checks across the arms that ran for one seed."""
    b, e = window(cfg)
    ref = segs.get("keep_all")
    out = {}
    if ref is None:
        return {"status": "FAIL", "reason": "keep_all did not run"}
    ref_b = [s["batch_hash"] for s in ref["steps"] if b <= s["step"] < e]
    ref_d = [s["dropout_hash"] for s in ref["steps"] if b <= s["step"] < e]
    for name, seg in segs.items():
        bh = [s["batch_hash"] for s in seg["steps"] if b <= s["step"] < e]
        dh = [s["dropout_hash"] for s in seg["steps"] if b <= s["step"] < e]
        out[name] = {"same_batches": bh == ref_b, "same_dropout_masks": dh == ref_d}
    replay = [a["name"] for a in cfg["arms"] if a.get("replays_base")]
    for name in replay:
        if name in seed_res["arms"] and "keep_all" in seed_res["arms"]:
            out[name]["bitwise_equal_to_keep_all"] = bitwise_equal(
                seed_res["arms"][name]["final_hash"], seed_res["arms"]["keep_all"]["final_hash"])
    ok = all(v["same_batches"] and v["same_dropout_masks"] and v.get("bitwise_equal_to_keep_all", True)
             for v in out.values())
    out["status"] = "PASS" if ok else "FAIL"
    return out


def paired_deltas(cfg: dict, summary: dict) -> dict:
    """Per-seed (arm - keep_all) differences for the declared window metrics."""
    keys = ("new_task_auc_acc", "new_task_final_acc", "forgetting_prev_task", "forgetting_all_prev_mean",
            "cka_to_branch_end", "mean_update_norm_first10")
    out = {}
    for arm in cfg["arms"]:
        name = arm["name"]
        per_key = {}
        for k in keys:
            vals = []
            for seed, sr in summary["seeds"].items():
                a, r = sr["arms"].get(name), sr["arms"].get("keep_all")
                if a and r and a["metrics"].get(k) is not None and r["metrics"].get(k) is not None:
                    vals.append(a["metrics"][k] - r["metrics"][k])
            if vals:
                per_key[k] = {"per_seed": vals, "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
                              "n": len(vals)}
        out[name] = per_key
    return out


def render_summary(cfg: dict, manifest: dict, summary: dict) -> str:
    lines = [f"# Run {manifest['run_id']}", "",
             f"- purpose: {cfg.get('purpose')}",
             f"- status: {manifest['status']}",
             f"- git: {manifest['git']['commit']} (dirty={manifest['git']['dirty']})",
             f"- config_sha256: {manifest['config_sha256']}",
             f"- wall: {manifest['wall_seconds']:.1f}s, cpu: {manifest['cpu_seconds']:.1f}s, "
             f"peak RSS: {manifest['peak_rss_kb'] / 1024:.1f} MiB",
             f"- base optimizer (resolved): {manifest.get('base_optimizer_resolved')}",
             ""]
    for fam, r in (manifest.get("tuning") or {}).get("results", {}).items():
        if r.get("best_at_grid_edge"):
            lines.append(f"- WARNING: tuned family {fam!r} selected a grid-edge value {r['best_at_grid_edge']}; "
                         f"the optimum may lie outside the grid (baseline not demonstrably tuned).")
    lines += ["",
             "Paired differences vs keep_all over test seeds (mean [min, max]).",
             "Descriptive only: n is tiny and nothing here is a hypothesis test.", "",
             "| arm | timing | status | d new_task_auc_acc | d forgetting_prev | d cka_end | d upd_norm_first10 |",
             "|---|---|---|---|---|---|---|"]
    for arm in cfg["arms"]:
        name = arm["name"]
        st = sorted({v["status"] for v in manifest["arm_status"][name].values()}) or ["NOT_RUN"]
        d = manifest["paired_deltas"].get(name, {})

        def fmt(k):
            if k not in d:
                return "n/a"
            v = d[k]
            return f"{v['mean']:+.4f} [{v['min']:+.4f}, {v['max']:+.4f}]"
        lines.append(f"| {name} | {arm.get('timing')} | {'/'.join(st)} | {fmt('new_task_auc_acc')} | "
                     f"{fmt('forgetting_prev_task')} | {fmt('cka_to_branch_end')} | {fmt('mean_update_norm_first10')} |")
    lines += ["", "Integrity checks per seed:", ""]
    for seed, sr in summary["seeds"].items():
        lines.append(f"- seed {seed}: {sr['checks'].get('status')}")
    return "\n".join(lines) + "\n"

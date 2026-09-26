"""Stage 1: is there a loss of learning ability on a comparable new task, and if
so, is a correctly bias-corrected optimizer-state intervention distinguishable
from its update-magnitude effect?

Protocol (fixed in configs/p4_stage1.json before running):

* Stream: ``n_train_tasks`` tasks of ``steps_per_task`` steps, trained online.
  One extra task (index ``n_train_tasks``) is never trained in the stream and
  serves as the fixed *probe task*.
* Checkpoints: early (end of task 0) and late (end of the stream), each with
  its carried optimizer state. Reference: the same initial weights with a
  fresh optimizer (never trained).
* Probe: from each starting point, train ``probe_steps`` steps on the probe
  task with identical batches and dropout masks (probe RNG streams replace
  the checkpoint's RNG), same optimizer hyperparameters, and evaluate the
  probe task and the starting point's last-trained task every ``eval_every``.
  Same task, same batches, same update budget: task difficulty cancels in
  late-vs-early comparisons.
* Interventions (only for conditions where the phenomenon is established):
  from the same late checkpoint, state-only interventions and, for each, a
  keep_all branch whose per-step, per-tensor update norms are copied from that
  intervention's branch (update-norm-matched control).

Caps: CPU time of this process (tuning included; no child processes), address
space, one thread. Cells that would exceed the remaining CPU budget are
recorded as NOT_RUN; nothing is inferred from missing cells.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import random
import resource
import sys
import time
import traceback
from datetime import datetime, timezone

from .model import MLP
from .rng import RNG, bit_hash, bitwise_equal, derive_seed
from .runner import _finite, code_hash, config_hash, env_info, git_info, grid_edges, make_norm_matcher
from .stream import TaskStream, check_seeds
from .trainer import (branch, evaluate, feature_diagnostics, hash_parts, init_state, restore, snapshot,
                      train_step, verify_state_only)


# ------------------------------------------------------------------ budget
class Budget:
    """CPU-time budget for this process, including any child processes."""

    def __init__(self, cap_s: float):
        self.cap = float(cap_s)
        self._t0 = self._now()

    @staticmethod
    def _now() -> float:
        ch = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() + ch.ru_utime + ch.ru_stime

    def used(self) -> float:
        return self._now() - self._t0

    def allows(self, est_s: float) -> bool:
        return self.used() + est_s <= self.cap


def apply_os_limits(caps: dict) -> dict:
    """Hard backstops. The CPU limit sits slightly above the planned cap so the
    in-process budget check (which records NOT_RUN cells) normally fires first."""
    applied = {}
    as_bytes = int(caps["max_rss_gib"] * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
    applied["RLIMIT_AS_bytes"] = as_bytes
    cpu = int(caps["max_cpu_seconds"] + caps.get("os_cpu_grace_seconds", 60))
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    new_hard = hard if hard != resource.RLIM_INFINITY and hard < cpu else cpu
    resource.setrlimit(resource.RLIMIT_CPU, (min(cpu, new_hard), new_hard))
    applied["RLIMIT_CPU_seconds"] = resource.getrlimit(resource.RLIMIT_CPU)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    applied["threads"] = 1
    return applied


# --------------------------------------------------------------- helpers
class Log:
    def __init__(self, path: str):
        self._f = gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "a")

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(_finite(rec), separators=(",", ":"), allow_nan=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def make_stream(cfg: dict, cond: dict, seed: int) -> TaskStream:
    s = cfg["stream_common"]
    stream = TaskStream(seed=seed, in_dim=s["in_dim"], n_classes=cond["n_classes"], n_tasks=s["n_train_tasks"] + 1,
                        steps_per_task=s["steps_per_task"], batch_size=s["batch_size"], family=cond["family"],
                        eval_size=s["eval_size"])
    if cond["family"] == "label_permutation":
        probe = stream._perms[s["n_train_tasks"]]
        if probe in stream._perms[:s["n_train_tasks"]]:
            raise RuntimeError(f"probe permutation was trained before (seed {seed}); probe task is not new")
    return stream


def model_cfg(cfg: dict, cond: dict) -> dict:
    return {**cfg["model"], "in_dim": cfg["stream_common"]["in_dim"], "out_dim": cond["n_classes"]}


def data_hash(stream: TaskStream, tasks: list[int]) -> str:
    return bit_hash({"teachers": stream._teachers, "perms": stream._perms,
                     "eval": [stream.eval_set(k) for k in tasks]})


def all_params(model: MLP) -> list[float]:
    return [v for p in model.params.values() for v in p.data]


def _norm(xs):
    return math.sqrt(sum(v * v for v in xs))


# ----------------------------------------------------------------- trunk
def run_trunk(cfg, cond, opt_spec, seed, snap_steps=(), log=None, tags=None):
    """Online training over the train tasks. Logs per-task aggregates."""
    s = cfg["stream_common"]
    stream = make_stream(cfg, cond, seed)
    st = init_state(model_cfg(cfg, cond), opt_spec, seed)
    init_ck = snapshot(st)
    total = s["n_train_tasks"] * s["steps_per_task"]
    snaps, acc_sum, per_task, diverged = {}, 0.0, [], False
    agg = {"loss": 0.0, "acc": 0.0, "upd": 0.0}
    for _ in range(total):
        if st.step in snap_steps:
            snaps[st.step] = snapshot(st)
        rec = train_step(st, stream)
        if not math.isfinite(rec["train_loss"]):
            diverged = True
            break
        acc_sum += rec["train_acc"]
        agg["loss"] += rec["train_loss"]
        agg["acc"] += rec["train_acc"]
        agg["upd"] += rec["update_norm"]
        if st.step % s["steps_per_task"] == 0:
            n = s["steps_per_task"]
            row = {"task": st.step // n - 1, "online_loss": agg["loss"] / n, "online_acc": agg["acc"] / n,
                   "mean_update_norm": agg["upd"] / n}
            per_task.append(row)
            if log:
                log.write({**(tags or {}), "kind": "trunk_task", **row})
            agg = {"loss": 0.0, "acc": 0.0, "upd": 0.0}
    if st.step in snap_steps:
        snaps[st.step] = snapshot(st)
    return {"stream": stream, "init_ck": init_ck, "snaps": snaps, "state": st, "diverged": diverged,
            "online_acc": acc_sum / max(st.step, 1) if not diverged else -math.inf, "per_task": per_task}


# ----------------------------------------------------------------- probe
def with_probe_rng(ck: dict, seed: int) -> dict:
    ck = copy.deepcopy(ck)
    for name in ("data", "dropout"):
        ck["rng"][name] = RNG(derive_seed("probe", name, seed)).get_state()
    return ck


def run_probe(st, stream, cfg, probe_task, old_task, probe_x, log, tags, norm_ref=None):
    s = cfg["probe"]
    every = s["eval_every"]
    feats0 = st.model.features(probe_x)
    params0 = copy.deepcopy({k: list(p.data) for k, p in st.model.params.items()})
    theta0 = all_params(st.model)
    counters = {}
    if norm_ref is not None:
        st.update_transform = make_norm_matcher(norm_ref, counters)
    tasks = [probe_task] + ([old_task] if old_task is not None else [])
    evals, steps = [], []

    def do_eval(k):
        ev = evaluate(st.model, stream, tasks)
        fd = feature_diagnostics(st.model, probe_x, feats0, params0)
        rec = {"k": k, "probe_acc": ev[probe_task]["acc"], "probe_loss": ev[probe_task]["loss"],
               "old_acc": ev[old_task]["acc"] if old_task is not None else None, **fd,
               "theta_rel_change": _norm([a - b for a, b in zip(all_params(st.model), theta0)]) / _norm(theta0)}
        evals.append(rec)
        log.write({**tags, "kind": "probe_eval", **rec})

    do_eval(0)
    for k in range(1, s["steps"] + 1):
        rec = train_step(st, stream, task=probe_task)
        r = {"step": rec["step"], "train_loss": rec["train_loss"], "train_acc": rec["train_acc"],
             "update_norm": rec["update_norm"], "update_norms": rec["update_norms"],
             "batch_hash": rec["batch_hash"], "dropout_hash": rec["dropout_hash"]}
        steps.append(r)
        log.write({**tags, "kind": "probe_step", "k": k, **r})
        if not math.isfinite(rec["train_loss"]):
            break
        if k % every == 0:
            do_eval(k)
    after = [e for e in evals if e["k"] > 0]
    early = [e for e in after if e["k"] <= s["early_window"]]
    last = evals[-1]
    m = {
        "probe_auc_acc": sum(e["probe_acc"] for e in after) / len(after),
        "probe_early_acc": sum(e["probe_acc"] for e in early) / len(early),
        "probe_final_acc": last["probe_acc"],
        "probe_acc_start": evals[0]["probe_acc"],
        "old_task_acc_start": evals[0]["old_acc"],
        "old_task_forgetting": (evals[0]["old_acc"] - last["old_acc"]) if old_task is not None else None,
        "update_norm_first10": sum(x["update_norm"] for x in steps[:10]) / min(10, len(steps)),
        "update_norm_mean": sum(x["update_norm"] for x in steps) / len(steps),
        "theta_rel_change_end": last["theta_rel_change"],
        "w1_rel_change_end": last.get("w1_rel_change"),
        "cka_to_start_end": last.get("cka_to_branch"),
        "dead_frac_start": evals[0]["dead_frac"],
        "dead_frac_end": last["dead_frac"],
        "diverged": any(not math.isfinite(x["train_loss"]) for x in steps),
        "norm_match_zero_norm": counters.get("zero_norm", 0),
    }
    return m, steps


# ---------------------------------------------------------------- tuning
def tune_family(cfg, cond, fam_name, fam, seeds, budget, cells, log):
    est = cfg["cost_model"]["trunk_cpu_s"]
    rows = {}

    def run(lr):
        vals = []
        for seed in seeds:
            if not budget.allows(est):
                cells.append({"cell": "tune", "cond": cond["name"], "family": fam_name, "lr": lr, "seed": seed,
                              "status": "NOT_RUN", "reason": "cpu budget"})
                return None
            c0 = budget.used()
            spec = {**fam["base"], "lr": lr}
            tr = run_trunk(cfg, cond, spec, seed, log=None)
            status = "FAIL" if tr["diverged"] else "PASS"
            cells.append({"cell": "tune", "cond": cond["name"], "family": fam_name, "lr": lr, "seed": seed,
                          "status": status, "online_acc": tr["online_acc"], "cpu_s": budget.used() - c0})
            log.write({"kind": "tune", "cond": cond["name"], "family": fam_name, "optimizer": spec, "seed": seed,
                       "online_acc": tr["online_acc"], "diverged": tr["diverged"], "per_task": tr["per_task"]})
            vals.append(tr["online_acc"])
        return sum(vals) / len(vals)

    grid = list(fam["lr_grid"])
    for lr in grid:
        rows[lr] = run(lr)
    extended = None
    if all(v is not None for v in rows.values()):
        best = max(grid, key=lambda lr: (rows[lr], -grid.index(lr)))
        edge = grid_edges([{"lr": x} for x in grid], {"lr": best}).get("lr")
        if edge:
            f = cfg["tuning"]["edge_extension"]["factor"]
            new = min(grid) / f if edge == "min" else max(grid) * f
            extended = {"side": edge, "value": new}
            grid.append(new)
            rows[new] = run(new)
    complete = all(v is not None for v in rows.values())
    if not complete:
        return {"status": "NOT_RUN_PARTIAL", "rows": rows, "extended": extended}
    best = max(grid, key=lambda lr: (rows[lr], -grid.index(lr)))
    edge = grid_edges([{"lr": x} for x in grid], {"lr": best}).get("lr")
    return {"status": "TUNING_UNRESOLVED" if edge else "RESOLVED", "best": {**fam["base"], "lr": best},
            "best_at_edge": edge, "rows": {str(k): v for k, v in rows.items()}, "extended": extended}


# ------------------------------------------------------------------ main
def run_stage(cfg: dict, out_dir: str, argv: list[str]) -> dict:
    os.makedirs(out_dir, exist_ok=False)
    caps = cfg["resource_caps"]
    limits = apply_os_limits(caps)
    budget = Budget(caps["max_cpu_seconds"] - caps.get("reserved_cpu_seconds", 0))
    t_wall = time.perf_counter()
    rnd_before = random.getstate()
    for name in ("dev", "test"):
        check_seeds(cfg["seeds"][name], name)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    manifest = {"run_id": os.path.basename(out_dir.rstrip("/")), "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": argv, "config_sha256": config_hash(cfg), "code_sha256": code_hash(), "git": git_info(),
                "env": env_info(), "limits_applied": limits, "status": "RUNNING"}
    cells: list[dict] = []
    results = {"tuning": {}, "phenomenon": {}, "interventions": {}, "verdicts": {}}
    log = Log(os.path.join(out_dir, "log.jsonl.gz"))

    def save():
        manifest["cells"] = cells
        manifest["results"] = results
        manifest["cpu_seconds_used"] = budget.used()
        manifest["wall_seconds"] = time.perf_counter() - t_wall
        manifest["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(_finite(manifest), f, indent=1, sort_keys=True, allow_nan=False)

    try:
        s = cfg["stream_common"]
        n_train = s["n_train_tasks"]
        early_step, late_step = s["steps_per_task"], n_train * s["steps_per_task"]
        # -------------------------------------------------- 1. tuning (dev)
        for cond in cfg["conditions"]:
            results["tuning"][cond["name"]] = {}
            for fam_name, fam in cfg["tuning"]["families"].items():
                results["tuning"][cond["name"]][fam_name] = tune_family(
                    cfg, cond, fam_name, fam, cfg["seeds"]["dev"], budget, cells, log)
                save()
        # ------------------------------------------ 2. phenomenon (test seeds)
        late_cks: dict = {}
        est_cell = cfg["cost_model"]["trunk_cpu_s"] + 3 * cfg["cost_model"]["probe_cpu_s"]
        for cond in cfg["conditions"]:
            _phenomenon(cfg, cond, cfg["base_family"], results, cells, log, budget, est_cell, late_cks,
                        early_step, late_step)
            save()
        for cond in cfg["conditions"]:
            results["verdicts"][cond["name"]] = _phenomenon_verdict(cfg, cond, results)
        save()
        # ------------------------------------ 3. interventions (if established)
        for cond in cfg["conditions"]:
            v = results["verdicts"][cond["name"]]
            if v["phenomenon"] != "ESTABLISHED":
                cells.append({"cell": "interventions", "cond": cond["name"], "status": "NOT_RUN",
                              "reason": f"phenomenon {v['phenomenon']}"})
                continue
            _interventions(cfg, cond, results, cells, log, budget, late_cks)
            results["verdicts"][cond["name"]].update(_intervention_verdict(cfg, cond, results))
            save()
        # ---------------------------- 4. secondary families (lowest priority)
        for fam_name in cfg["secondary_families"]:
            for cond in cfg["conditions"]:
                _phenomenon(cfg, cond, fam_name, results, cells, log, budget, est_cell, None, early_step, late_step)
                save()
        results["overall"] = _overall(cfg, results)
        manifest["status"] = "COMPLETED"
    except Exception as ex:  # noqa: BLE001 - failures are data
        manifest["status"] = "FAIL"
        manifest["error"] = repr(ex)
        manifest["traceback"] = traceback.format_exc()
    finally:
        log.close()
        manifest["global_random_untouched"] = random.getstate() == rnd_before
        save()
        with open(os.path.join(out_dir, "summary.md"), "w") as f:
            f.write(render(cfg, manifest))
    return manifest


def _resolved_spec(results, cond_name, fam_name):
    t = results["tuning"][cond_name].get(fam_name, {})
    return t.get("best"), t.get("status")


def _phenomenon(cfg, cond, fam_name, results, cells, log, budget, est_cell, late_cks, early_step, late_step):
    n_train = cfg["stream_common"]["n_train_tasks"]
    spec, tstatus = _resolved_spec(results, cond["name"], fam_name)
    res = results["phenomenon"].setdefault(cond["name"], {}).setdefault(fam_name, {"tuning_status": tstatus,
                                                                                 "seeds": {}})
    for seed in cfg["seeds"]["test"]:
        cell = {"cell": "phenomenon", "cond": cond["name"], "family": fam_name, "seed": seed}
        if spec is None:
            cells.append({**cell, "status": "NOT_RUN", "reason": f"tuning {tstatus}"})
            continue
        if not budget.allows(est_cell):
            cells.append({**cell, "status": "NOT_RUN", "reason": "cpu budget"})
            continue
        c0 = budget.used()
        try:
            tags = {"cond": cond["name"], "family": fam_name, "seed": seed}
            tr = run_trunk(cfg, cond, spec, seed, snap_steps=(early_step, late_step), log=log,
                           tags={**tags, "phase": "trunk"})
            if tr["diverged"]:
                raise RuntimeError("trunk diverged")
            stream = tr["stream"]
            probe_x = stream.probe_set(cfg["probe"]["probe_size"])
            starts = {"fresh": (tr["init_ck"], None), "early": (tr["snaps"][early_step], 0),
                      "late": (tr["snaps"][late_step], n_train - 1)}
            out = {"data_hash": data_hash(stream, [0, n_train - 1, n_train]),
                   "checkpoint_hash": {k: hash_parts(v[0]) for k, v in starts.items()},
                   "trunk_online_acc_first5": [r["online_acc"] for r in tr["per_task"][:5]],
                   "trunk_online_acc_last5": [r["online_acc"] for r in tr["per_task"][-5:]],
                   "probes": {}}
            batches = {}
            for name, (ck, old) in starts.items():
                st = restore(with_probe_rng(ck, seed))
                m, steps = run_probe(st, stream, cfg, n_train, old, probe_x, log, {**tags, "phase": "probe",
                                                                                   "start": name})
                out["probes"][name] = m
                batches[name] = ([x["batch_hash"] for x in steps], [x["dropout_hash"] for x in steps])
                if name == "late" and late_cks is not None:
                    late_cks[(cond["name"], seed)] = {"ck": with_probe_rng(ck, seed), "stream": stream,
                                                      "probe_x": probe_x, "keep_all_final": hash_parts(snapshot(st))}
            out["paired_batches_and_masks"] = all(bitwise_equal(batches["fresh"], v) for v in batches.values())
            if not out["paired_batches_and_masks"]:
                raise AssertionError("probe batches/masks differ across starting points")
            out["cpu_s"] = budget.used() - c0
            res["seeds"][seed] = out
            cells.append({**cell, "status": "FAIL" if any(p["diverged"] for p in out["probes"].values()) else "PASS",
                          "cpu_s": out["cpu_s"]})
        except Exception as ex:  # noqa: BLE001
            cells.append({**cell, "status": "FAIL", "error": repr(ex), "traceback": traceback.format_exc()})


def _deltas(seeds: dict, a: str, b: str, key: str, getter=None):
    vals = []
    for sr in seeds.values():
        g = getter or (lambda s, name: s["probes"][name])
        x, y = g(sr, a).get(key), g(sr, b).get(key)
        if x is not None and y is not None:
            vals.append(x - y)
    return vals


def _phenomenon_verdict(cfg, cond, results):
    rule = cfg["decision"]
    ph = results["phenomenon"].get(cond["name"], {}).get(cfg["base_family"], {})
    tstatus = ph.get("tuning_status")
    seeds = ph.get("seeds", {})
    out = {"tuning_status": tstatus, "n_seeds": len(seeds)}
    for a, b in (("late", "early"), ("late", "fresh"), ("early", "fresh")):
        for key in ("probe_auc_acc", "probe_early_acc", "probe_final_acc"):
            out[f"d_{key}_{a}_minus_{b}"] = _deltas(seeds, a, b, key)
    d = out["d_probe_auc_acc_late_minus_early"]
    if len(seeds) < len(cfg["seeds"]["test"]):
        out["phenomenon"] = "INCOMPLETE"
        out["reason"] = "missing phenomenon cells (FAIL/NOT_RUN); no verdict from missing cells"
    elif tstatus != "RESOLVED":
        out["phenomenon"] = "INCOMPLETE"
        out["reason"] = f"base tuning {tstatus}"
    elif sum(d) / len(d) <= -rule["mie_acc"] and all(x < 0 for x in d):
        out["phenomenon"] = "ESTABLISHED"
    else:
        out["phenomenon"] = "NOT_ESTABLISHED"
    return out


def _interventions(cfg, cond, results, cells, log, budget, late_cks):
    arms = cfg["interventions"]["arms"]
    est = cfg["cost_model"]["probe_cpu_s"] * len(arms)
    res = results["interventions"].setdefault(cond["name"], {"seeds": {}})
    for seed in cfg["seeds"]["test"]:
        cell = {"cell": "interventions", "cond": cond["name"], "seed": seed}
        item = late_cks.get((cond["name"], seed))
        if item is None:
            cells.append({**cell, "status": "NOT_RUN", "reason": "late checkpoint missing"})
            continue
        if not budget.allows(est):
            cells.append({**cell, "status": "NOT_RUN", "reason": "cpu budget"})
            continue
        c0 = budget.used()
        try:
            ck, stream, probe_x = item["ck"], item["stream"], item["probe_x"]
            n_train = cfg["stream_common"]["n_train_tasks"]
            out, steps_by_arm, checks = {}, {}, {}
            for arm in arms:
                st, rec = branch(ck, arm["intervention"])
                chk = verify_state_only(restore(ck), st, probe_x)
                if not chk["state_only"]:
                    raise AssertionError(f"{arm['name']}: not state-only: {chk}")
                checks[arm["name"]] = {"state_only": chk["state_only"],
                                       "optimizer_state_changed": chk["optimizer_state_changed"]}
                ref = arm.get("norm_match_reference")
                m, steps = run_probe(st, stream, cfg, n_train, n_train - 1, probe_x, log,
                                     {"cond": cond["name"], "seed": seed, "phase": "intervention", "arm": arm["name"]},
                                     norm_ref=steps_by_arm[ref] if ref else None)
                out[arm["name"]] = m
                steps_by_arm[arm["name"]] = steps
                if arm["name"] == "keep_all":
                    checks["keep_all_equals_phenomenon_late_probe"] = hash_parts(snapshot(st)) == item["keep_all_final"]
            hashes = {a: ([x["batch_hash"] for x in s_], [x["dropout_hash"] for x in s_])
                      for a, s_ in steps_by_arm.items()}
            checks["paired_batches_and_masks"] = all(v == hashes["keep_all"] for v in hashes.values())
            if not (checks["paired_batches_and_masks"] and checks["keep_all_equals_phenomenon_late_probe"]):
                raise AssertionError(f"pairing checks failed: {checks}")
            res["seeds"][seed] = {"probes": out, "checks": checks, "cpu_s": budget.used() - c0}
            cells.append({**cell, "status": "FAIL" if any(p["diverged"] for p in out.values()) else "PASS",
                          "cpu_s": budget.used() - c0})
        except Exception as ex:  # noqa: BLE001
            cells.append({**cell, "status": "FAIL", "error": repr(ex), "traceback": traceback.format_exc()})


def _intervention_verdict(cfg, cond, results):
    rule = cfg["decision"]
    seeds = results["interventions"].get(cond["name"], {}).get("seeds", {})
    out = {"intervention_n_seeds": len(seeds), "per_intervention": {}}
    if len(seeds) < len(cfg["seeds"]["test"]):
        out["intervention_stage"] = "INCOMPLETE"
        return out
    any_remaining = False
    for x in cfg["interventions"]["primary"]:
        ctrl = f"keepdir_{x}mag"
        eff = _deltas(seeds, x, "keep_all", "probe_auc_acc")
        resid = _deltas(seeds, x, ctrl, "probe_auc_acc")
        effect_present = sum(eff) / len(eff) >= rule["mie_acc"] and all(v > 0 for v in eff)
        remains = abs(sum(resid) / len(resid)) >= rule["mie_acc"] and (all(v > 0 for v in resid)
                                                                      or all(v < 0 for v in resid))
        verdict = ("DIFFERENCE_REMAINS" if effect_present and remains
                   else "NO_POSITIVE_EFFECT_OVER_KEEP_ALL" if not effect_present else "EXPLAINED_BY_UPDATE_NORM")
        any_remaining |= verdict == "DIFFERENCE_REMAINS"
        extra = {k: _deltas(seeds, x, "keep_all", k) for k in
                 ("probe_early_acc", "probe_final_acc", "old_task_forgetting", "update_norm_first10",
                  "theta_rel_change_end", "cka_to_start_end")}
        out["per_intervention"][x] = {"d_auc_vs_keep_all": eff, "d_auc_vs_norm_matched": resid, "verdict": verdict,
                                      "d_vs_keep_all_other": extra,
                                      "control_d_auc_vs_keep_all": _deltas(seeds, ctrl, "keep_all", "probe_auc_acc")}
    for a in cfg["interventions"]["artifacts"]:
        out["per_intervention"][a] = {"d_auc_vs_keep_all": _deltas(seeds, a, "keep_all", "probe_auc_acc"),
                                      "d_update_norm_first10": _deltas(seeds, a, "keep_all", "update_norm_first10"),
                                      "role": "miscorrection artifact control"}
    out["intervention_stage"] = "MECHANISM_PILOT_CANDIDATE" if any_remaining else "STATE_REPAIR_BRANCH_ON_HOLD"
    return out


def _overall(cfg, results):
    v = results["verdicts"]
    phen = [x["phenomenon"] for x in v.values()]
    if any(x.get("intervention_stage") == "MECHANISM_PILOT_CANDIDATE" for x in v.values()):
        return "MECHANISM_PILOT_CANDIDATE"
    if any(x.get("intervention_stage") == "STATE_REPAIR_BRANCH_ON_HOLD" for x in v.values()):
        return "STATE_REPAIR_BRANCH_ON_HOLD"
    if "INCOMPLETE" in phen or any(x.get("intervention_stage") == "INCOMPLETE" for x in v.values()):
        return "INCOMPLETE"
    return "PHENOMENON_NOT_ESTABLISHED"


def _fmt(vals):
    if not vals:
        return "n/a"
    return f"{sum(vals) / len(vals):+.4f} [" + ", ".join(f"{v:+.3f}" for v in vals) + "]"


def render(cfg, manifest) -> str:
    r = manifest.get("results", {})
    L = [f"# {manifest['run_id']}", "", f"- status: {manifest['status']}  overall: {r.get('overall')}",
         f"- git {manifest['git']['commit']} dirty={manifest['git']['dirty']}; config_sha256 {manifest['config_sha256'][:16]}; "
         f"code_sha256 {manifest['code_sha256'][:16]}",
         f"- CPU used {manifest.get('cpu_seconds_used', 0):.1f}s (cap {cfg['resource_caps']['max_cpu_seconds']}), "
         f"wall {manifest.get('wall_seconds', 0):.1f}s, peak RSS {manifest.get('peak_rss_kb', 0) / 1024:.1f} MiB", "",
         "## Tuning (dev seeds, online train accuracy over the stream)", ""]
    for cn, fams in r.get("tuning", {}).items():
        for fn, t in fams.items():
            L.append(f"- {cn}/{fn}: {t.get('status')} best={t.get('best')} edge={t.get('best_at_edge')} "
                     f"extended={t.get('extended')} rows={t.get('rows')}")
    L += ["", "## Phenomenon (base family; probe AUC deltas, mean [per seed])", ""]
    for cn, v in r.get("verdicts", {}).items():
        L.append(f"### {cn}: {v.get('phenomenon')} {v.get('reason', '')}")
        for k in ("d_probe_auc_acc_late_minus_early", "d_probe_early_acc_late_minus_early",
                  "d_probe_final_acc_late_minus_early", "d_probe_auc_acc_late_minus_fresh",
                  "d_probe_auc_acc_early_minus_fresh"):
            L.append(f"- {k}: {_fmt(v.get(k, []))}")
        if "per_intervention" in v:
            L.append(f"- intervention stage: {v.get('intervention_stage')}")
            for x, d in v["per_intervention"].items():
                L.append(f"  - {x}: d_auc_vs_keep_all {_fmt(d['d_auc_vs_keep_all'])}; "
                         f"vs norm-matched {_fmt(d.get('d_auc_vs_norm_matched', []))}; {d.get('verdict', d.get('role'))}")
        L.append("")
    L += ["## Secondary families (descriptive)", ""]
    for cn, fams in r.get("phenomenon", {}).items():
        for fn, ph in fams.items():
            if fn == cfg["base_family"]:
                continue
            L.append(f"- {cn}/{fn}: late-early AUC {_fmt(_deltas(ph['seeds'], 'late', 'early', 'probe_auc_acc'))}; "
                     f"late-fresh {_fmt(_deltas(ph['seeds'], 'late', 'fresh', 'probe_auc_acc'))}")
    L += ["", "## Cells", ""]
    counts: dict = {}
    for c in manifest.get("cells", []):
        counts[(c["cell"], c["status"])] = counts.get((c["cell"], c["status"]), 0) + 1
    for (c, s_), n in sorted(counts.items()):
        L.append(f"- {c}: {s_} x{n}")
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    m = run_stage(cfg, a.out, ["python", "-m", "osrepair.stage1", *argv])
    print(open(os.path.join(a.out, "summary.md")).read())
    return 0 if m["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

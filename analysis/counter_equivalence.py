"""Counter-policy equivalence: deterministic counterexample and raw-trace precision checks.

No training. Reads the existing stage-1 log only.

1. Counterexample. Two coordinates with constant gradients of very different scale, 10,000 steps of
   history, then one more step from the same (m, v) with the counter kept (t) or reset (r = 1).
   With eps = 0 the per-coordinate update ratio is the same scalar kappa(r, t) for both coordinates;
   with eps = 1e-8 it is not. Implementation values are compared with Eq. (ratio) of the manuscript.
2. Raw traces (runs/p4_stage1_0404c14/log.jsonl.gz). Only the first post-intervention step shares
   (m, v) across branches, so one-step ratios are checked at k = 1 only:
     reset_t / keep_all                -> kappa(1, t+1)
     reset_both(shared_keep) / reset_both -> 1 / kappa(1, t+1)
     reset_v(shared_keep) / reset_v       -> sqrt(b_{t+1} / b_1)
   The whole-trajectory comparison reset_t vs keepdir_reset_tmag uses continuous logged quantities.
3. Deterministic schedules implied by the algebra (eps = 0): kappa(k, t+k) and rho_k = 1/kappa(k, t+k).
4. Static check that the norm control computes its own direction (runner.make_norm_matcher).
"""

import gzip
import inspect
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from osrepair.interventions import apply_intervention  # noqa: E402
from osrepair.model import Param  # noqa: E402
from osrepair.optim import Adam  # noqa: E402
from osrepair import runner  # noqa: E402

B1, B2, LR = 0.9, 0.999, 1e-3
T_LATE = 10000  # optimizer steps at the late checkpoint (50 tasks x 200)
LOG = os.path.join(ROOT, "runs/p4_stage1_0404c14/log.jsonl.gz")
OUT = os.path.join(ROOT, "runs/p4_analysis_counter")


def a(s):
    return 1.0 - B1 ** s


def b(s):
    return 1.0 - B2 ** s


def kappa(r, t):
    return math.sqrt(b(r)) * a(t) / (a(r) * math.sqrt(b(t)))


def eq_ratio(r, t, v, eps):
    return kappa(r, t) * (math.sqrt(v) + eps * math.sqrt(b(t))) / (math.sqrt(v) + eps * math.sqrt(b(r)))


def counterexample(eps, g):
    p = {"w": Param([0.0, 0.0], (2,))}
    opt = Adam(p, lr=LR, betas=(B1, B2), eps=eps)
    for _ in range(T_LATE):
        opt.compute_update({"w": list(g)})
    keep_sd = opt.state_dict()
    reset_sd, _ = apply_intervention(keep_sd, {"kind": "reset_t"})
    out = {}
    for name, sd in (("keep", keep_sd), ("reset_t", reset_sd)):
        o = Adam({"w": Param([0.0, 0.0], (2,))}, lr=LR, betas=(B1, B2), eps=eps)
        o.load_state_dict(sd)
        out[name] = o.compute_update({"w": list(g)})["w"]
        out[name + "_v"] = o.state["w"]["v"]
    ratios = [out["reset_t"][i] / out["keep"][i] for i in range(2)]
    v = out["keep_v"]  # post-gradient v, identical in both branches
    pred = [eq_ratio(1, T_LATE + 1, v[i], eps) for i in range(2)]
    return {"eps": eps, "g": list(g), "ratio": ratios, "eq_ratio": pred,
            "max_rel_err_vs_eq": max(abs(x / y - 1) for x, y in zip(ratios, pred)),
            "kappa": kappa(1, T_LATE + 1), "coordinate_ratio_spread": ratios[1] / ratios[0]}


def traces():
    first, steps, evals = {}, {}, {}
    arms = ("keep_all", "reset_t", "reset_both", "reset_both__shared_keep", "reset_v", "reset_v__shared_keep",
            "keepdir_reset_tmag")
    with gzip.open(LOG, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") != "intervention" or r["arm"] not in arms:
                continue
            key = (r["cond"], r["seed"], r["arm"])
            if r["kind"] == "probe_step":
                if r["k"] == 1:
                    first[key] = r["update_norms"]
                steps.setdefault(key, {})[r["k"]] = r
            elif r["kind"] == "probe_eval":
                evals.setdefault(key, {})[r["k"]] = r
    t1 = T_LATE + 1
    checks = {"reset_t/keep_all": ("reset_t", "keep_all", kappa(1, t1)),
              "reset_both_sk/reset_both": ("reset_both__shared_keep", "reset_both", 1 / kappa(1, t1)),
              "reset_v_sk/reset_v": ("reset_v__shared_keep", "reset_v", math.sqrt(b(t1) / b(1)))}
    one_step = {}
    for name, (num, den, pred) in checks.items():
        devs, ratios = [], []
        for (cond, seed, arm), un in first.items():
            if arm != num:
                continue
            ud = first[(cond, seed, den)]
            for tensor in un:
                if ud[tensor] > 0:
                    ratio = un[tensor] / ud[tensor]
                    ratios.append(ratio)
                    devs.append(abs(ratio / pred - 1))
        one_step[name] = {"eps0_prediction": pred, "max_rel_dev": max(devs), "min_ratio": min(ratios),
                          "max_ratio": max(ratios), "n": len(devs)}
        if name == "reset_v_sk/reset_v":
            one_step[name]["per_tensor"] = {
                f"{cond}/{seed}/{tensor}": {"ratio": first[(cond, seed, num)][tensor] / first[(cond, seed, den)][tensor],
                                            "norm_decoupled": first[(cond, seed, den)][tensor]}
                for (cond, seed, arm) in first if arm == num for tensor in first[(cond, seed, num)]}
    traj = {"train_loss_max_abs": 0.0, "update_norm_max_rel": 0.0, "probe_loss_max_abs": 0.0,
            "probe_acc_max_abs": 0.0, "theta_rel_change_max_abs": 0.0, "cka_max_abs": 0.0,
            "bitwise_equal_train_loss_all": True, "n_pairs": 0}
    for (cond, seed, arm), st in steps.items():
        if arm != "reset_t":
            continue
        ctrl = steps[(cond, seed, "keepdir_reset_tmag")]
        traj["n_pairs"] += 1
        for k, r in st.items():
            c = ctrl[k]
            traj["train_loss_max_abs"] = max(traj["train_loss_max_abs"], abs(r["train_loss"] - c["train_loss"]))
            traj["bitwise_equal_train_loss_all"] &= r["train_loss"] == c["train_loss"]
            traj["update_norm_max_rel"] = max(traj["update_norm_max_rel"],
                                              abs(r["update_norm"] / c["update_norm"] - 1))
        e, ec = evals[(cond, seed, arm)], evals[(cond, seed, "keepdir_reset_tmag")]
        for k in e:
            traj["probe_loss_max_abs"] = max(traj["probe_loss_max_abs"], abs(e[k]["probe_loss"] - ec[k]["probe_loss"]))
            traj["probe_acc_max_abs"] = max(traj["probe_acc_max_abs"], abs(e[k]["probe_acc"] - ec[k]["probe_acc"]))
            traj["theta_rel_change_max_abs"] = max(traj["theta_rel_change_max_abs"],
                                                   abs(e[k]["theta_rel_change"] - ec[k]["theta_rel_change"]))
            if e[k].get("cka_to_branch") is not None:
                traj["cka_max_abs"] = max(traj["cka_max_abs"], abs(e[k]["cka_to_branch"] - ec[k]["cka_to_branch"]))
    return one_step, traj


def schedules():
    kap = [kappa(k, T_LATE + k) for k in range(1, 201)]
    rho = [1 / x for x in kap]
    kmin = min(range(200), key=kap.__getitem__)
    kmax = max(range(200), key=rho.__getitem__)
    return {"kappa_k1": kap[0], "kappa_min": kap[kmin], "kappa_argmin_k": kmin + 1, "kappa_k200": kap[-1],
            "rho_k1": rho[0], "rho_max": rho[kmax], "rho_argmax_k": kmax + 1, "rho_k200": rho[-1],
            "kappa_inf_limit": math.sqrt(1 - B2) / (1 - B1), "kappa_1_t1": kappa(1, T_LATE + 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    ce0 = counterexample(0.0, (1.0, 1e-8))
    ce = counterexample(1e-8, (1.0, 1e-8))
    one_step, traj = traces()
    sch = schedules()
    src = inspect.getsource(runner.make_norm_matcher)
    own_direction = ("n = math.sqrt(sum(x * x for x in d))" in src and "out[k] = [x * s for x in d]" in src
                     and "tgt[k] / n" in src)
    res = {"counterexample_eps0": ce0, "counterexample_eps1e-8": ce, "one_step": one_step,
           "trajectory_reset_t_vs_control": traj, "schedules": sch,
           "norm_control_uses_own_direction": own_direction,
           "per_coordinate_attribution": "CHECKPOINT_TRACE_UNAVAILABLE (late checkpoints not stored; logs hold "
                                         "per-tensor norms only)"}
    M = {}

    def m(name, value, fmt, path):
        M[name] = {"value": value, "fmt": fmt, "path": path}
    m("CeKappa", ce["kappa"], "{:.6f}", "counterexample_eps1e-8.kappa")
    m("CeKappaInf", sch["kappa_inf_limit"], "{:.6f}", "schedules.kappa_inf_limit")
    m("CeEpsZeroRatioA", ce0["ratio"][0], "{:.6f}", "counterexample_eps0.ratio[0]")
    m("CeEpsZeroRatioB", ce0["ratio"][1], "{:.6f}", "counterexample_eps0.ratio[1]")
    m("CeRatioA", ce["ratio"][0], "{:.6f}", "counterexample_eps1e-8.ratio[0]")
    m("CeRatioB", ce["ratio"][1], "{:.6f}", "counterexample_eps1e-8.ratio[1]")
    m("CeMaxErr", max(ce["max_rel_err_vs_eq"], ce0["max_rel_err_vs_eq"]), "{:.1e}",
      "max(counterexample_*.max_rel_err_vs_eq)")
    m("TraceResetTDev", one_step["reset_t/keep_all"]["max_rel_dev"], "{:.1e}", "one_step.reset_t/keep_all.max_rel_dev")
    m("TraceBothSkPred", one_step["reset_both_sk/reset_both"]["eps0_prediction"], "{:.3f}",
      "one_step.reset_both_sk/reset_both.eps0_prediction")
    m("TraceBothSkDev", one_step["reset_both_sk/reset_both"]["max_rel_dev"], "{:.1e}",
      "one_step.reset_both_sk/reset_both.max_rel_dev")
    m("TraceVSkPred", one_step["reset_v_sk/reset_v"]["eps0_prediction"], "{:.2f}",
      "one_step.reset_v_sk/reset_v.eps0_prediction")
    m("TraceVSkMin", one_step["reset_v_sk/reset_v"]["min_ratio"], "{:.3g}", "one_step.reset_v_sk/reset_v.min_ratio")
    m("TraceVSkMax", one_step["reset_v_sk/reset_v"]["max_ratio"], "{:.3g}", "one_step.reset_v_sk/reset_v.max_ratio")
    pt = one_step["reset_v_sk/reset_v"]["per_tensor"]
    m("TraceVSkEpsLimited", sum(abs(v["ratio"] - 1) < 0.01 for v in pt.values()), "{}",
      "count(one_step.reset_v_sk/reset_v.per_tensor[*].ratio within 1% of 1)")
    m("TraceVSkTensors", len(pt), "{}", "len(one_step.reset_v_sk/reset_v.per_tensor)")
    big = [v for v in pt.values() if v["norm_decoupled"] > 1.0]
    m("TraceVSkBigCount", len(big), "{}", "count(per_tensor with norm_decoupled > 1)")
    m("TraceVSkBigNearOne", sum(abs(v["ratio"] - 1) < 0.01 for v in big), "{}",
      "count(per_tensor with norm_decoupled > 1 and ratio within 1% of 1)")
    m("TraceVSkBigMaxRatio", max(v["ratio"] for v in big), "{:.2f}", "max(per_tensor[norm_decoupled > 1].ratio)")
    m("TraceVSkBiasMin", min(v["ratio"] for k, v in pt.items() if k.endswith("fc2.bias")), "{:.2f}",
      "min(one_step.reset_v_sk/reset_v.per_tensor[*/fc2.bias].ratio)")
    m("TrajLossDiff", traj["train_loss_max_abs"], "{:.1e}", "trajectory_reset_t_vs_control.train_loss_max_abs")
    m("TrajProbeLossDiff", traj["probe_loss_max_abs"], "{:.1e}", "trajectory_reset_t_vs_control.probe_loss_max_abs")
    m("TrajProbeAccDiff", traj["probe_acc_max_abs"], "{:.1e}", "trajectory_reset_t_vs_control.probe_acc_max_abs")
    m("TrajThetaDiff", traj["theta_rel_change_max_abs"], "{:.1e}",
      "trajectory_reset_t_vs_control.theta_rel_change_max_abs")
    m("SchedKappaMin", sch["kappa_min"], "{:.3f}", "schedules.kappa_min")
    m("SchedKappaArgmin", sch["kappa_argmin_k"], "{}", "schedules.kappa_argmin_k")
    m("SchedKappaEnd", sch["kappa_k200"], "{:.3f}", "schedules.kappa_k200")
    m("SchedRhoMax", sch["rho_max"], "{:.2f}", "schedules.rho_max")
    m("SchedRhoArgmax", sch["rho_argmax_k"], "{}", "schedules.rho_argmax_k")
    m("SchedRhoEnd", sch["rho_k200"], "{:.2f}", "schedules.rho_k200")
    res["macros"] = M
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "macros"}, indent=1))


if __name__ == "__main__":
    main()

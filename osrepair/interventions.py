"""Optimizer-state interventions applied at a branch point.

State-only interventions (weights, buffers, RNG and data position untouched):

    keep_all    no change
    reset_m     m <- 0
    reset_v     v <- 0
    reset_both  m <- 0, v <- 0
    mix         m <- rho_m * m + (1 - rho_m) * m_ref,  v likewise.
                The default reference is the freshly initialised state
                (m_ref = v_ref = 0, age 0), i.e. a soft reset.

Counter policy (what happens to the bias-correction ages):

    decoupled     each moment's age is set so that 1 - beta^age equals the EMA
                  weight that moment actually places on observed gradients.
                  A reset moment gets age 0; a mixed moment gets
                  age = log(1 - w) / log(beta), with
                  w = rho * (1 - beta^t) + (1 - rho) * (1 - beta^t_ref).
                  This keeps m_hat, v_hat consistent (unbiased under a
                  stationary gradient distribution).
    shared_keep   ages untouched. This is what zeroing torch's exp_avg /
                  exp_avg_sq without touching state['step'] does. After a
                  reset, m_hat is under-corrected (too small) and v_hat is
                  under-corrected (too small => update too large).
    shared_reset  both ages set to 0 whenever either moment is reset. This
                  is what deleting torch's state['step'] (or re-creating the
                  step) while keeping the other moment does; the kept
                  moment's correction is then far too large.

``shared_keep`` and ``shared_reset`` exist to measure bias-correction
artifacts; they are not proposed methods.

Branch-point hyperparameter changes (NOT state-only; used as controls):

    beta_switch   change (beta1, beta2) from the branch point on, converting
                  ages so each moment keeps its EMA weight:
                  age' = age * log(beta_old) / log(beta_new).
"""

from __future__ import annotations

import copy
import math

STATE_ONLY_KINDS = ("keep_all", "reset_m", "reset_v", "reset_both", "mix")
COUNTER_POLICIES = ("decoupled", "shared_keep", "shared_reset")


class NotApplicable(ValueError):
    """Intervention does not exist for this optimizer (e.g. reset_v on SGD)."""


def ema_weight(beta: float, age: float) -> float:
    """Total EMA weight on observed gradients after ``age`` updates: 1 - beta^age.

    Computed as -expm1(age * log(beta)) for precision at small ages. For large
    ages it saturates to exactly 1.0 in float64, after which the age cannot
    be recovered from the weight (see :func:`age_for_weight`).
    """
    if age == 0.0:
        return 0.0
    if beta == 0.0:
        return 1.0
    return -math.expm1(age * math.log(beta))


def age_for_weight(beta: float, w: float) -> float:
    """Inverse of :func:`ema_weight`: the age whose EMA weight is ``w``."""
    if w <= 0.0:
        return 0.0
    if not 0.0 < beta < 1.0:
        raise ValueError("fractional ages need 0 < beta < 1")
    if w >= 1.0:
        return math.inf
    return math.log1p(-w) / math.log(beta)


def mixed_age(beta: float, rho: float, age_a: float, age_b: float) -> float:
    """Age of the EMA ``rho * a + (1 - rho) * b`` so that bias correction stays exact."""
    if rho == 1.0:
        return age_a
    if rho == 0.0:
        return age_b
    w = rho * ema_weight(beta, age_a) + (1.0 - rho) * ema_weight(beta, age_b)
    age = age_for_weight(beta, w)
    if math.isinf(age):
        # Both EMA weights underflowed to exactly 1; the correction is 1 either way.
        return max(age_a, age_b)
    return age


def _mix_list(rho: float, a: list[float], b: list[float]) -> list[float]:
    if rho == 1.0:
        return list(a)
    if rho == 0.0:
        return list(b)
    return [rho * x + (1.0 - rho) * y for x, y in zip(a, b)]


def apply_adam_intervention(opt_sd: dict, spec: dict, reference_sd: dict | None = None) -> tuple[dict, dict]:
    """Return (new_state_dict, record). The input state dict is not modified."""
    kind = spec["kind"]
    sd = copy.deepcopy(opt_sd)
    hp = sd["hparams"]
    b1, b2 = hp["beta1"], hp["beta2"]
    record = {"kind": kind, "counter": spec.get("counter"), "state_only": kind in STATE_ONLY_KINDS}

    if kind == "keep_all":
        return sd, record

    if kind == "beta_switch":
        nb1, nb2 = spec.get("beta1", b1), spec.get("beta2", b2)
        for st in sd["state"].values():
            if nb1 != b1:
                st["t_m"] = st["t_m"] * math.log(b1) / math.log(nb1) if st["t_m"] > 0 else 0.0
            if nb2 != b2:
                st["t_v"] = st["t_v"] * math.log(b2) / math.log(nb2) if st["t_v"] > 0 else 0.0
        hp["beta1"], hp["beta2"] = nb1, nb2
        record.update({"beta1": [b1, nb1], "beta2": [b2, nb2]})
        return sd, record

    counter = spec.get("counter", "decoupled")
    if counter not in COUNTER_POLICIES:
        raise ValueError(f"unknown counter policy {counter!r}")

    if kind in ("reset_m", "reset_v", "reset_both"):
        do_m = kind in ("reset_m", "reset_both")
        do_v = kind in ("reset_v", "reset_both")
        for st in sd["state"].values():
            if do_m:
                st["m"] = [0.0] * len(st["m"])
            if do_v:
                st["v"] = [0.0] * len(st["v"])
            if counter == "decoupled":
                if do_m:
                    st["t_m"] = 0.0
                if do_v:
                    st["t_v"] = 0.0
            elif counter == "shared_reset":
                st["t_m"] = 0.0
                st["t_v"] = 0.0
        return sd, record

    if kind == "mix":
        rho_m, rho_v = float(spec["rho_m"]), float(spec["rho_v"])
        if not (0.0 <= rho_m <= 1.0 and 0.0 <= rho_v <= 1.0):
            raise ValueError("mixing coefficients must be in [0, 1]")
        if counter == "shared_reset":
            raise ValueError("shared_reset is undefined for mix")
        record.update({"rho_m": rho_m, "rho_v": rho_v, "reference": "fresh" if reference_sd is None else "given"})
        for name, st in sd["state"].items():
            if reference_sd is None:
                n = len(st["m"])
                ref = {"m": [0.0] * n, "v": [0.0] * n, "t_m": 0.0, "t_v": 0.0}
            else:
                ref = reference_sd["state"][name]
            new_tm = mixed_age(b1, rho_m, st["t_m"], ref["t_m"])
            new_tv = mixed_age(b2, rho_v, st["t_v"], ref["t_v"])
            st["m"] = _mix_list(rho_m, st["m"], ref["m"])
            st["v"] = _mix_list(rho_v, st["v"], ref["v"])
            if counter == "decoupled":
                st["t_m"], st["t_v"] = new_tm, new_tv
        return sd, record

    raise ValueError(f"unknown intervention kind {kind!r}")


def apply_sgd_intervention(opt_sd: dict, spec: dict) -> tuple[dict, dict]:
    kind = spec["kind"]
    sd = copy.deepcopy(opt_sd)
    record = {"kind": kind, "counter": None, "state_only": kind in STATE_ONLY_KINDS}
    if kind == "keep_all":
        return sd, record
    if kind == "reset_m":
        for st in sd["state"].values():
            st["buf"] = [0.0] * len(st["buf"])
            st["has_buf"] = False
        return sd, record
    if kind == "mix" and spec.get("rho_v", 1.0) == 1.0:
        rho = float(spec["rho_m"])
        for st in sd["state"].values():
            st["buf"] = _mix_list(rho, st["buf"], [0.0] * len(st["buf"]))
        return sd, record
    raise NotApplicable(f"{kind} is not defined for SGD")


def apply_intervention(opt_sd: dict, spec: dict, reference_sd: dict | None = None) -> tuple[dict, dict]:
    if opt_sd["kind"] == "adam":
        return apply_adam_intervention(opt_sd, spec, reference_sd)
    if opt_sd["kind"] == "sgd":
        return apply_sgd_intervention(opt_sd, spec)
    raise ValueError(f"unknown optimizer kind {opt_sd['kind']!r}")

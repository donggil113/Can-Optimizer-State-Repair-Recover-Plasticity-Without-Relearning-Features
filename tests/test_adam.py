"""Adam formula, step-counter and bias-correction tests.

Closed forms used below (constant gradient g after a moment was zeroed k
steps ago, with the moment's bias-correction age equal to a):

    m_k = (1 - b1^k) g            =>  m_hat = (1 - b1^k) / (1 - b1^a) * g
    v_k = (1 - b2^k) g^2          =>  v_hat = (1 - b2^k) / (1 - b2^a) * g^2

Correct (decoupled) ages have a = k, giving m_hat = g and v_hat = g^2.
These are implementation checks against the stated formulas, not proofs.
"""

import math
import unittest

from osrepair.interventions import (NotApplicable, age_for_weight, apply_intervention, ema_weight, mixed_age)
from osrepair.model import Param
from osrepair.optim import SGD, Adam, apply_update
from osrepair.rng import RNG, bitwise_equal

B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3


def make_adam(n=3, lr=LR, b1=B1, b2=B2, eps=EPS):
    params = {"w": Param([0.0] * n, (n,))}
    return params, Adam(params, lr=lr, betas=(b1, b2), eps=eps)


def step(params, opt, g):
    d = opt.compute_update({"w": list(g)})
    apply_update(params, d)
    return d["w"]


def intervene(opt, spec):
    sd, rec = apply_intervention(opt.state_dict(), spec)
    opt.load_state_dict(sd)
    return rec


def close(a, b, rel=1e-12, abs_=0.0):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


class TestAdamFormula(unittest.TestCase):
    def test_constant_gradient_fresh(self):
        g = [0.3, -2.0, 1e-3]
        params, opt = make_adam()
        for t in range(1, 60):
            d = step(params, opt, g)
            st = opt.state["w"]
            mh, vh = opt.bias_corrected("w")
            for i, gi in enumerate(g):
                self.assertTrue(close(st["m"][i], (1 - B1 ** t) * gi))
                self.assertTrue(close(st["v"][i], (1 - B2 ** t) * gi * gi))
                self.assertTrue(close(mh[i], gi))
                self.assertTrue(close(vh[i], gi * gi))
                self.assertTrue(close(d[i], -LR * gi / (abs(gi) + EPS)))

    def test_hand_computed_two_steps(self):
        params, opt = make_adam(n=1, lr=0.1)
        d1 = step(params, opt, [0.5])
        # t=1: m=0.05, v=0.00025, m_hat=0.5, v_hat=0.25 -> d = -0.1*0.5/(0.5+1e-8)
        self.assertTrue(close(d1[0], -0.1 * 0.5 / (0.5 + 1e-8)))
        d2 = step(params, opt, [-1.0])
        m2 = 0.9 * 0.05 + 0.1 * -1.0          # -0.055
        v2 = 0.999 * 0.00025 + 0.001 * 1.0    # 0.00124975
        mh, vh = m2 / (1 - 0.81), v2 / (1 - 0.999 ** 2)
        self.assertTrue(close(d2[0], -0.1 * mh / (math.sqrt(vh) + 1e-8)))
        self.assertTrue(close(params["w"].data[0], d1[0] + d2[0]))

    def test_matches_textbook_algorithm1(self):
        """Independent scalar transcription of Kingma & Ba Algorithm 1 (shared t)."""
        r = RNG(7)
        grads = [[r.gauss() * (1 + 5 * (t % 7 == 0)) for _ in range(4)] for t in range(200)]
        params, opt = make_adam(n=4, lr=3e-3, b1=0.8, b2=0.99, eps=1e-6)
        theta, m, v = [0.0] * 4, [0.0] * 4, [0.0] * 4
        for t, g in enumerate(grads, start=1):
            step(params, opt, g)
            for i in range(4):
                m[i] = 0.8 * m[i] + 0.2 * g[i]
                v[i] = 0.99 * v[i] + 0.01 * g[i] ** 2
                theta[i] -= 3e-3 * (m[i] / (1 - 0.8 ** t)) / (math.sqrt(v[i] / (1 - 0.99 ** t)) + 1e-6)
        for a, b in zip(params["w"].data, theta):
            self.assertTrue(close(a, b, rel=1e-12, abs_=1e-15))

    def test_ages_equal_shared_counter_in_ordinary_training(self):
        params, opt = make_adam()
        for t in range(1, 25):
            step(params, opt, [1.0, 2.0, 3.0])
            st = opt.state["w"]
            self.assertEqual(st["t_m"], float(t))
            self.assertEqual(st["t_v"], float(t))
            self.assertEqual(opt.num_steps, t)


class TestResetBiasCorrection(unittest.TestCase):
    T_OLD = 500  # steps of history before the reset

    def trained(self):
        params, opt = make_adam()
        r = RNG(11)
        for _ in range(self.T_OLD):
            step(params, opt, [r.gauss() * 0.1 for _ in range(3)])
        return params, opt

    def test_reset_m_decoupled_is_exactly_corrected(self):
        params, opt = self.trained()
        intervene(opt, {"kind": "reset_m", "counter": "decoupled"})
        g = [0.7, -0.2, 3.0]
        for _ in range(30):
            step(params, opt, g)
            mh, _ = opt.bias_corrected("w")
            for a, b in zip(mh, g):
                self.assertTrue(close(a, b))

    def test_reset_m_shared_keep_under_corrects(self):
        params, opt = self.trained()
        intervene(opt, {"kind": "reset_m", "counter": "shared_keep"})
        g = [0.7, -0.2, 3.0]
        for k in range(1, 30):
            step(params, opt, g)
            a = self.T_OLD + k
            factor = (1 - B1 ** k) / (1 - B1 ** a)
            mh, _ = opt.bias_corrected("w")
            for x, gi in zip(mh, g):
                self.assertTrue(close(x, factor * gi))
        # First step after reset: m_hat is ~(1 - b1) = 10% of the true mean.
        self.assertLess((1 - B1) / (1 - B1 ** (self.T_OLD + 1)), 0.1001)

    def test_reset_v_decoupled_is_exactly_corrected(self):
        params, opt = self.trained()
        intervene(opt, {"kind": "reset_v", "counter": "decoupled"})
        g = [0.7, -0.2, 3.0]
        for _ in range(30):
            step(params, opt, g)
            _, vh = opt.bias_corrected("w")
            for a, gi in zip(vh, g):
                self.assertTrue(close(a, gi * gi))

    def test_reset_v_shared_keep_inflates_update(self):
        """Zeroing v but keeping the counter blows the step up by ~1/sqrt(1-b2) ~ 31.6x."""
        params, opt = self.trained()
        ref_params, ref = self.trained()
        intervene(opt, {"kind": "reset_v", "counter": "shared_keep"})
        intervene(ref, {"kind": "reset_v", "counter": "decoupled"})
        g = [0.7, -0.2, 3.0]
        for k in range(1, 20):
            d = step(params, opt, g)
            d_ref = step(ref_params, ref, g)
            a = self.T_OLD + k
            vfac = (1 - B2 ** k) / (1 - B2 ** a)
            _, vh = opt.bias_corrected("w")
            for x, gi in zip(vh, g):
                self.assertTrue(close(x, vfac * gi * gi))
            if k == 1:
                ratio = abs(d[0]) / abs(d_ref[0])
                expected = (abs(0.7) + EPS) / (math.sqrt(vfac) * abs(0.7) + EPS)
                self.assertTrue(close(ratio, expected, rel=1e-9))
                # With 500 steps of history the inflation is sqrt((1-b2^501)/(1-b2)) ~ 19.9x;
                # it approaches 1/sqrt(1-b2) ~ 31.6x only for history >> 1/(1-b2).
                self.assertTrue(close(ratio, math.sqrt((1 - B2 ** (self.T_OLD + 1)) / (1 - B2)), rel=1e-6))

    def test_reset_artifacts_asymptotic_magnitude(self):
        """Long history (t >> 1/(1-b2)): first-step inflation -> 1/sqrt(1-b2) for reset_v and
        (1-b1)/sqrt(1-b2) for reset_both under shared_keep."""
        g = [0.7]
        out = {}
        for kind in ("reset_v", "reset_both"):
            for counter in ("shared_keep", "decoupled"):
                params, opt = make_adam(n=1)
                for _ in range(40000):
                    opt.compute_update({"w": [0.05]})
                intervene(opt, {"kind": kind, "counter": counter})
                out[kind, counter] = abs(step(params, opt, g)[0])
        self.assertTrue(close(out["reset_v", "shared_keep"] / out["reset_v", "decoupled"],
                              1 / math.sqrt(1 - B2), rel=1e-6))
        self.assertTrue(close(out["reset_both", "shared_keep"] / out["reset_both", "decoupled"],
                              (1 - B1) / math.sqrt(1 - B2), rel=1e-6))

    def test_reset_m_shared_reset_overcorrects_kept_v(self):
        """Resetting the shared counter makes the kept v's correction ~1/(1-b2) too large."""
        params, opt = self.trained()
        v_old = list(opt.state["w"]["v"])
        intervene(opt, {"kind": "reset_m", "counter": "shared_reset"})
        g = [0.7, -0.2, 3.0]
        step(params, opt, g)
        _, vh = opt.bias_corrected("w")
        for x, vo, gi in zip(vh, v_old, g):
            self.assertTrue(close(x, (B2 * vo + (1 - B2) * gi * gi) / (1 - B2)))

    def test_reset_both_decoupled_equals_fresh_adam(self):
        params, opt = self.trained()
        intervene(opt, {"kind": "reset_both", "counter": "decoupled"})
        fparams, fresh = make_adam()
        fparams["w"].data = list(params["w"].data)
        r = RNG(3)
        for _ in range(20):
            g = [r.gauss() for _ in range(3)]
            self.assertTrue(bitwise_equal(step(params, opt, g), step(fparams, fresh, g)))
        self.assertTrue(bitwise_equal(opt.state, fresh.state))

    def test_reset_both_shared_keep_first_step(self):
        params, opt = self.trained()
        intervene(opt, {"kind": "reset_both", "counter": "shared_keep"})
        g = [0.7, -0.2, 3.0]
        d = step(params, opt, g)
        a = self.T_OLD + 1
        for x, gi in zip(d, g):
            mh = (1 - B1) * gi / (1 - B1 ** a)
            vh = (1 - B2) * gi * gi / (1 - B2 ** a)
            self.assertTrue(close(x, -LR * mh / (math.sqrt(vh) + EPS)))
        # vs fresh Adam's |step| = LR: (1-b1)/sqrt(1-b2) * sqrt(1-b2^a)/(1-b1^a) ~ 1.99x at a=501
        ratio = abs(d[0]) / LR
        a = self.T_OLD + 1
        self.assertTrue(close(ratio, (1 - B1) / math.sqrt(1 - B2) * math.sqrt(1 - B2 ** a) / (1 - B1 ** a), rel=1e-6))


class TestResetT(unittest.TestCase):
    """Step-counter-only reset (Adam-Rel, Ellis et al. 2024), checked against their closed forms.

    Their Thm 3.1 (eps = 0): gradient g for t' -> inf steps, then k*g; at step t after the jump
        u_t = (b1^(t+1) + k(1 - b1^(t+1))) / sqrt(b2^(t+1) + k^2 (1 - b2^(t+1)))
    and with the counter reset (their Eq. 4) u_t is multiplied by sqrt(1 - b2^(t+1)) / (1 - b1^(t+1)).
    We check t = 0 (first step after the jump), with t' = 40000 standing in for infinity and
    g = 1: as k grows, u_0 -> (1-b1)/sqrt(1-b2) without reset and -> 1 with reset.
    Formulas read from arXiv:2412.17113 (HTML), 2026-09-26.
    """

    def first_step_ratio(self, k, reset_t, history=40000):
        params, opt = make_adam(n=1)
        for _ in range(history):
            opt.compute_update({"w": [1.0]})
        if reset_t:
            intervene(opt, {"kind": "reset_t"})
        return abs(step(params, opt, [k])[0]) / LR

    def test_first_step_formula(self):
        params, opt = make_adam(n=1)
        r = RNG(2)
        for _ in range(300):
            step(params, opt, [r.gauss()])
        m0, v0 = opt.state["w"]["m"][0], opt.state["w"]["v"][0]
        rec = intervene(opt, {"kind": "reset_t"})
        self.assertTrue(rec["state_only"])
        self.assertEqual(opt.state["w"]["m"][0], m0)   # moments untouched
        g = 0.4
        d = step(params, opt, [g])[0]
        mh = (B1 * m0 + (1 - B1) * g) / (1 - B1)
        vh = (B2 * v0 + (1 - B2) * g * g) / (1 - B2)
        self.assertTrue(close(d, -LR * mh / (math.sqrt(vh) + EPS)))

    def test_ellis_thm31_and_eq4_limits(self):
        for k in (1e4, 1e6):
            no_reset = self.first_step_ratio(k, reset_t=False)
            with_reset = self.first_step_ratio(k, reset_t=True)
            exact_no = (B1 + k * (1 - B1)) / math.sqrt(B2 + k * k * (1 - B2))
            exact_rel = ((B1 + k * (1 - B1)) / (1 - B1)) / math.sqrt((B2 + k * k * (1 - B2)) / (1 - B2))
            self.assertTrue(close(no_reset, exact_no, rel=1e-6))
            self.assertTrue(close(with_reset, exact_rel, rel=1e-6))
        self.assertTrue(close(self.first_step_ratio(1e6, False), (1 - B1) / math.sqrt(1 - B2), rel=1e-4))
        self.assertTrue(close(self.first_step_ratio(1e6, True), 1.0, rel=1e-4))


class TestMixing(unittest.TestCase):
    def trained(self, n_steps=40):
        params, opt = make_adam()
        r = RNG(5)
        for _ in range(n_steps):
            step(params, opt, [r.gauss() for _ in range(3)])
        return params, opt

    def test_age_inverse(self):
        # Bias correction depends on the EMA weight, so the weight must round-trip precisely.
        # The age itself is ill-conditioned near saturation (d age / d w = 1/(beta^age * |log beta|)),
        # so it is only required to round-trip where beta^age is not tiny.
        for beta in (0.5, 0.9, 0.999):
            for age in (0.0, 0.3, 1.0, 17.0, 250.5):
                if beta ** age < 1e-15:
                    continue  # saturated; covered below
                w = ema_weight(beta, age)
                self.assertTrue(close(ema_weight(beta, age_for_weight(beta, w)), w, rel=1e-12, abs_=1e-300))
                if beta ** age > 1e-6:
                    self.assertTrue(close(age_for_weight(beta, w), age, rel=1e-9, abs_=1e-12))

    def test_saturated_age(self):
        # 0.5^250.5 underflows relative to 1.0: the weight is exactly 1 and the age is lost.
        self.assertEqual(ema_weight(0.5, 250.5), 1.0)
        self.assertEqual(age_for_weight(0.5, 1.0), math.inf)
        # Mixing two saturated moments falls back to the larger age (correction is 1 either way).
        self.assertEqual(mixed_age(0.5, 0.3, 250.5, 400.0), 400.0)
        # Mixing a saturated moment with a fresh one is still exact.
        self.assertTrue(close(ema_weight(0.5, mixed_age(0.5, 0.3, 250.5, 0.0)), 0.3))

    def test_rho_one_is_keep_and_rho_zero_is_reset(self):
        _, opt = self.trained()
        sd = opt.state_dict()
        keep, _ = apply_intervention(sd, {"kind": "mix", "rho_m": 1.0, "rho_v": 1.0})
        self.assertTrue(bitwise_equal(keep, sd))
        zero, _ = apply_intervention(sd, {"kind": "mix", "rho_m": 0.0, "rho_v": 0.0})
        reset, _ = apply_intervention(sd, {"kind": "reset_both", "counter": "decoupled"})
        self.assertTrue(bitwise_equal(zero, reset))

    def test_decoupled_mix_preserves_bias_corrected_moments(self):
        _, opt = self.trained()
        before = opt.bias_corrected("w")
        for rho_m, rho_v in ((0.5, 0.5), (0.1, 0.9), (0.99, 0.01)):
            sd, _ = apply_intervention(opt.state_dict(), {"kind": "mix", "rho_m": rho_m, "rho_v": rho_v})
            _, o2 = make_adam()
            o2.load_state_dict(sd)
            after = o2.bias_corrected("w")
            for x, y in zip(before[0] + before[1], after[0] + after[1]):
                self.assertTrue(close(x, y, rel=1e-9))
            st = sd["state"]["w"]
            t0 = opt.state["w"]["t_m"]
            self.assertTrue(close(ema_weight(B1, st["t_m"]), rho_m * ema_weight(B1, t0)))
            self.assertTrue(close(ema_weight(B2, st["t_v"]), rho_v * ema_weight(B2, t0), rel=1e-9))

    def test_shared_keep_mix_scales_bias_corrected_moments(self):
        _, opt = self.trained()
        mh0, vh0 = opt.bias_corrected("w")
        sd, _ = apply_intervention(opt.state_dict(),
                                   {"kind": "mix", "rho_m": 0.3, "rho_v": 0.6, "counter": "shared_keep"})
        _, o2 = make_adam()
        o2.load_state_dict(sd)
        mh, vh = o2.bias_corrected("w")
        for a, b in zip(mh, mh0):
            self.assertTrue(close(a, 0.3 * b))
        for a, b in zip(vh, vh0):
            self.assertTrue(close(a, 0.6 * b))

    def test_mix_with_reference_state(self):
        _, a = self.trained(40)
        _, b = self.trained(7)
        sd, rec = apply_intervention(a.state_dict(), {"kind": "mix", "rho_m": 0.25, "rho_v": 0.5}, b.state_dict())
        self.assertEqual(rec["reference"], "given")
        st, sa, sb = sd["state"]["w"], a.state["w"], b.state["w"]
        for i in range(3):
            self.assertTrue(close(st["m"][i], 0.25 * sa["m"][i] + 0.75 * sb["m"][i]))
            self.assertTrue(close(st["v"][i], 0.5 * sa["v"][i] + 0.5 * sb["v"][i]))
        self.assertTrue(close(st["t_m"], mixed_age(B1, 0.25, 40.0, 7.0)))
        self.assertTrue(close(ema_weight(B1, st["t_m"]),
                              0.25 * ema_weight(B1, 40.0) + 0.75 * ema_weight(B1, 7.0)))

    def test_beta_switch_preserves_ema_weight(self):
        _, opt = self.trained()
        mh0, vh0 = opt.bias_corrected("w")
        sd, rec = apply_intervention(opt.state_dict(), {"kind": "beta_switch", "beta1": 0.9, "beta2": 0.9})
        self.assertFalse(rec["state_only"])
        _, o2 = make_adam(b2=0.9)
        o2.load_state_dict(sd)
        self.assertEqual(o2.hparams["beta2"], 0.9)
        self.assertTrue(close(ema_weight(0.9, sd["state"]["w"]["t_v"]), ema_weight(B2, 40.0), rel=1e-9))
        mh, vh = o2.bias_corrected("w")
        for x, y in zip(mh0 + vh0, mh + vh):
            self.assertTrue(close(x, y, rel=1e-9))

    def test_interventions_do_not_mutate_input(self):
        _, opt = self.trained()
        sd = opt.state_dict()
        frozen = opt.state_dict()
        for spec in ({"kind": "reset_m"}, {"kind": "reset_v"}, {"kind": "reset_both"},
                     {"kind": "mix", "rho_m": 0.5, "rho_v": 0.5}, {"kind": "beta_switch", "beta2": 0.9}):
            apply_intervention(sd, spec)
            self.assertTrue(bitwise_equal(sd, frozen), spec)

    def test_rejects_bad_specs(self):
        _, opt = self.trained()
        sd = opt.state_dict()
        with self.assertRaises(ValueError):
            apply_intervention(sd, {"kind": "mix", "rho_m": 1.5, "rho_v": 0.5})
        with self.assertRaises(ValueError):
            apply_intervention(sd, {"kind": "reset_m", "counter": "bogus"})
        with self.assertRaises(ValueError):
            apply_intervention(sd, {"kind": "mix", "rho_m": 0.5, "rho_v": 0.5, "counter": "shared_reset"})
        with self.assertRaises(ValueError):
            apply_intervention(sd, {"kind": "unknown"})


class TestSGD(unittest.TestCase):
    def test_torch_style_momentum_and_reset(self):
        params = {"w": Param([0.0, 0.0], (2,))}
        opt = SGD(params, lr=0.1, momentum=0.9)
        d1 = opt.compute_update({"w": [1.0, -2.0]})
        self.assertEqual(d1["w"], [-0.1, 0.2])          # buf initialised to g
        d2 = opt.compute_update({"w": [1.0, -2.0]})
        self.assertTrue(close(d2["w"][0], -0.1 * 1.9))
        sd, _ = apply_intervention(opt.state_dict(), {"kind": "reset_m"})
        opt.load_state_dict(sd)
        d3 = opt.compute_update({"w": [1.0, -2.0]})
        self.assertEqual(d3["w"], [-0.1, 0.2])
        with self.assertRaises(NotApplicable):
            apply_intervention(opt.state_dict(), {"kind": "reset_v"})


if __name__ == "__main__":
    unittest.main()

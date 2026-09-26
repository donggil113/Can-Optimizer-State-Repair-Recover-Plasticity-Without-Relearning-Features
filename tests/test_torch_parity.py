"""Parity against PyTorch (reference implementation), float64, CPU, 1 thread.

Runs only where torch is importable (the repo-local .venv). Under the system
Python these tests are SKIPPED, which is not a pass; see STATUS.md for the
logged run.

What is compared:
  * MLP forward, loss, parameter gradients and BatchNorm buffers vs torch
    modules with the same weights, batch and dropout mask.
  * Adam weight / m / v / update / step traces vs torch.optim.Adam with
    fixed betas and weight_decay=0, amsgrad=False, maximize=False,
    foreach=False, fused=False, capturable=False, eps added after sqrt(v_hat).
  * Interventions: which ones equal a stock manipulation of torch's Adam
    state (fresh optimizer, zeroing exp_avg / exp_avg_sq / step), and which
    ones (decoupled single-moment resets, mixing) have NO stock equivalent
    and are validated only by the explicit formula tests in test_adam.py.
"""

import math
import unittest
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
    torch.set_num_threads(1)
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on environment
    HAVE_TORCH = False

from osrepair.interventions import apply_intervention
from osrepair.model import MLP, Param
from osrepair.optim import Adam, apply_update
from osrepair.rng import RNG

LR, B1, B2, EPS = 3e-3, 0.9, 0.999, 1e-8


def max_rel(a, b):
    worst = 0.0
    for x, y in zip(a, b):
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-300))
    return worst


@unittest.skipUnless(HAVE_TORCH, "torch not installed (SKIP is not PASS)")
class TestModelParity(unittest.TestCase):
    def build(self):
        m = MLP(6, 9, 4, dropout=0.3, init_rng=RNG(3))
        r = RNG(4)
        m.params["bn.weight"].data = [0.5 + r.random() for _ in range(9)]
        m.params["bn.bias"].data = [0.2 * r.gauss() for _ in range(9)]
        fc1 = torch.nn.Linear(6, 9, bias=False, dtype=torch.float64)
        bn = torch.nn.BatchNorm1d(9, eps=1e-5, momentum=0.1, dtype=torch.float64)
        fc2 = torch.nn.Linear(9, 4, dtype=torch.float64)
        with torch.no_grad():
            fc1.weight.copy_(torch.tensor(m.params["fc1.weight"].data, dtype=torch.float64).view(9, 6))
            bn.weight.copy_(torch.tensor(m.params["bn.weight"].data, dtype=torch.float64))
            bn.bias.copy_(torch.tensor(m.params["bn.bias"].data, dtype=torch.float64))
            fc2.weight.copy_(torch.tensor(m.params["fc2.weight"].data, dtype=torch.float64).view(4, 9))
            fc2.bias.copy_(torch.tensor(m.params["fc2.bias"].data, dtype=torch.float64))
        return m, (fc1, bn, fc2)

    def test_forward_backward_and_buffers(self):
        m, (fc1, bn, fc2) = self.build()
        r = RNG(5)
        x = [[r.gauss() for _ in range(6)] for _ in range(12)]
        y = [r.randrange(4) for _ in range(12)]
        mask = m.draw_dropout_mask(12, RNG(6))
        cache = m.forward(x, train=True, mask=mask)
        loss, dl, _ = MLP.cross_entropy(cache.logits, y)
        grads = m.backward(cache, dl)

        bn.train()
        xt = torch.tensor(x, dtype=torch.float64)
        h = torch.relu(bn(fc1(xt))) * torch.tensor(mask, dtype=torch.float64)
        logits = fc2(h)
        tl = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
        tl.backward()
        self.assertLess(max_rel([v for row in cache.logits for v in row], logits.flatten().tolist()), 1e-12)
        self.assertLess(abs(loss - tl.item()) / abs(loss), 1e-13)
        pairs = {"fc1.weight": fc1.weight, "bn.weight": bn.weight, "bn.bias": bn.bias,
                 "fc2.weight": fc2.weight, "fc2.bias": fc2.bias}
        for k, p in pairs.items():
            self.assertLess(max_rel(grads[k], p.grad.flatten().tolist()), 1e-10, k)
        self.assertLess(max_rel(m.buffers["bn.running_mean"], bn.running_mean.tolist()), 1e-12)
        self.assertLess(max_rel(m.buffers["bn.running_var"], bn.running_var.tolist()), 1e-12)
        self.assertEqual(m.buffers["bn.num_batches_tracked"][0], int(bn.num_batches_tracked))
        bn.eval()
        with torch.no_grad():
            ev = fc2(torch.relu(bn(fc1(xt))))
        self.assertLess(max_rel([v for row in m.forward(x, train=False).logits for v in row],
                                ev.flatten().tolist()), 1e-12)


class _Pair:
    """Our Adam and torch.optim.Adam driven by identical gradient sequences."""

    def __init__(self, n=5, lr=LR, betas=(B1, B2)):
        self.p = {"w": Param([0.1 * i for i in range(n)], (n,))}
        self.ours = Adam(self.p, lr=lr, betas=betas, eps=EPS)
        self.tp = torch.nn.Parameter(torch.tensor(self.p["w"].data, dtype=torch.float64))
        self.theirs = torch.optim.Adam([self.tp], lr=lr, betas=betas, eps=EPS, weight_decay=0.0, amsgrad=False,
                                       maximize=False, foreach=False, fused=False, capturable=False)

    def step(self, g):
        before = self.tp.detach().clone()
        d = self.ours.compute_update({"w": list(g)})["w"]
        apply_update(self.p, {"w": d})
        self.tp.grad = torch.tensor(g, dtype=torch.float64)
        self.theirs.step()
        return d, (self.tp.detach() - before).tolist()

    def tstate(self):
        return self.theirs.state[self.tp]

    def compare(self, tc, tol):
        st = self.ours.state["w"]
        ts = self.tstate()
        tc.assertLess(max_rel(self.p["w"].data, self.tp.detach().tolist()), tol)
        tc.assertLess(max_rel(st["m"], ts["exp_avg"].tolist()), tol)
        tc.assertLess(max_rel(st["v"], ts["exp_avg_sq"].tolist()), tol)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (SKIP is not PASS)")
class TestAdamTraceParity(unittest.TestCase):
    def grads(self, n_steps, n=5, seed=7):
        r = RNG(seed)
        return [[r.gauss() * (1 + 4 * (t % 11 == 0)) for _ in range(n)] for t in range(n_steps)]

    def test_vanilla_trace(self):
        pair = _Pair()
        worst_upd = 0.0
        for t, g in enumerate(self.grads(400), start=1):
            d, td = pair.step(g)
            worst_upd = max(worst_upd, max_rel(d, td))
            pair.compare(self, 1e-12)
            st = pair.ours.state["w"]
            self.assertEqual(st["t_m"], float(t))
            self.assertEqual(st["t_v"], float(t))
            self.assertEqual(int(pair.tstate()["step"]), t)
        # Update difference comes from sqrt(v/bc2) vs sqrt(v)/sqrt(bc2) and the
        # subtraction used to read torch's update; both are rounding-level.
        self.assertLess(worst_upd, 1e-9)

    def test_other_fixed_betas(self):
        pair = _Pair(betas=(0.8, 0.99), lr=1e-2)
        for g in self.grads(200, seed=8):
            pair.step(g)
            pair.compare(self, 1e-12)

    def _intervened(self, ours_spec, torch_op, n_hist=150, n_after=60):
        pair = _Pair()
        gs = self.grads(n_hist + n_after, seed=9)
        for g in gs[:n_hist]:
            pair.step(g)
        sd, _ = apply_intervention(pair.ours.state_dict(), ours_spec)
        pair.ours.load_state_dict(sd)
        torch_op(pair)
        worst = 0.0
        for g in gs[n_hist:]:
            d, td = pair.step(g)
            worst = max(worst, max_rel(d, td), max_rel(pair.p["w"].data, pair.tp.detach().tolist()))
        return worst

    def test_stock_equivalents(self):
        def fresh(pair):
            pair.theirs = torch.optim.Adam([pair.tp], lr=LR, betas=(B1, B2), eps=EPS, foreach=False,
                                           fused=False)

        def zero(*keys):
            def op(pair):
                for k in keys:
                    pair.tstate()[k].zero_()
            return op

        cases = {
            "reset_both/decoupled == new torch Adam": ({"kind": "reset_both", "counter": "decoupled"}, fresh),
            "reset_t == zero step only (Adam-Rel)": ({"kind": "reset_t"}, zero("step")),
            "reset_v/shared_keep == zero exp_avg_sq only": ({"kind": "reset_v", "counter": "shared_keep"},
                                                            zero("exp_avg_sq")),
            "reset_m/shared_keep == zero exp_avg only": ({"kind": "reset_m", "counter": "shared_keep"},
                                                         zero("exp_avg")),
            "reset_both/shared_keep == zero both moments, keep step": (
                {"kind": "reset_both", "counter": "shared_keep"}, zero("exp_avg", "exp_avg_sq")),
            "reset_m/shared_reset == zero exp_avg and step": ({"kind": "reset_m", "counter": "shared_reset"},
                                                             zero("exp_avg", "step")),
        }
        for name, (spec, op) in cases.items():
            self.assertLess(self._intervened(spec, op), 1e-9, name)

    def test_counter_counterexample_in_torch(self):
        """Stock torch: zeroing `step` rescales coordinates differently when eps > 0."""
        ratios = {}
        for eps in (0.0, EPS):
            ps = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)) for _ in range(2)]
            opts = [torch.optim.Adam([p], lr=LR, betas=(B1, B2), eps=eps, foreach=False, fused=False) for p in ps]
            g = torch.tensor([1.0, 1e-8], dtype=torch.float64)
            for _ in range(10000):
                for p, o in zip(ps, opts):
                    p.grad = g.clone()
                    o.step()
            opts[1].state[ps[1]]["step"].zero_()
            before = [p.detach().clone() for p in ps]
            for p, o in zip(ps, opts):
                p.grad = g.clone()
                o.step()
            u = [(p.detach() - b0) for p, b0 in zip(ps, before)]
            ratios[eps] = (u[1] / u[0]).tolist()
        self.assertLess(abs(ratios[0.0][0] / ratios[0.0][1] - 1), 1e-9)
        self.assertGreater(ratios[EPS][1] / ratios[EPS][0], 1.5)

    def test_decoupled_single_moment_and_mix_have_no_stock_equivalent(self):
        def zero(*keys):
            def op(pair):
                for k in keys:
                    pair.tstate()[k].zero_()
            return op

        stock_ops = [zero("exp_avg"), zero("exp_avg_sq"), zero("exp_avg", "step"), zero("exp_avg_sq", "step"),
                     zero("step")]
        for spec in ({"kind": "reset_m", "counter": "decoupled"}, {"kind": "reset_v", "counter": "decoupled"},
                     {"kind": "mix", "rho_m": 0.5, "rho_v": 0.5, "counter": "decoupled"}):
            for op in stock_ops:
                self.assertGreater(self._intervened(spec, op), 1e-6, (spec, op))


if __name__ == "__main__":
    unittest.main()

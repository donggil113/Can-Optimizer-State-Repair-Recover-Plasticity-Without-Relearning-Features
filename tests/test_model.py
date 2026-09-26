"""Model tests: finite-difference gradients, BatchNorm buffers, dropout RNG use."""

import math
import unittest

from osrepair.model import MLP
from osrepair.rng import RNG, bitwise_equal


def batch(rng, B=6, D=5, C=3):
    x = [[rng.gauss() for _ in range(D)] for _ in range(B)]
    y = [rng.randrange(C) for _ in range(B)]
    return x, y


class TestGradients(unittest.TestCase):
    def check(self, norm, dropout, train):
        rng = RNG(1)
        m = MLP(5, 7, 3, dropout=dropout, norm=norm, init_rng=RNG(2))
        if norm == "batchnorm":
            # non-trivial affine and running stats
            m.params["bn.weight"].data = [0.5 + rng.random() for _ in range(7)]
            m.params["bn.bias"].data = [rng.gauss() * 0.3 for _ in range(7)]
            m.buffers["bn.running_mean"] = [rng.gauss() * 0.2 for _ in range(7)]
            m.buffers["bn.running_var"] = [0.5 + rng.random() for _ in range(7)]
        x, y = batch(rng)
        mask = m.draw_dropout_mask(len(x), RNG(9)) if (train and dropout > 0) else None

        def loss_fn():
            c = m.forward(x, train=train, mask=mask, update_buffers=False)
            return MLP.cross_entropy(c.logits, y)[0]

        c = m.forward(x, train=train, mask=mask, update_buffers=False)
        _, dl, _ = MLP.cross_entropy(c.logits, y)
        grads = m.backward(c, dl)
        h = 1e-6
        worst = 0.0
        for name, p in m.params.items():
            for i in range(p.numel):
                old = p.data[i]
                p.data[i] = old + h
                lp = loss_fn()
                p.data[i] = old - h
                lm = loss_fn()
                p.data[i] = old
                num = (lp - lm) / (2 * h)
                err = abs(num - grads[name][i]) / max(1e-6, abs(num) + abs(grads[name][i]))
                worst = max(worst, err)
        self.assertLess(worst, 1e-5, f"norm={norm} dropout={dropout} train={train}")

    def test_bn_train_dropout(self):
        self.check("batchnorm", 0.3, True)

    def test_bn_eval(self):
        self.check("batchnorm", 0.3, False)

    def test_no_norm_train(self):
        self.check("none", 0.2, True)

    def test_no_norm_no_dropout(self):
        self.check("none", 0.0, True)


class TestBatchNormBuffers(unittest.TestCase):
    def test_running_stats_follow_torch_convention(self):
        rng = RNG(3)
        m = MLP(4, 5, 2, dropout=0.0, init_rng=RNG(4), bn_momentum=0.1)
        x, _ = batch(rng, B=8, D=4, C=2)
        rm0, rv0 = list(m.buffers["bn.running_mean"]), list(m.buffers["bn.running_var"])
        c = m.forward(x, train=True)
        B = len(x)
        for i in range(5):
            col = [c.z1[b][i] for b in range(B)]
            mu = sum(col) / B
            var_u = sum((v - mu) ** 2 for v in col) / (B - 1)
            self.assertTrue(math.isclose(m.buffers["bn.running_mean"][i], 0.9 * rm0[i] + 0.1 * mu, rel_tol=1e-12))
            self.assertTrue(math.isclose(m.buffers["bn.running_var"][i], 0.9 * rv0[i] + 0.1 * var_u, rel_tol=1e-12))
        self.assertEqual(m.buffers["bn.num_batches_tracked"], [1])

    def test_eval_forward_is_pure(self):
        rng = RNG(5)
        m = MLP(4, 5, 2, dropout=0.5, init_rng=RNG(4))
        x, _ = batch(rng, B=8, D=4, C=2)
        m.forward(x, train=True, dropout_rng=RNG(0))
        before = m.state()
        out1 = m.forward(x, train=False).logits
        out2 = m.forward(x, train=False).logits
        self.assertTrue(bitwise_equal(before, m.state()))
        self.assertTrue(bitwise_equal(out1, out2))

    def test_update_buffers_false_leaves_buffers(self):
        rng = RNG(6)
        m = MLP(4, 5, 2, init_rng=RNG(4))
        x, _ = batch(rng, B=8, D=4, C=2)
        before = m.state()
        m.forward(x, train=True, update_buffers=False)
        self.assertTrue(bitwise_equal(before, m.state()))


class TestDropout(unittest.TestCase):
    def test_eval_consumes_no_rng_and_train_is_deterministic(self):
        rng = RNG(7)
        m = MLP(4, 6, 2, dropout=0.5, init_rng=RNG(4))
        x, _ = batch(rng, B=8, D=4, C=2)
        r = RNG(11)
        s0 = r.get_state()
        m.forward(x, train=False)
        self.assertEqual(r.get_state(), s0)
        r1, r2 = RNG(11), RNG(11)
        c1 = m.forward(x, train=True, dropout_rng=r1, update_buffers=False)
        c2 = m.forward(x, train=True, dropout_rng=r2, update_buffers=False)
        self.assertTrue(bitwise_equal(c1.mask, c2.mask))
        self.assertEqual(r1.get_state(), r2.get_state())
        vals = {v for row in c1.mask for v in row}
        self.assertTrue(vals <= {0.0, 2.0})

    def test_train_dropout_requires_randomness_source(self):
        m = MLP(4, 6, 2, dropout=0.5, init_rng=RNG(4))
        x, _ = batch(RNG(1), B=4, D=4, C=2)
        with self.assertRaises(ValueError):
            m.forward(x, train=True)


if __name__ == "__main__":
    unittest.main()

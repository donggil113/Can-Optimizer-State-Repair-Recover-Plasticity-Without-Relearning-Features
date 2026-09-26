"""Checkpoint branching tests: independence, determinism, state-only invariance.

The negative controls make sure ``verify_state_only`` actually detects each
kind of non-optimizer change; without them a vacuous checker would pass.
"""

import random
import unittest

from osrepair.rng import bitwise_equal
from osrepair.stream import TaskStream
from osrepair.trainer import (adam_state_diagnostics, branch, hash_parts, init_state, restore, snapshot,
                              train_step, verify_state_only)

MODEL = {"in_dim": 6, "hidden": 10, "out_dim": 3, "dropout": 0.2, "norm": "batchnorm"}
ADAM = {"kind": "adam", "lr": 3e-3}
STATE_ONLY = [
    {"kind": "keep_all"},
    {"kind": "reset_m", "counter": "decoupled"},
    {"kind": "reset_v", "counter": "decoupled"},
    {"kind": "reset_both", "counter": "decoupled"},
    {"kind": "mix", "rho_m": 0.5, "rho_v": 0.5, "counter": "decoupled"},
    {"kind": "reset_v", "counter": "shared_keep"},
    {"kind": "reset_m", "counter": "shared_reset"},
]


def setup(steps=40, seed=3000):
    stream = TaskStream(seed, in_dim=6, n_classes=3, n_tasks=3, steps_per_task=30, batch_size=8, eval_size=32)
    st = init_state(MODEL, ADAM, seed)
    for _ in range(steps):
        train_step(st, stream)
    return stream, st


def run(st, stream, n):
    return [train_step(st, stream) for _ in range(n)]


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip_and_independence(self):
        _, st = setup()
        ck = snapshot(st)
        a, b = restore(ck), restore(ck)
        self.assertEqual(hash_parts(snapshot(a)), hash_parts(ck))
        a.model.params["fc1.weight"].data[0] += 1.0
        a.optimizer.state["fc1.weight"]["m"][0] += 1.0
        a.model.buffers["bn.running_mean"][0] += 1.0
        a.rngs["data"].random()
        self.assertEqual(hash_parts(snapshot(b)), hash_parts(ck))
        # The checkpoint itself was not aliased by the mutated branch.
        self.assertEqual(hash_parts(snapshot(restore(ck))), hash_parts(snapshot(b)))

    def test_checkpoint_does_not_perturb_trajectory(self):
        stream, st = setup(steps=0)
        uninterrupted = restore(snapshot(st))
        run(uninterrupted, stream, 50)
        run(st, stream, 20)
        resumed = restore(snapshot(st))
        run(resumed, stream, 30)
        self.assertTrue(bitwise_equal(snapshot(uninterrupted), snapshot(resumed)))

    def test_sibling_keep_all_branches_are_bit_identical(self):
        stream, st = setup()
        ck = snapshot(st)
        a, _ = branch(ck, {"kind": "keep_all"})
        b, _ = branch(ck, {"kind": "keep_all"})
        ra, rb = run(a, stream, 25), run(b, stream, 25)
        self.assertEqual([r["batch_hash"] for r in ra], [r["batch_hash"] for r in rb])
        self.assertEqual([r["dropout_hash"] for r in ra], [r["dropout_hash"] for r in rb])
        self.assertTrue(bitwise_equal(snapshot(a), snapshot(b)))

    def test_interventions_keep_batches_and_masks_paired_but_change_trajectory(self):
        stream, st = setup()
        ck = snapshot(st)
        ref, _ = branch(ck, {"kind": "keep_all"})
        rr = run(ref, stream, 20)
        for spec in STATE_ONLY[1:]:
            br, _ = branch(ck, spec)
            rb = run(br, stream, 20)
            self.assertEqual([r["batch_hash"] for r in rb], [r["batch_hash"] for r in rr], spec)
            self.assertEqual([r["dropout_hash"] for r in rb], [r["dropout_hash"] for r in rr], spec)
            self.assertFalse(bitwise_equal(snapshot(br)["model"]["params"], snapshot(ref)["model"]["params"]),
                             f"{spec} had no effect on the trajectory")

    def test_training_leaves_global_random_untouched(self):
        random.seed(123)
        before = random.getstate()
        stream, st = setup(steps=10)
        adam_state_diagnostics(st, stream)
        verify_state_only(st, restore(snapshot(st)), stream.probe_set(8))
        self.assertEqual(random.getstate(), before)


class TestStateOnlyInvariance(unittest.TestCase):
    def setUp(self):
        self.stream, st = setup()
        self.ck = snapshot(st)
        self.probe = self.stream.probe_set(16)

    def test_all_state_only_interventions_pass(self):
        for spec in STATE_ONLY:
            br, rec = branch(self.ck, spec)
            chk = verify_state_only(restore(self.ck), br, self.probe)
            self.assertTrue(rec["state_only"])
            self.assertTrue(chk["state_only"], (spec, chk))
            self.assertTrue(chk["weights_equal"] and chk["eval_forward_equal"] and chk["train_forward_equal"])
            self.assertEqual(chk["optimizer_state_changed"], spec["kind"] != "keep_all", spec)

    def test_check_has_no_side_effects(self):
        ref, br = restore(self.ck), branch(self.ck, {"kind": "reset_both"})[0]
        h_ref, h_br = hash_parts(snapshot(ref)), hash_parts(snapshot(br))
        verify_state_only(ref, br, self.probe)
        self.assertEqual(hash_parts(snapshot(ref)), h_ref)
        self.assertEqual(hash_parts(snapshot(br)), h_br)

    def _expect_detected(self, mutate, key):
        br = restore(self.ck)
        mutate(br)
        chk = verify_state_only(restore(self.ck), br, self.probe)
        self.assertFalse(chk["state_only"], key)
        self.assertFalse(chk[key], key)

    def test_detects_weight_change(self):
        def shrink_perturb(s):
            p = s.model.params["fc1.weight"]
            p.data = [0.9 * w for w in p.data]
        self._expect_detected(shrink_perturb, "weights_equal")

    def test_detects_tiny_weight_change(self):
        def nudge(s):
            p = s.model.params["fc2.bias"]
            p.data[0] = p.data[0] + abs(p.data[0]) * 1e-15 + 1e-300
        self._expect_detected(nudge, "weights_equal")

    def test_detects_buffer_change(self):
        def bump(s):
            s.model.buffers["bn.running_var"][0] *= 1.5
        self._expect_detected(bump, "buffers_equal")

    def test_detects_dropout_rng_advance(self):
        self._expect_detected(lambda s: s.rngs["dropout"].random(), "rng_equal")

    def test_detects_data_rng_advance(self):
        self._expect_detected(lambda s: s.rngs["data"].random(), "rng_equal")

    def test_detects_step_change(self):
        def step(s):
            s.step += 1
        self._expect_detected(step, "step_equal")

    def test_hyperparameter_change_is_not_state_only(self):
        br, rec = branch(self.ck, {"kind": "beta_switch", "beta2": 0.9})
        chk = verify_state_only(restore(self.ck), br, self.probe)
        self.assertFalse(rec["state_only"])
        self.assertFalse(chk["optimizer_hparams_equal"])
        self.assertFalse(chk["state_only"])
        self.assertTrue(chk["outside_optimizer_unchanged"])
        self.assertTrue(chk["weights_equal"] and chk["eval_forward_equal"])


if __name__ == "__main__":
    unittest.main()

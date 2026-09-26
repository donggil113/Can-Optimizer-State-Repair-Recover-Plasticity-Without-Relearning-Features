"""End-to-end runner tests on a shrunken copy of the real smoke config."""

import copy
import gzip
import json
import math
import os
import shutil
import tempfile
import unittest

from osrepair.runner import grid_edges, run_experiment, validate_config

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.path.join(os.path.dirname(HERE), "configs", "p4_smoke.json")


def tiny_config():
    with open(SMOKE) as f:
        cfg = json.load(f)
    cfg["name"] = "test_tiny"
    cfg["model"].update({"in_dim": 5, "hidden": 8, "out_dim": 3})
    cfg["stream"].update({"in_dim": 5, "n_classes": 3, "n_tasks": 3, "steps_per_task": 20, "batch_size": 6,
                          "eval_size": 24})
    cfg["branch"]["task_index"] = 2
    cfg["post_branch_steps"] = 20
    cfg["eval_every"] = 5
    cfg["probe_size"] = 8
    cfg["tuning"]["seeds"] = [1000]
    for fam in cfg["tuning"]["families"].values():
        fam["grid"] = fam["grid"][:2]
    cfg["test_seeds"] = [3000, 3001]
    for arm in cfg["arms"]:
        if arm.get("schedule"):
            arm["schedule"]["period"] = 7
    return cfg


class TestRunnerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cfg = tiny_config()
        cls.out = os.path.join(cls.tmp, "run")
        cls.manifest = run_experiment(cls.cfg, cls.out, argv=["test"])
        with open(os.path.join(cls.out, "summary.json")) as f:
            cls.summary = json.load(f)
        with gzip.open(os.path.join(cls.out, "log.jsonl.gz"), "rt") as f:
            cls.log = [json.loads(line) for line in f]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_completed_and_all_arms_pass(self):
        self.assertEqual(self.manifest["status"], "COMPLETED", self.manifest.get("traceback"))
        for name, per_seed in self.manifest["arm_status"].items():
            self.assertEqual(set(per_seed), {3000, 3001}, name)
            for seed, st in per_seed.items():
                self.assertEqual(st["status"], "PASS", (name, seed, st))

    def test_outputs_exist_and_manifest_fields(self):
        for f in ("config.json", "manifest.json", "summary.json", "summary.md", "tuning.jsonl", "log.jsonl.gz"):
            self.assertTrue(os.path.exists(os.path.join(self.out, f)), f)
        for k in ("config_sha256", "code_sha256", "git", "env", "wall_seconds", "cpu_seconds", "peak_rss_kb"):
            self.assertIn(k, self.manifest)
        self.assertTrue(self.manifest["global_random_untouched"])

    def test_integrity_checks_pass(self):
        for seed, sr in self.summary["seeds"].items():
            self.assertEqual(sr["checks"]["status"], "PASS", sr["checks"])
            self.assertTrue(sr["checks"]["adam_base_replay"]["bitwise_equal_to_keep_all"])
            for name, c in sr["checks"].items():
                if name != "status":
                    self.assertTrue(c["same_batches"] and c["same_dropout_masks"], name)

    def test_state_only_checks_recorded(self):
        for sr in self.summary["seeds"].values():
            for name, arm in sr["arms"].items():
                if arm["type"] != "branch":
                    continue
                chk = arm["state_only_check"]
                self.assertTrue(chk["weights_equal"] and chk["eval_forward_equal"] and chk["train_forward_equal"])
                self.assertTrue(chk["outside_optimizer_unchanged"], name)
                self.assertEqual(chk["state_only"], arm["intervention_record"]["state_only"], name)

    def test_norm_matching_reproduces_reference_norms(self):
        steps = {}
        for r in self.log:
            if r["kind"] == "step" and r["seed"] == 3000:
                steps.setdefault(r["arm"], {})[r["step"]] = r
        for arm, ref in (("keepdir_resetbothmag", "reset_both"), ("resetbothdir_keepmag", "keep_all")):
            for s, rec in steps[arm].items():
                for k, n in rec["update_norms"].items():
                    self.assertTrue(math.isclose(n, steps[ref][s]["update_norms"][k], rel_tol=1e-9, abs_tol=1e-15))

    def test_lr_bump_scales_first_update_exactly(self):
        b = self.cfg["branch"]["task_index"] * self.cfg["stream"]["steps_per_task"]
        recs = {r["arm"]: r for r in self.log if r["kind"] == "step" and r["seed"] == 3000 and r["step"] == b}
        self.assertTrue(math.isclose(recs["lr_bump_x3_20"]["update_norm"], 3.0 * recs["keep_all"]["update_norm"],
                                     rel_tol=1e-12))

    def test_periodic_arm_fires_without_oracle(self):
        e = self.cfg["branch"]["task_index"] * self.cfg["stream"]["steps_per_task"] + self.cfg["post_branch_steps"]
        for sr in self.summary["seeds"].values():
            arm = sr["arms"]["periodic_reset_both_P128"]
            self.assertEqual(arm["scheduled_interventions"], (e - 1) // 7)
        acc = {a["name"]: a["access"]["boundary_oracle"] for a in self.cfg["arms"]}
        self.assertFalse(acc["periodic_reset_both_P128"])
        self.assertTrue(all(acc[a["name"]] for a in self.cfg["arms"] if a["type"] == "branch"))

    def test_tuning_used_dev_seeds_only(self):
        with open(os.path.join(self.out, "tuning.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        self.assertTrue(rows)
        self.assertEqual({r["seed"] for r in rows}, {1000})

    def test_paired_deltas_zero_for_keep_all_and_replay(self):
        for name in ("keep_all", "adam_base_replay"):
            for k, v in self.manifest["paired_deltas"][name].items():
                self.assertEqual(v["per_seed"], [0.0] * len(v["per_seed"]), (name, k))


class TestGridEdges(unittest.TestCase):
    def test_flags_min_and_max_only(self):
        grid = [{"lr": 0.001, "beta2": 0.999}, {"lr": 0.003, "beta2": 0.99}, {"lr": 0.01, "beta2": 0.9}]
        self.assertEqual(grid_edges(grid, {"kind": "adam", "lr": 0.01, "beta2": 0.99}), {"lr": "max"})
        self.assertEqual(grid_edges(grid, {"lr": 0.003, "beta2": 0.999}), {"beta2": "max"})
        self.assertEqual(grid_edges(grid, {"lr": 0.001, "beta2": 0.99}), {"lr": "min"})
        self.assertEqual(grid_edges([{"lr": 0.1}], {"lr": 0.1}), {})

    def test_recorded_in_manifest(self):
        cfg = tiny_config()
        cfg["arms"] = [a for a in cfg["arms"] if a["name"] == "keep_all"]
        cfg["test_seeds"] = [3000]
        tmp = tempfile.mkdtemp()
        try:
            m = run_experiment(cfg, os.path.join(tmp, "r"), argv=["test"])
            for fam, r in m["tuning"]["results"].items():
                self.assertEqual(r["best_at_grid_edge"], grid_edges(cfg["tuning"]["families"][fam]["grid"], r["best"]))
            with open(os.path.join(tmp, "r", "summary.md")) as f:
                text = f.read()
            if any(r["best_at_grid_edge"] for r in m["tuning"]["results"].values()):
                self.assertIn("WARNING", text)
        finally:
            shutil.rmtree(tmp)


class TestValidationAndFailures(unittest.TestCase):
    def test_rejects_tuning_on_test_seeds(self):
        cfg = tiny_config()
        cfg["tuning"]["split"] = "test"
        cfg["tuning"]["seeds"] = [3000]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_rejects_test_seeds_outside_test_split(self):
        cfg = tiny_config()
        cfg["test_seeds"] = [1000]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_rejects_forward_reference(self):
        cfg = tiny_config()
        arms = cfg["arms"]
        i = next(i for i, a in enumerate(arms) if a["name"] == "keepdir_resetbothmag")
        arms.insert(0, arms.pop(i))
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_rejects_misaligned_eval(self):
        cfg = tiny_config()
        cfg["eval_every"] = 7
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_requires_keep_all(self):
        cfg = tiny_config()
        cfg["arms"] = [a for a in cfg["arms"] if a["name"] != "keep_all"]
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_resource_cap_is_recorded_as_failure_not_pass(self):
        cfg = tiny_config()
        cfg["resource_caps"]["max_wall_seconds"] = 0
        tmp = tempfile.mkdtemp()
        try:
            m = run_experiment(cfg, os.path.join(tmp, "r"), argv=["test"])
            self.assertEqual(m["status"], "FAIL")
            self.assertIn("TimeoutError", m["error"])
            for per_seed in m["arm_status"].values():
                self.assertFalse(any(s.get("status") == "PASS" for s in per_seed.values()))
        finally:
            shutil.rmtree(tmp)

    def test_arm_failure_is_preserved(self):
        cfg = tiny_config()
        cfg["arms"].append({"name": "bad_arm", "type": "branch", "timing": "oracle_boundary",
                            "intervention": {"kind": "mix", "rho_m": 2.0, "rho_v": 0.5},
                            "access": {"boundary_oracle": True}})
        cfg["test_seeds"] = [3000]
        cfg["arms"] = [a for a in cfg["arms"] if a["type"] == "branch" and not a.get("norm_match")]
        tmp = tempfile.mkdtemp()
        try:
            m = run_experiment(cfg, os.path.join(tmp, "r"), argv=["test"])
            self.assertEqual(m["arm_status"]["bad_arm"][3000]["status"], "FAIL")
            self.assertIn("ValueError", m["arm_status"]["bad_arm"][3000]["error"])
            self.assertEqual(m["arm_status"]["keep_all"][3000]["status"], "PASS")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()

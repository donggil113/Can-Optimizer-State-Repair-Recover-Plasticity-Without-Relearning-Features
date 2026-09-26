"""Regression tests for the stage-1 runner (probe protocol, caps, decision rules)."""

import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from osrepair import stage1
from osrepair.rng import RNG
from osrepair.stream import TaskStream
from osrepair.trainer import init_state, train_step

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "p4_stage1.json")


def tiny(max_cpu=600):
    with open(CFG) as f:
        cfg = json.load(f)
    cfg["model"]["hidden"] = 8
    cfg["stream_common"].update({"in_dim": 5, "n_train_tasks": 3, "steps_per_task": 12, "batch_size": 6,
                                 "eval_size": 24})
    cfg["conditions"][1]["n_classes"] = 6
    cfg["probe"].update({"steps": 12, "eval_every": 4, "early_window": 8, "probe_size": 8})
    cfg["seeds"] = {"dev": [1000], "test": [3000, 3001, 3002]}
    for fam in cfg["tuning"]["families"].values():
        fam["lr_grid"] = fam["lr_grid"][:2]
    cfg["cost_model"].update({"trunk_cpu_s": 0.3, "probe_cpu_s": 0.05})
    cfg["resource_caps"].update({"max_cpu_seconds": max_cpu, "reserved_cpu_seconds": 0})
    return cfg


def run_sub(cfg, out):
    path = out + ".json"
    with open(path, "w") as f:
        json.dump(cfg, f)
    p = subprocess.run([sys.executable, "-m", "osrepair.stage1", "--config", path, "--out", out], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    with open(os.path.join(out, "manifest.json")) as f:
        return p, json.load(f)


class TestTaskOverride(unittest.TestCase):
    def test_train_step_task_override(self):
        s = TaskStream(3000, 5, 3, 3, 10, 6, eval_size=8)
        a = init_state({"in_dim": 5, "hidden": 6, "out_dim": 3}, {"kind": "adam", "lr": 1e-3}, 3000)
        b = init_state({"in_dim": 5, "hidden": 6, "out_dim": 3}, {"kind": "adam", "lr": 1e-3}, 3000)
        self.assertEqual(train_step(a, s)["task"], 0)
        self.assertEqual(train_step(b, s, task=2)["task"], 2)

    def test_repeated_permutation_rejected(self):
        cfg = tiny()
        cfg["stream_common"]["n_train_tasks"] = 60
        cond = {"name": "x", "family": "label_permutation", "n_classes": 3}   # only 6 permutations
        with self.assertRaises(RuntimeError):
            stage1.make_stream(cfg, cond, 3000)


class TestStage1EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.proc, cls.m = run_sub(tiny(), os.path.join(cls.tmp, "r"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_completed_with_manifest_fields(self):
        self.assertEqual(self.m["status"], "COMPLETED", self.m.get("traceback"))
        for k in ("config_sha256", "code_sha256", "git", "env", "limits_applied", "cpu_seconds_used", "peak_rss_kb"):
            self.assertIn(k, self.m)
        self.assertTrue(self.m["global_random_untouched"])
        self.assertEqual(self.m["limits_applied"]["threads"], 1)

    def test_phenomenon_cells_paired(self):
        ph = self.m["results"]["phenomenon"]
        for cond, fams in ph.items():
            for fam, r in fams.items():
                for seed, sr in r["seeds"].items():
                    self.assertTrue(sr["paired_batches_and_masks"], (cond, fam, seed))
                    self.assertEqual(set(sr["probes"]), {"fresh", "early", "late"})
                    self.assertIsNone(sr["probes"]["fresh"]["old_task_forgetting"])

    def test_every_cell_has_a_status(self):
        for c in self.m["cells"]:
            self.assertIn(c["status"], ("PASS", "FAIL", "NOT_RUN"))
        self.assertIn(self.m["results"]["overall"], ("PHENOMENON_NOT_ESTABLISHED", "INCOMPLETE",
                                                     "STATE_REPAIR_BRANCH_ON_HOLD", "MECHANISM_PILOT_CANDIDATE"))

    def test_intervention_stage_consistent_with_phenomenon(self):
        for cond, v in self.m["results"]["verdicts"].items():
            ran = cond in self.m["results"]["interventions"]
            self.assertEqual(ran, v["phenomenon"] == "ESTABLISHED", cond)


class TestForcedInterventionStage(unittest.TestCase):
    """Drive the phenomenon and intervention stages directly (no OS limits, no tuning)
    so the intervention code path is always exercised, whatever the phenomenon verdict."""

    def test_interventions_run_and_pair(self):
        cfg = tiny()
        cond = cfg["conditions"][0]
        spec = {"kind": "adam", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "lr": 3e-3}
        results = {"tuning": {cond["name"]: {"adam": {"best": spec, "status": "RESOLVED"}}},
                   "phenomenon": {}, "interventions": {}}
        cells, late_cks = [], {}
        tmp = tempfile.mkdtemp()
        try:
            log = stage1.Log(os.path.join(tmp, "log.jsonl"))
            budget = stage1.Budget(1e9)
            s_ = cfg["stream_common"]
            stage1._phenomenon(cfg, cond, "adam", results, cells, log, budget, 0.0, late_cks,
                               s_["steps_per_task"], s_["n_train_tasks"] * s_["steps_per_task"])
            stage1._interventions(cfg, cond, results, cells, log, budget, late_cks)
            log.close()
            self.assertTrue(all(c["status"] == "PASS" for c in cells), cells)
            iv = results["interventions"][cond["name"]]["seeds"]
            self.assertEqual(len(iv), 3)
            for sr in iv.values():
                self.assertTrue(sr["checks"]["paired_batches_and_masks"])
                self.assertTrue(sr["checks"]["keep_all_equals_phenomenon_late_probe"])
                self.assertEqual(len(sr["probes"]), len(cfg["interventions"]["arms"]))
                self.assertFalse(sr["checks"]["keepdir_reset_bothmag"]["optimizer_state_changed"])
                self.assertTrue(sr["checks"]["reset_t"]["optimizer_state_changed"])
            v = stage1._intervention_verdict(cfg, cond, results)
            self.assertIn(v["intervention_stage"], ("MECHANISM_PILOT_CANDIDATE", "STATE_REPAIR_BRANCH_ON_HOLD"))
            with open(os.path.join(tmp, "log.jsonl")) as f:
                recs = [json.loads(line) for line in f]
            steps = {}
            for r in recs:
                if r["kind"] == "probe_step" and r.get("phase") == "intervention" and r["seed"] == 3000:
                    steps.setdefault(r["arm"], []).append(r["update_norms"])
            for x in cfg["interventions"]["primary"]:
                self.assertEqual(len(steps[f"keepdir_{x}mag"]), cfg["probe"]["steps"])
                for a, b in zip(steps[f"keepdir_{x}mag"], steps[x]):
                    for k in a:
                        self.assertAlmostEqual(a[k], b[k], delta=1e-12 + 1e-9 * abs(b[k]))
        finally:
            shutil.rmtree(tmp)


class TestBudget(unittest.TestCase):
    def test_exhausted_budget_gives_not_run_and_incomplete(self):
        cfg = tiny(max_cpu=1)
        cfg["cost_model"]["trunk_cpu_s"] = 5.0
        tmp = tempfile.mkdtemp()
        try:
            proc, m = run_sub(cfg, os.path.join(tmp, "r"))
            self.assertEqual(m["status"], "COMPLETED", m.get("traceback"))
            self.assertTrue(all(c["status"] == "NOT_RUN" for c in m["cells"]))
            self.assertEqual(m["results"]["overall"], "INCOMPLETE")
        finally:
            shutil.rmtree(tmp)


class TestDecisionRules(unittest.TestCase):
    def cfg(self):
        return tiny()

    def phen(self, late_minus_early, tstatus="RESOLVED"):
        seeds = {}
        for i, d in enumerate(late_minus_early):
            p = {k: {"probe_auc_acc": 0.5, "probe_early_acc": 0.5, "probe_final_acc": 0.5} for k in ("fresh", "early")}
            p["late"] = {"probe_auc_acc": 0.5 + d, "probe_early_acc": 0.5, "probe_final_acc": 0.5}
            seeds[3000 + i] = {"probes": p}
        cfg = self.cfg()
        cond = cfg["conditions"][0]
        res = {"phenomenon": {cond["name"]: {"adam": {"tuning_status": tstatus, "seeds": seeds}}}}
        return stage1._phenomenon_verdict(cfg, cond, res)["phenomenon"]

    def test_phenomenon_rule(self):
        self.assertEqual(self.phen([-0.05, -0.03, -0.04]), "ESTABLISHED")
        self.assertEqual(self.phen([-0.05, -0.03, +0.001]), "NOT_ESTABLISHED")   # not all seeds
        self.assertEqual(self.phen([-0.01, -0.01, -0.01]), "NOT_ESTABLISHED")    # below MIE
        self.assertEqual(self.phen([-0.05, -0.05]), "INCOMPLETE")                # missing cell
        self.assertEqual(self.phen([-0.05, -0.05, -0.05], "TUNING_UNRESOLVED"), "INCOMPLETE")

    def iv(self, x_minus_keep, x_minus_ctrl):
        cfg = self.cfg()
        cond = cfg["conditions"][0]
        seeds = {}
        for i, (a, b) in enumerate(zip(x_minus_keep, x_minus_ctrl)):
            base = {"probe_auc_acc": 0.5}
            probes = {n: dict(base) for n in [a_["name"] for a_ in cfg["interventions"]["arms"]]}
            for x in cfg["interventions"]["primary"]:
                probes[x] = {"probe_auc_acc": 0.5 + a}
                probes[f"keepdir_{x}mag"] = {"probe_auc_acc": 0.5 + a - b}
            seeds[3000 + i] = {"probes": probes}
        res = {"interventions": {cond["name"]: {"seeds": seeds}}}
        return stage1._intervention_verdict(cfg, cond, res)

    def test_intervention_rule(self):
        v = self.iv([0.05, 0.04, 0.06], [0.03, 0.03, 0.04])
        self.assertEqual(v["per_intervention"]["reset_v"]["verdict"], "DIFFERENCE_REMAINS")
        self.assertEqual(v["intervention_stage"], "MECHANISM_PILOT_CANDIDATE")
        v = self.iv([0.05, 0.04, 0.06], [0.001, -0.002, 0.003])
        self.assertEqual(v["per_intervention"]["reset_v"]["verdict"], "EXPLAINED_BY_UPDATE_NORM")
        self.assertEqual(v["intervention_stage"], "STATE_REPAIR_BRANCH_ON_HOLD")
        v = self.iv([0.05, -0.01, 0.06], [0.05, 0.05, 0.05])
        self.assertEqual(v["per_intervention"]["reset_v"]["verdict"], "NO_POSITIVE_EFFECT_OVER_KEEP_ALL")


if __name__ == "__main__":
    unittest.main()

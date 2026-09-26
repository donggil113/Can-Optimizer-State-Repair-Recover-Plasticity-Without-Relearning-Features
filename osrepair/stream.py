"""Synthetic supervised task stream with disjoint seed splits.

Inputs are x ~ N(0, I_D). Each task defines labels through a teacher:

    random_teacher      task k has its own random linear teacher W_k;
                        y = argmax_c (W_k x)_c. New decision boundaries per task.
    label_permutation   one teacher W_0 for all tasks; task k applies a fixed
                        label permutation pi_k. Features transfer, the readout
                        must be relearned.

Teachers, permutations and held-out evaluation sets are pure functions of the
stream seed. Training batches are drawn from a stateful data RNG owned by the
training state, so a checkpoint captures the data position exactly and paired
branches receive identical batches.

The stream knows its task boundaries; the learner does not. Boundary steps are
exposed only via :meth:`TaskStream.oracle_boundaries`, and every use of it must
be declared as oracle access (see ``schedules`` in runner.py).
"""

from __future__ import annotations

from .rng import RNG, bit_hash, derive_seed

# Disjoint stream-seed ranges. Tuning uses DEV only, calibration is reserved
# for thresholds of non-oracle triggers, TEST is only used for reported runs.
SPLITS = {
    "dev": range(1000, 1100),
    "calibration": range(2000, 2100),
    "test": range(3000, 3100),
}


def split_of(seed: int) -> str:
    for name, r in SPLITS.items():
        if seed in r:
            return name
    raise ValueError(f"seed {seed} is not in any declared split")


def check_seeds(seeds: list[int], split: str) -> None:
    bad = [s for s in seeds if s not in SPLITS[split]]
    if bad:
        raise ValueError(f"seeds {bad} are not in split {split!r}")


class TaskStream:
    def __init__(self, seed: int, in_dim: int, n_classes: int, n_tasks: int, steps_per_task: int,
                 batch_size: int, family: str = "random_teacher", eval_size: int = 256):
        if family not in ("random_teacher", "label_permutation"):
            raise ValueError(f"unknown task family {family!r}")
        self.seed = seed
        self.in_dim, self.n_classes = in_dim, n_classes
        self.n_tasks, self.steps_per_task = n_tasks, steps_per_task
        self.batch_size, self.family, self.eval_size = batch_size, family, eval_size
        self._teachers = [self._make_teacher(k) for k in range(n_tasks)]
        self._perms = [self._make_perm(k) for k in range(n_tasks)]
        self._eval_cache: dict[tuple[str, int], tuple] = {}

    @property
    def total_steps(self) -> int:
        return self.n_tasks * self.steps_per_task

    def _make_teacher(self, k: int) -> list[list[float]]:
        tk = 0 if self.family == "label_permutation" else k
        r = RNG(derive_seed("teacher", self.seed, tk))
        return [[r.gauss() for _ in range(self.in_dim)] for _ in range(self.n_classes)]

    def _make_perm(self, k: int) -> list[int]:
        perm = list(range(self.n_classes))
        if self.family == "label_permutation" and k > 0:
            r = RNG(derive_seed("perm", self.seed, k))
            r.shuffle(perm)
        return perm

    def task_at(self, step: int) -> int:
        """Task index for global training step ``step`` (environment side)."""
        return min(step // self.steps_per_task, self.n_tasks - 1)

    def oracle_boundaries(self) -> list[int]:
        """Global steps at which a new task starts. ORACLE information."""
        return [k * self.steps_per_task for k in range(1, self.n_tasks)]

    def label(self, task: int, x: list[float]) -> int:
        W = self._teachers[task]
        scores = [sum(w * xi for w, xi in zip(row, x)) for row in W]
        c = max(range(self.n_classes), key=scores.__getitem__)
        return self._perms[task][c]

    def sample_batch(self, task: int, data_rng: RNG) -> tuple[list[list[float]], list[int]]:
        x = [[data_rng.gauss() for _ in range(self.in_dim)] for _ in range(self.batch_size)]
        return x, [self.label(task, xb) for xb in x]

    def eval_set(self, task: int, split: str = "heldout") -> tuple[list[list[float]], list[int]]:
        """Fixed held-out set for ``task``, independent of the training RNG."""
        key = (split, task)
        if key not in self._eval_cache:
            r = RNG(derive_seed("eval", self.seed, split, task))
            x = [[r.gauss() for _ in range(self.in_dim)] for _ in range(self.eval_size)]
            self._eval_cache[key] = (x, [self.label(task, xb) for xb in x])
        return self._eval_cache[key]

    def probe_set(self, n: int) -> list[list[float]]:
        """Fixed unlabeled inputs for representation diagnostics."""
        r = RNG(derive_seed("probe", self.seed))
        return [[r.gauss() for _ in range(self.in_dim)] for _ in range(n)]

    def fingerprint(self) -> str:
        """Hash of everything that defines the stream (teachers, perms, eval sets)."""
        return bit_hash({
            "teachers": self._teachers,
            "perms": self._perms,
            "eval": [self.eval_set(k) for k in range(self.n_tasks)],
        })

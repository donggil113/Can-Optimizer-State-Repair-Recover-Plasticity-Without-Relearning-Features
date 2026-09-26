"""Stream and split tests."""

import unittest

from osrepair.rng import RNG
from osrepair.stream import SPLITS, TaskStream, check_seeds, split_of


def make(seed=3000, family="random_teacher"):
    return TaskStream(seed, in_dim=5, n_classes=4, n_tasks=3, steps_per_task=10, batch_size=8,
                      family=family, eval_size=64)


class TestSplits(unittest.TestCase):
    def test_disjoint(self):
        names = list(SPLITS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(set(SPLITS[a]) & set(SPLITS[b]), (a, b))

    def test_check_seeds(self):
        check_seeds([1000, 1001], "dev")
        self.assertEqual(split_of(3005), "test")
        with self.assertRaises(ValueError):
            check_seeds([3000], "dev")
        with self.assertRaises(ValueError):
            split_of(5)


class TestStream(unittest.TestCase):
    def test_deterministic_and_seed_dependent(self):
        self.assertEqual(make().fingerprint(), make().fingerprint())
        self.assertNotEqual(make(3000).fingerprint(), make(3001).fingerprint())

    def test_eval_sets_independent_of_training_rng(self):
        s = make()
        e1 = s.eval_set(1)
        s.sample_batch(1, RNG(0))
        self.assertEqual(make().eval_set(1), e1)

    def test_task_schedule_and_oracle(self):
        s = make()
        self.assertEqual([s.task_at(t) for t in (0, 9, 10, 29, 35)], [0, 0, 1, 2, 2])
        self.assertEqual(s.oracle_boundaries(), [10, 20])

    def test_label_permutation_relabels_same_partition(self):
        s = make(family="label_permutation")
        x, _ = s.eval_set(0)
        y0 = [s.label(0, xb) for xb in x]
        for k in (1, 2):
            yk = [s.label(k, xb) for xb in x]
            mapping = {}
            for a, b in zip(y0, yk):
                self.assertEqual(mapping.setdefault(a, b), b)   # a function of the task-0 label
            self.assertEqual(len(set(mapping.values())), len(mapping))  # injective

    def test_random_teacher_tasks_differ(self):
        s = make()
        x, _ = s.eval_set(0)
        agree = sum(s.label(0, xb) == s.label(1, xb) for xb in x) / len(x)
        self.assertLess(agree, 0.8)

    def test_classes_are_all_used(self):
        s = make()
        _, y = s.eval_set(0)
        self.assertEqual(set(y), set(range(4)))


if __name__ == "__main__":
    unittest.main()

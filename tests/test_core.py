import unittest
import numpy as np
from lens_luotain_target import (Setup, transport, catalog, twins,
                                posterior_update, choose_probe, trial, benchmark)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.s = Setup()
        self.a = transport(self.s)

    def test_photometry_and_occlusion(self):
        self.assertEqual(self.a.shape, (self.s.wall_pixels, self.s.patches))
        self.assertTrue(np.allclose(self.a @ np.ones(self.s.patches), 1.0))
        all_dark = transport(self.s, np.ones(self.s.mask_cells, dtype=bool))
        self.assertTrue(np.all(all_dark <= self.a + 1e-12))
        self.assertTrue(np.any(all_dark < self.a))

    def test_candidates_are_bounded_and_indistinguishable_initially(self):
        x = twins(self.a, 4100)
        self.assertEqual(x.shape, (8, self.s.patches))
        self.assertTrue(np.all((x >= 0.099999) & (x <= 0.900001)))
        # Initial weak observation should be far closer than the same masked observations.
        start = x @ self.a.T
        masks = catalog(self.s)
        differences = [np.linalg.norm((x[0]-x[1]) @ transport(self.s,m,shift).T)
                       for m,shift in masks]
        self.assertGreater(max(differences), np.linalg.norm(start[0]-start[1]) + 1e-4)

    def test_candidate_families_change_across_seeds(self):
        a, b = twins(self.a, 6100), twins(self.a, 6101)
        self.assertFalse(np.allclose(a, b))
        self.assertLess(np.linalg.norm((a @ self.a.T) - (b @ self.a.T)), 1e-9)

    def test_selection_uses_only_predicted_disagreement(self):
        pred = np.array([[[0.], [0.]], [[0.], [4.]], [[1.], [2.]]])
        p = np.array([0.5, 0.5])
        self.assertEqual(choose_probe(p, pred, np.array([0,1,2])), 1)
        self.assertEqual(choose_probe(p, pred, np.array([0,2])), 2)

    def test_bayes_update_prefers_supported_candidate(self):
        p = posterior_update(np.array([0.5,0.5]), np.array([[0.], [1.]]), np.array([1.]), 0.1)
        self.assertGreater(p[1], 0.999)
        self.assertAlmostEqual(p.sum(), 1.0)

    def test_reproducible_shared_truth(self):
        a, b, c = [trial(4101, budget=2, method=m) for m in ('active','random','repeat')]
        self.assertEqual(a['truth'], b['truth'])
        self.assertEqual(b['truth'], c['truth'])
        self.assertEqual(a, trial(4101, budget=2, method='active'))
        self.assertAlmostEqual(a['initial_truth_probability'], b['initial_truth_probability'])

    def test_benchmark_has_honest_comparisons(self):
        result = benchmark(range(4100, 4105), budget=2)
        self.assertEqual(len(result['runs']['active']), 5)
        for method in ('active','random','repeat'):
            self.assertTrue(0 <= result['summary'][method]['accuracy'] <= 1)

if __name__ == '__main__':
    unittest.main()

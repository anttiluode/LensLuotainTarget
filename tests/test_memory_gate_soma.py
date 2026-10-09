import importlib
import importlib.util
import unittest

import numpy as np


class MemoryGateSomaTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('memory_gate_soma'),
                             'The memory-gate-soma reference model is missing')
        self.m = importlib.import_module('memory_gate_soma')
        self.s = self.m.MemorySetup()

    def test_history_is_retained_by_diverse_branch_filters(self):
        np.testing.assert_allclose(self.m.trace_history(np.array([1., 0., 0.]),
                                   np.array([.5, .9])), [.125, .081], atol=1e-14)

    def test_different_histories_share_current_input_and_initial_soma(self):
        histories, states = self.m.candidate_histories(8100, self.s)
        self.assertEqual(histories.shape, (8, 32))
        self.assertEqual(states.shape, (8, 12))
        np.testing.assert_array_equal(histories[:, -1], 0)
        self.assertLess(np.max(np.abs(histories)), 1.00000001)
        self.assertLess(np.max(np.abs(states @ self.m.uniform_gate(self.s))), 1e-12)
        self.assertGreater(np.linalg.norm(states[0]-states[1]), .01)
        _, other = self.m.candidate_histories(8101, self.s)
        self.assertFalse(np.allclose(states, other))

    def test_gates_have_equal_gain_and_pre_mix_access(self):
        gates = self.m.gate_catalog(self.s)
        np.testing.assert_allclose(np.linalg.norm(gates, axis=1), 1., atol=1e-14)
        _, states = self.m.candidate_histories(8100, self.s)
        self.assertGreater(np.max(np.std(states @ gates.T, axis=0)), .01)

    def test_retained_observation_context_changes_selected_gate(self):
        predictions = np.array([[-1., 1., 1.], [1., 1., -1.], [.2, .2, .2]])
        self.assertEqual(self.m.choose_gate(np.array([.5,.5,0.]), predictions, [0,1,2]), 0)
        self.assertEqual(self.m.choose_gate(np.array([0.,.5,.5]), predictions, [0,1,2]), 1)

    def test_read_disturbance_is_optional_and_does_not_mutate_input(self):
        state = np.array([1., -1.]); gate = np.array([1., 0.])
        value, same = self.m.read_state(state, gate, 0)
        self.assertEqual(value, 1.)
        np.testing.assert_array_equal(same, state)
        value, changed = self.m.read_state(state, gate, .25)
        np.testing.assert_allclose(changed, [.75,-1.])
        np.testing.assert_array_equal(state, [1.,-1.])

    def test_first_gate_cannot_depend_on_the_unobserved_true_history(self):
        a = self.m.trial(8100, 'active', self.s, truth=0)
        b = self.m.trial(8100, 'active', self.s, truth=7)
        self.assertEqual(a['actions'][0], b['actions'][0])

    def test_all_policies_receive_the_same_truth_and_measurement_budget(self):
        runs = [self.m.trial(8100, method, self.s) for method in self.m.METHODS]
        self.assertEqual(len({r['truth'] for r in runs}), 1)
        self.assertTrue(all(len(r['observations']) == 4 for r in runs))
        self.assertTrue(all(len(r['actions']) == 3 for r in runs))
        self.assertEqual(runs[0], self.m.trial(8100, 'active', self.s))

    def test_post_mix_and_erased_reads_cannot_identify_the_original_history(self):
        for method in ['post_mix', 'erased']:
            r = self.m.trial(8100, method, self.s)
            np.testing.assert_allclose(r['posterior'], np.full(8,.125), atol=1e-12)
            self.assertAlmostEqual(r['logloss'], np.log(8), places=10)
            self.assertGreater(r['forecast_mse'], 0)

    def test_zero_damage_preserves_retained_state(self):
        r = self.m.trial(8100, 'active', self.m.MemorySetup(read_damage=0))
        self.assertEqual(r['disturbance'], 0.)

    def test_invalid_sensor_and_read_parameters_are_rejected(self):
        for kw in [{'noise':0}, {'read_damage':-1}, {'read_damage':1.1}]:
            with self.assertRaises(ValueError): self.m.MemorySetup(**kw)
        with self.assertRaises(ValueError): self.m.trial(8100,'active',self.s,budget=25)


if __name__ == '__main__':
    unittest.main()

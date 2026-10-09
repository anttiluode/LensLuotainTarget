"""Changing binary reads to scalars or giving controls different heads breaks these."""
import importlib
import inspect
import unittest
import numpy as np


class SpikeReceiverTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('spike_receiver'),
                             'binary receiver module is missing')
        self.r = importlib.import_module('spike_receiver')
        self.d = importlib.import_module('spike_data')

    def test_new_splits_are_reproducible_disjoint_and_initially_ambiguous(self):
        b = self.d.make_batch('train', 123, 6, draws=3)
        again = self.d.make_batch('train', 123, 6, draws=3)
        np.testing.assert_array_equal(b['states'], again['states'])
        np.testing.assert_allclose(b['states'].sum(axis=1), 0, atol=1e-14)
        self.assertEqual(b['noise'].shape, (18, 9))
        self.assertFalse(np.array_equal(b['states'], self.d.make_batch('test', 123, 6, draws=3)['states']))
        from learned_data import make_batch
        self.assertFalse(np.array_equal(b['states'][::3], make_batch('train', 123, 6)['states']))
        self.assertFalse(np.array_equal(b['noise'][0], b['noise'][1]))

    def test_hard_bits_and_eight_unit_positive_queries(self):
        b = self.d.make_batch('demo', 6, 3, draws=1)
        out = self.r.rollout(self.r.init_model('adaptive', 9), b)
        self.assertEqual(out['observations'].shape, (3, 9))
        self.assertTrue(set(out['observations'].ravel()).issubset({-1., 1.}))
        np.testing.assert_allclose(np.linalg.norm(out['gates'], axis=-1), 1)
        self.assertTrue(np.all(out['gates'] > 0))
        self.assertTrue(np.all(np.abs(out['thresholds']) <= .15))
        for invalid in (-1, 8, .5):
            with self.assertRaises(ValueError):
                self.r.select_command(self.r.init_model('adaptive', 1), np.zeros((1,16)), invalid)

    def test_physical_adaptation_is_separated_from_internal_context(self):
        h = np.stack([np.zeros(16), np.ones(16)])
        commands = {}
        for method in ('adaptive','fixed','gate_only','threshold_only','zero_threshold'):
            model = self.r.init_model(method, 9)
            commands[method] = self.r.select_command(model, h, 0)
            self.assertFalse(np.allclose(commands[method]['context_gate'][0], commands[method]['context_gate'][1]))
        for method in ('fixed', 'threshold_only'):
            np.testing.assert_array_equal(commands[method]['g'][0], commands[method]['g'][1])
        for method in ('fixed', 'gate_only'):
            self.assertEqual(commands[method]['theta'][0], commands[method]['theta'][1])
        self.assertNotEqual(commands['threshold_only']['theta'][0], commands['threshold_only']['theta'][1])
        np.testing.assert_array_equal(commands['zero_threshold']['theta'], 0)

    def test_all_controls_have_matched_computation_and_parameters(self):
        for method in self.r.METHODS:
            model = self.r.init_model(method, 2)
            resources = self.r.resources(model)
            self.assertEqual(resources['parameters'], 1252)
            self.assertEqual(resources['neural_macs'], 7376)
            self.assertEqual(resources['new_readings'], 8)
            self.assertEqual(resources['command_values_per_read'], 13)

    def test_receiver_interfaces_cannot_consume_hidden_state_or_future_query(self):
        allowed = {
            'initial_receiver': ('model','initial_answer'),
            'select_command': ('model','h','step','random_gate','random_threshold'),
            'receive': ('model','h','command','answer'),
            'predict_state': ('model','h')}
        for name, names in allowed.items():
            self.assertEqual(tuple(inspect.signature(getattr(self.r,name)).parameters), names)

    def test_threshold_changes_bit_but_not_physical_write(self):
        b = self.d.make_batch('demo', 15, 2, draws=1)
        low = self.r.init_model('fixed', 1)
        high = self.r.init_model('fixed', 1)
        low['params']['threshold_logits'][:] = -20
        high['params']['threshold_logits'][:] = 20
        a, c = self.r.rollout(low,b), self.r.rollout(high,b)
        self.assertFalse(np.array_equal(a['observations'], c['observations']))
        np.testing.assert_array_equal(a['final_state'], c['final_state'])

    def test_common_sum_control_contains_only_sensor_noise(self):
        b = self.d.make_batch('demo', 82, 5, draws=1)
        out = self.r.rollout(self.r.init_model('adaptive', 1), b, mode='post_mix')
        np.testing.assert_allclose(out['final_state'], b['states'], atol=1e-14)
        np.testing.assert_array_equal(out['observations'][:,1:],
            np.where(b['noise'][:,1:] > out['thresholds'], 1., -1.))


if __name__ == '__main__':
    unittest.main()

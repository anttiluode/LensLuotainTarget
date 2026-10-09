import importlib
import importlib.util
import unittest

import numpy as np


class LearnedReceiverTests(unittest.TestCase):
    def setUp(self):
        for name in ('learned_data', 'learned_receiver'):
            self.assertIsNotNone(importlib.util.find_spec(name), f'{name} implementation is missing')
        self.data = importlib.import_module('learned_data')
        self.core = importlib.import_module('learned_receiver')

    def test_fresh_histories_share_current_input_and_initial_sum(self):
        batch = self.data.make_batch('test', 99123, 64)
        self.assertEqual(np.unique(batch['histories'], axis=0).shape[0], 64)
        np.testing.assert_array_equal(batch['histories'][:, -1], 0)
        self.assertLess(np.max(np.abs(batch['histories'])), 1.000000001)
        np.testing.assert_allclose(batch['states'].sum(axis=1), 0, atol=1e-13)
        self.assertGreater(np.var(batch['states']), 1e-5)

    def test_split_streams_are_independent_and_reproducible(self):
        a = self.data.make_batch('train', 9, 12)
        again = self.data.make_batch('train', 9, 12)
        other = self.data.make_batch('test', 9, 12)
        changed_queries = self.data.make_batch('train', 9, 12, query_kind='novel')
        changed_history = self.data.make_batch('train', 9, 12, history_kind='switching')
        np.testing.assert_array_equal(a['histories'], again['histories'])
        self.assertFalse(np.array_equal(a['histories'], other['histories']))
        for field in ('histories', 'noise', 'random_gates'):
            np.testing.assert_array_equal(a[field], changed_queries[field])
        np.testing.assert_array_equal(a['noise'], changed_history['noise'])
        self.assertTrue(np.any(changed_queries['queries'] < 0))

    def test_gate_uses_receiver_context_and_has_unit_gain(self):
        model = self.core.init_model('adaptive', 42)
        h = np.zeros((2, 16))
        h[1, 3] = 1
        gate = self.core.select_gate(model, h, 1)
        np.testing.assert_allclose(np.linalg.norm(gate, axis=1), 1, atol=1e-13)
        self.assertTrue(np.all(gate >= 0))
        self.assertFalse(np.allclose(gate[0], gate[1]))
        np.testing.assert_array_equal(gate, self.core.select_gate(model, h, 1))

    def test_fixed_gate_does_not_use_receiver_answers(self):
        model = self.core.init_model('fixed', 42)
        gate = self.core.select_gate(model, np.zeros((2, 16)), 1)
        other = self.core.select_gate(model, np.ones((2, 16)), 1)
        np.testing.assert_array_equal(gate, other)

    def test_future_questions_cannot_change_acquired_readings(self):
        model = self.core.init_model('adaptive', 42)
        a = self.data.make_batch('demo', 81000, 8)
        b = self.data.make_batch('demo', 81000, 8, query_kind='novel')
        ra, rb = self.core.rollout(model, a), self.core.rollout(model, b)
        np.testing.assert_array_equal(ra['gates'], rb['gates'])
        np.testing.assert_array_equal(ra['observations'], rb['observations'])
        np.testing.assert_array_equal(ra['predicted_state'], rb['predicted_state'])
        self.assertFalse(np.allclose(ra['targets'], rb['targets']))

    def test_all_methods_have_equal_read_budget_and_disturbance(self):
        batch = self.data.make_batch('demo', 81000, 8)
        for method in ('adaptive', 'fixed', 'random', 'recurrent'):
            result = self.core.rollout(self.core.init_model(method, 42), batch)
            self.assertEqual(result['observations'].shape, (8, 4))
            self.assertEqual(result['gates'].shape, (8, 3, 12))
            self.assertTrue(np.all(result['relative_disturbance'] > 0))
            np.testing.assert_allclose(np.linalg.norm(result['gates'], axis=2), 1, atol=1e-13)

    def test_information_loss_controls_still_target_original_memory(self):
        batch = self.data.make_batch('demo', 81000, 8, query_kind='novel')
        model = self.core.init_model('adaptive', 42)
        expected = np.einsum('bkd,bd->bk', batch['queries'], batch['states'])
        for mode in ('erased', 'post_mix'):
            result = self.core.rollout(model, batch, mode=mode)
            np.testing.assert_allclose(result['targets'], expected)
            np.testing.assert_allclose(result['observations'], batch['noise'], atol=1e-13)
        erased = self.core.rollout(model, batch, mode='erased')
        np.testing.assert_array_equal(erased['final_state'], 0)
        np.testing.assert_allclose(erased['relative_disturbance'], 1)

    def test_matched_recurrent_resource_account_is_comparable(self):
        adaptive = self.core.resources(self.core.init_model('adaptive', 42))
        recurrent = self.core.resources(self.core.init_model('recurrent', 42))
        self.assertEqual(adaptive['parameters'], 944)
        self.assertEqual(recurrent['parameters'], 941)
        self.assertLess(abs(recurrent['neural_macs']/adaptive['neural_macs']-1), .06)
        self.assertLess(abs(recurrent['persistent_values']/adaptive['persistent_values']-1), .12)

    def test_initial_gate_schedule_is_shared_across_trained_policies(self):
        adaptive = self.core.init_model('adaptive', 42)
        for method in ('fixed', 'recurrent'):
            other = self.core.init_model(method, 42)
            np.testing.assert_array_equal(adaptive['params']['gate_logits'], other['params']['gate_logits'])

    def test_invalid_gate_index_and_unknown_method_are_rejected(self):
        with self.assertRaises(ValueError):
            self.core.init_model('truth_lookup', 42)
        model = self.core.init_model('adaptive', 42)
        with self.assertRaises(ValueError):
            self.core.select_gate(model, np.zeros((1, 16)), 3)


if __name__ == '__main__':
    unittest.main()

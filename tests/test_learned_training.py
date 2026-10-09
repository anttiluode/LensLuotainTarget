import importlib
import importlib.util
import unittest

import numpy as np

from learned_data import make_batch
from learned_receiver import init_model, rollout


class LearnedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('learned_training'), 'learning implementation is missing')
        self.train = importlib.import_module('learned_training')

    def check_gradient(self, model, batch, names):
        loss, gradients = self.train.loss_and_grad(model, batch)
        self.assertAlmostEqual(loss, float(np.mean(rollout(model, batch)['objective'])), places=12)
        for name in names:
            weight = model['params'][name]
            for flat_index in (0, weight.size//2, weight.size-1):
                index = np.unravel_index(flat_index, weight.shape)
                original = weight[index]
                weight[index] = original+1e-6
                plus = np.mean(rollout(model, batch)['objective'])
                weight[index] = original-1e-6
                minus = np.mean(rollout(model, batch)['objective'])
                weight[index] = original
                self.assertAlmostEqual(gradients[name][index], (plus-minus)/2e-6,
                                       delta=2e-7, msg=f'{name}{index}')

    def test_recurrent_and_decoder_gradients_match_finite_differences(self):
        self.check_gradient(init_model('random', 7), make_batch('train', 9, 5),
                            ['initial_w', 'initial_b', 'recurrent', 'gate_input',
                             'answer_input', 'bias', 'decoder', 'decoder_b'])

    def test_policy_gradient_includes_read_and_write_paths(self):
        self.check_gradient(init_model('adaptive', 7), make_batch('train', 9, 5),
                            ['policy', 'gate_logits', 'recurrent', 'answer_input'])

    def test_write_disturbance_gradient_survives_zero_prediction_head(self):
        model = init_model('fixed', 7)
        model['params']['decoder'][:] = 0
        self.check_gradient(model, make_batch('train', 9, 5), ['gate_logits'])
        _, gradient = self.train.loss_and_grad(model, make_batch('train', 9, 5))
        self.assertGreater(np.linalg.norm(gradient['gate_logits']), 1e-6)

    def test_training_reduces_validation_loss_and_selects_its_best_checkpoint(self):
        result = self.train.train('fixed', 7, steps=100)
        trace = result['trace']
        self.assertLess(trace[-1]['validation_objective'], trace[0]['validation_objective'])
        expected = min(trace, key=lambda row: row['validation_objective'])
        self.assertEqual(result['selected_step'], expected['step'])
        validation = make_batch('validation', 10000, 512)
        observed = np.mean(rollout(result['model'], validation)['objective'])
        self.assertAlmostEqual(observed, expected['validation_objective'], places=12)
        self.assertEqual(result['updates'], 100)

    def test_paired_interval_preserves_episode_pairing(self):
        before = np.arange(1., 513.)
        after = before-1
        interval = self.train.paired_interval(before, after)
        self.assertAlmostEqual(interval['low'], 1)
        self.assertAlmostEqual(interval['high'], 1)
        swapped = self.train.paired_interval(after, before)
        self.assertAlmostEqual(swapped['low'], -1)


if __name__ == '__main__':
    unittest.main()

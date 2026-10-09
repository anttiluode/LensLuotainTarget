"""Unbiased expected gradients, not straight-through gradients of hard bits."""
import importlib
import itertools
import unittest
import numpy as np
from scipy.special import ndtr
from spike_data import make_batch
from spike_receiver import init_model, rollout, NOISE_SD


class SpikeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('spike_training'), 'binary training is missing')
        self.t = importlib.import_module('spike_training')

    def exact_tree(self, model, batch, reads=2, gradient=False):
        value = 0.
        total = {k: np.zeros_like(v) for k,v in model['params'].items()}
        for sequence in itertools.product((-1.,1.), repeat=reads+1):
            bits = np.asarray([sequence])
            out = rollout(model,batch,cache=True,reads=reads,forced_bits=bits)
            probability = .5
            for i,row in enumerate(out['_cache']['steps']):
                probability *= float(ndtr(sequence[i+1]*(row['value'][0]-row['theta'][0])/NOISE_SD))
            value += probability*float(out['objective'][0])
            if gradient:
                _, grad = self.t.loss_and_grad(model,batch,reads=reads,forced_bits=bits)
                for k in total:
                    total[k] += probability*grad[k]
        return (value,total) if gradient else value

    def test_exact_bit_tree_gradient_includes_policy_and_read_disturbance(self):
        batch = make_batch('train', 145, 1, draws=1)
        for method in ('adaptive','fixed','gate_only','threshold_only','zero_threshold','random'):
            model = init_model(method, 57)
            _, grad = self.exact_tree(model,batch,gradient=True)
            for key, weight in model['params'].items():
                for flat in (0, weight.size//2, weight.size-1):
                    index = np.unravel_index(flat,weight.shape)
                    original = weight[index]
                    weight[index] = original+1e-5
                    plus = self.exact_tree(model,batch)
                    weight[index] = original-1e-5
                    minus = self.exact_tree(model,batch)
                    weight[index] = original
                    self.assertAlmostEqual(grad[key][index],(plus-minus)/2e-5,delta=2e-6,
                                           msg=f'{method} {key} {index}')

    def test_graded_reference_gradient_is_pathwise(self):
        batch = make_batch('train', 77, 3, draws=1)
        for method in ('graded_adaptive','graded_fixed'):
            model = init_model(method, 29)
            _, grad = self.t.loss_and_grad(model,batch,reads=3)
            for key, weight in model['params'].items():
                index = np.unravel_index(weight.size//2,weight.shape)
                original = weight[index]
                weight[index] = original+1e-5
                plus = np.mean(rollout(model,batch,reads=3)['objective'])
                weight[index] = original-1e-5
                minus = np.mean(rollout(model,batch,reads=3)['objective'])
                weight[index] = original
                self.assertAlmostEqual(grad[key][index],(plus-minus)/2e-5,delta=2e-6,msg=key)

    def test_leave_one_out_baseline_does_not_include_own_loss(self):
        losses = np.array([1.,2.,4.,8.,10.,20.,40.,80.])
        b = self.t.independent_baseline(losses,4)
        self.assertEqual(b[0],14/3)
        self.assertEqual(b[4],140/3)
        losses[0] = 10000
        self.assertEqual(self.t.independent_baseline(losses,4)[0],b[0])

    def test_tail_score_is_finite_and_asymptotically_correct(self):
        scores = self.t.probit_score(np.array([-40.,0.,40.]), np.ones(3))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertAlmostEqual(scores[0],40.02497,places=4)
        self.assertAlmostEqual(scores[1],np.sqrt(2/np.pi),places=10)
        self.assertEqual(scores[2],0.)

    def test_checkpoint_architecture_and_weights_are_verified(self):
        model = init_model('adaptive',9)
        value = self.t.model_to_json(model)
        self.assertEqual(self.t.model_hash(model),self.t.model_hash(self.t.model_from_json(value)))
        value['params']['threshold_policy'][0] = float('nan')
        with self.assertRaises(ValueError):
            self.t.model_from_json(value)

    def test_zero_update_training_selects_only_validation(self):
        receipt = self.t.train('adaptive',20261012,steps=0,batch_size=2)
        self.assertEqual(receipt['selected_step'],0)
        self.assertEqual(len(receipt['trace']),1)
        self.assertEqual(receipt['updates'],0)
        self.assertEqual(receipt['validation_seed'],11000)


if __name__ == '__main__':
    unittest.main()

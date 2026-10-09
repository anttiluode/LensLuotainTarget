"""Removing a frozen success requirement or accepting partial seals fails these."""
import importlib
import unittest
import numpy as np


class SpikeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('spike_experiment'),'sealed binary evaluation is missing')
        self.e = importlib.import_module('spike_experiment')

    def test_success_requires_effect_interval_seed_wins_and_objective(self):
        baseline = np.full((3,32),1.)
        adaptive = np.full((3,32),.8)
        self.assertTrue(self.e.comparison(baseline,adaptive,baseline,adaptive)['passed'])
        self.assertFalse(self.e.comparison(baseline,adaptive,baseline,baseline*1.1)['passed'])
        self.assertFalse(self.e.comparison(baseline,baseline*.99,baseline,adaptive)['passed'])
        one_winner = np.stack([np.full(32,.1),np.full(32,1.01),np.full(32,1.01)])
        self.assertFalse(self.e.comparison(baseline,one_winner,baseline,adaptive)['passed'])

    def test_bootstrap_uses_histories_not_independent_noise_draws(self):
        b = np.array([[1.,2.,3.,4.],[2.,4.,6.,8.]])
        a = b*.8
        c = self.e.comparison(b,a,b,a)
        self.assertEqual(c['independent_histories'],4)
        self.assertEqual(c['interval']['resamples'],2048)
        self.assertAlmostEqual(c['relative_gain'],.2)

    def test_heldout_evaluation_requires_all_predeclared_training_runs(self):
        with self.assertRaises(ValueError):
            self.e.validate_seal({'runs':[],'sources':self.e.source_fingerprints()})

    def test_resumed_runs_must_match_the_training_source(self):
        from spike_receiver import init_model, METHODS
        from spike_training import TRAIN_SEEDS, model_to_json, model_hash
        sources = self.e.source_fingerprints()
        runs = []
        for method in METHODS:
            for seed in TRAIN_SEEDS:
                model = init_model(method,seed)
                runs.append({'method':method,'seed':seed,'model':model_to_json(model),
                             'updates':4000,'batch_size':96,'noise_draws':4,'validation_seed':11000,
                             'trace':[{'step':i,'validation_objective':1.} for i in range(0,4001,100)],
                             'selected_step':0,'selected_value':1.,'weight_sha256':model_hash(model),
                             'training_sources':sources})
        runs[0]['training_sources'] = {'wrong':'source'}
        seal = {'runs':runs,'sources':sources}
        seal['seal_sha256'] = self.e.payload_hash(seal)
        with self.assertRaisesRegex(ValueError,'training source'):
            self.e.validate_seal(seal)


if __name__ == '__main__':
    unittest.main()

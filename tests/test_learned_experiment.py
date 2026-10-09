import importlib
import importlib.util
import unittest

from learned_training import train


class LearnedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('learned_experiment'), 'evaluation implementation is missing')
        self.experiment = importlib.import_module('learned_experiment')

    def receipt(self, strong):
        conditions = []
        for name in ('fresh_familiar', 'novel_queries', 'unseen_history'):
            comparisons = {}
            for method in ('random', 'fixed', 'recurrent'):
                gain = .2 if method == 'random' or strong else -.01
                comparisons[method] = {'mse': {'relative_gain': gain, 'low': gain/10},
                                       'positive_seed_wins': 3 if gain > 0 else 0,
                                       'objective_gain': gain/10}
            conditions.append({'name': name, 'comparisons': comparisons,
                               'summary': {'erased': {'nrmse': 1.01}, 'post_mix': {'nrmse': 1.02}}})
        return {'conditions': conditions}

    def test_random_win_cannot_pass_stronger_architecture_gates(self):
        gates = self.experiment.frozen_gates(self.receipt(False), contract_passed=True)
        self.assertTrue(gates['L2_random'])
        self.assertFalse(gates['L3_fixed_and_recurrent'])
        self.assertFalse(gates['L4_unseen_history_transfer'])
        self.assertFalse(all(v is True for v in gates.values()))

    def test_browser_contract_stays_pending_until_verified(self):
        gates = self.experiment.frozen_gates(self.receipt(True))
        self.assertIsNone(gates['L1_contract_checks'])
        self.assertFalse(all(v is True for v in gates.values()))
        verified = self.experiment.frozen_gates(self.receipt(True), contract_passed=True)
        self.assertTrue(all(v is True for v in verified.values()))

    def test_short_training_cannot_be_presented_as_frozen_run(self):
        record = train('fixed', 7, steps=1)
        with self.assertRaises(ValueError):
            self.experiment.validate_training_record(record)


if __name__ == '__main__':
    unittest.main()

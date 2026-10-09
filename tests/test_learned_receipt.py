import copy
import importlib
import importlib.util
import unittest


class LearnedReceiptTests(unittest.TestCase):
    def test_sealing_cannot_hide_changed_scientific_results(self):
        self.assertIsNotNone(importlib.util.find_spec('verify_learned_receipt'))
        compare = importlib.import_module('verify_learned_receipt').compare_receipts
        expected = {'conditions': [{'summary': {'adaptive': {'mse': .00024}}}],
                    'gates': {'L1_contract_checks': None, 'L3_fixed_and_recurrent': True},
                    'adaptive_advantage_earned': False, 'contract_note': 'pending'}
        sealed = copy.deepcopy(expected)
        sealed.update(adaptive_advantage_earned=True, contract_note='passed',
                      verification={'tests': 41})
        sealed['gates']['L1_contract_checks'] = True
        compare(expected, sealed)
        for path in ('mse', 'scientific_gate', 'extra_result'):
            damaged = copy.deepcopy(sealed)
            if path == 'mse':
                damaged['conditions'][0]['summary']['adaptive']['mse'] *= 1.1
            elif path == 'scientific_gate':
                damaged['gates']['L3_fixed_and_recurrent'] = False
            else:
                damaged['conditions'][0]['summary']['adaptive']['unreported'] = 1
            with self.subTest(path=path), self.assertRaises(ValueError):
                compare(expected, damaged)


if __name__ == '__main__':
    unittest.main()

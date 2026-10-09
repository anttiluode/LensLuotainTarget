import subprocess
import unittest

import make_learned_site_data as exporter


class LearnedExportTests(unittest.TestCase):
    def test_contract_seal_requires_executed_passing_tests(self):
        clean = subprocess.CompletedProcess([], 0, stderr='Ran 42 tests in 0.8s\n\nOK\n')
        self.assertEqual(exporter.checked_test_count(clean), 42)
        for result in (
                subprocess.CompletedProcess([], 0, stderr='Ran 42 tests in 0.8s\n\nOK (skipped=5)\n'),
                subprocess.CompletedProcess([], 0, stderr='Ran 0 tests in 0.0s\n\nOK\n'),
                subprocess.CompletedProcess([], 1, stderr='Ran 42 tests\nFAILED (failures=1)\n')):
            with self.subTest(stderr=result.stderr), self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                exporter.checked_test_count(result)


if __name__ == '__main__':
    unittest.main()

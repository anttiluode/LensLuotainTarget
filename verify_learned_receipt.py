"""Reproduce the held-out receipt from sealed weights, without retraining."""
import json
import math
import tempfile
from pathlib import Path

from learned_experiment import evaluate, frozen_gates


SEAL_FIELDS = {('gates', 'L1_contract_checks'), ('adaptive_advantage_earned',),
               ('contract_note',), ('verification',)}


def compare_receipts(expected, published, path=()):
    """Ignore only contract-sealing metadata; check all scientific outcomes."""
    if path in SEAL_FIELDS:
        return
    if type(expected) is not type(published):
        raise ValueError(f'receipt type changed at {path}')
    if isinstance(expected, dict):
        wanted = {k for k in expected if path+(k,) not in SEAL_FIELDS}
        actual = {k for k in published if path+(k,) not in SEAL_FIELDS}
        if wanted != actual:
            raise ValueError(f'receipt fields changed at {path}')
        for key in wanted:
            compare_receipts(expected[key], published[key], path+(key,))
    elif isinstance(expected, list):
        if len(expected) != len(published):
            raise ValueError(f'receipt length changed at {path}')
        for i, (a, b) in enumerate(zip(expected, published)):
            compare_receipts(a, b, path+(i,))
    elif isinstance(expected, float):
        if not math.isclose(expected, published, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'receipt value changed at {path}: {expected} vs {published}')
    elif expected != published:
        raise ValueError(f'receipt value changed at {path}: {expected} vs {published}')


def main():
    root = Path(__file__).resolve().parent
    published = json.loads((root/'results/learned_queries.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        reproduced = evaluate(root/'results/learned_weights.json', Path(temporary)/'receipt.json')
    compare_receipts(reproduced, published)
    passed = published['gates']['L1_contract_checks']
    if passed is not True or published['gates'] != frozen_gates(published, contract_passed=True):
        raise ValueError('published gate verdict is inconsistent')
    if published['adaptive_advantage_earned'] != all(v is True for v in published['gates'].values()):
        raise ValueError('published combined verdict is inconsistent')
    print('All held-out metrics, weight hashes and scientific gates reproduce from the sealed weights.')


if __name__ == '__main__':
    main()

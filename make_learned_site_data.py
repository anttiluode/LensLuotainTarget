"""Export real selected weights, independent worlds and the exact frozen receipt."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from learned_data import make_batch, FAMILY_NAMES
from learned_experiment import load_models


def export():
    root = Path(__file__).resolve().parent
    checkpoints, _ = load_models(root/'results/learned_weights.json')
    models = [r['model'] for r in checkpoints['models'] if r['model']['seed'] == 20261009]
    worlds = []
    for history_kind, seed in (('mixture', 81000), ('switching', 81001)):
        batch = make_batch('demo', seed, 5, query_kind='novel', history_kind=history_kind)
        for i in range(5):
            worlds.append({'name': f'{FAMILY_NAMES[batch["families"][i]]} example {i+1}',
                           'transfer': history_kind == 'switching',
                           **{key: batch[key][i].tolist() for key in
                              ('histories','states','queries','noise','random_gates')}})
    receipt = json.loads((root/'results/learned_queries.json').read_text())
    payload = {'models': models, 'worlds': worlds, 'receipt': receipt}
    source = "'use strict';\nconst LearnedQueryData="+json.dumps(payload,separators=(',',':'))+";\n"
    source += "if(typeof module!=='undefined'&&module.exports)module.exports=LearnedQueryData;else globalThis.LearnedQueryData=LearnedQueryData;\n"
    (root/'site/learned-data.js').write_text(source)
    page = root/'site/learned.html'
    if page.exists():
        html = page.read_text()
        for name in ('learned-core.js', 'learned-data.js', 'learned-ui.js', 'learned.css'):
            version = hashlib.sha256((root/'site'/name).read_bytes()).hexdigest()[:12]
            pattern = r'((?:src|href)="'+re.escape(name)+r')(?:\?v=[^"]+)?(")'
            html = re.sub(pattern, lambda m: m[1]+'?v='+version+m[2], html)
        page.write_text(html)


def checked_test_count(check):
    """A zero exit code with skipped parity tests is not a verified contract."""
    check.check_returncode()
    count = re.search(r'Ran (\d+) tests', check.stderr)
    if not count or int(count[1]) == 0 or 'skipped=' in check.stderr:
        raise RuntimeError('Contract sealing requires all tests to run, including Node browser parity.')
    return int(count[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true', help='Run contracts and seal L1 after successful browser parity')
    args = p.parse_args()
    export()
    if args.verify:
        root = Path(__file__).resolve().parent
        check = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                               cwd=root, capture_output=True, text=True)
        print(check.stderr, end='')
        count = checked_test_count(check)
        receipt_path = root/'results/learned_queries.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['gates']['L1_contract_checks'] = True
        receipt['adaptive_advantage_earned'] = all(v is True for v in receipt['gates'].values())
        receipt['contract_note'] = 'Reference, gradient, boundary and actual-weight Python/JavaScript parity tests passed.'
        receipt['verification'] = {'command': 'python -m unittest discover -s tests -v',
                                   'tests': count}
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        export()


if __name__ == '__main__':
    main()

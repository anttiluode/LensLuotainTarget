"""Export all sealed weights; verify binary inference before marking S1 passed."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from learned_data import FAMILY_NAMES
from make_learned_site_data import checked_test_count
from spike_data import make_batch
from spike_experiment import validate_seal

ROOT = Path(__file__).resolve().parent


def verification_sources():
    names = ['site/spike-core.js','site/spike-ui.js','make_spike_site_data.py']
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT/'tests').glob('test_*.py'))]
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def receipt_hash():
    return hashlib.sha256((ROOT/'results/learned_spikes.json').read_bytes()).hexdigest()


def verification_valid(value):
    return (value.get('S1_passed') is True and value.get('receipt_sha256') == receipt_hash()
            and value.get('sources') == verification_sources())


def build_payload():
    seal = json.loads((ROOT/'results/spike_weights.json').read_text())
    validate_seal(seal)
    receipt = json.loads((ROOT/'results/learned_spikes.json').read_text())
    if receipt['weights_seal_sha256'] != seal['seal_sha256'] or receipt['sources'] != seal['sources']:
        raise ValueError('receipt and checkpoint seal differ')
    worlds = []
    for kind,seed in (('mixture',82000),('switching',82001)):
        batch = make_batch('demo',seed,6,draws=1,query_kind='novel',history_kind=kind)
        for i in range(6):
            worlds.append({'name':f'{FAMILY_NAMES[batch["families"][i]]} · example {i+1}',
                           'seed':seed,'transfer':kind == 'switching',
                           **{key:batch[key][i].tolist() for key in
                              ('histories','states','queries','noise','random_gates','random_thresholds')}})
    path = ROOT/'results/spike_verification.json'
    proof = json.loads(path.read_text()) if path.exists() else {}
    verified = verification_valid(proof)
    return {'models':[r['model'] for r in seal['runs']], 'worlds':worlds,'receipt':receipt,
            'verification':{'S1_passed':verified,
                            'binary_interface_earned':verified and receipt['statistical_binary_interface_passed'],
                            'joint_gate_threshold_earned':verified and receipt['statistical_joint_gate_threshold_passed']}}


def javascript_payload(payload):
    return ("'use strict';\nconst SpikeData="+json.dumps(payload,separators=(',',':'))+";\n"
            "if(typeof module!=='undefined'&&module.exports)module.exports=SpikeData;else globalThis.SpikeData=SpikeData;\n")


def export():
    (ROOT/'site/spike-data.js').write_text(javascript_payload(build_payload()))
    page = ROOT/'site/spikes.html'
    html = page.read_text()
    for name in ('spike-core.js','spike-data.js','spike-ui.js','spike.css'):
        version = hashlib.sha256((ROOT/'site'/name).read_bytes()).hexdigest()[:12]
        pattern = r'((?:src|href)="'+re.escape(name)+r')(?:\?v=[^"]+)?(")'
        html = re.sub(pattern,lambda m:m[1]+'?v='+version+m[2],html)
    page.write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    export()
    if args.verify:
        check = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                               cwd=ROOT,capture_output=True,text=True)
        print(check.stderr,end='')
        count = checked_test_count(check)
        seal = json.loads((ROOT/'results/spike_weights.json').read_text())
        proof = {'S1_passed':True,'receipt_sha256':receipt_hash(),'weights_seal_sha256':seal['seal_sha256'],
                 'sources':verification_sources(),'command':'python -m unittest discover -s tests -v',
                 'tests':count,'note':'Exact expected gradients, observation-only boundary, matched resources, hard bits, budget and actual trained Python/JavaScript parity passed without skips.'}
        (ROOT/'results/spike_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
        export()
    print('All 24 trained receivers exported with independent demo worlds.')


if __name__ == '__main__':
    main()

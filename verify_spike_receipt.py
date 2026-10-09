"""Reproduce the frozen evaluation and reject a stale trained-browser export."""
import json
import subprocess
import sys
import numpy as np
from make_spike_site_data import ROOT, build_payload, javascript_payload, verification_valid


def verify_export(stored,expected):
    """Weights and receipt stay exact; regenerated worlds allow machine roundoff."""
    if stored.keys() != expected.keys():
        raise ValueError('browser payload keys changed')
    for key in ('models','receipt','verification'):
        if stored[key] != expected[key]:
            raise ValueError('browser '+key+' no longer matches the sealed publication')
    if len(stored['worlds']) != len(expected['worlds']):
        raise ValueError('demo world count changed')
    numeric = {'histories','states','queries','noise','random_gates','random_thresholds'}
    for actual,wanted in zip(stored['worlds'],expected['worlds']):
        if actual.keys() != wanted.keys():
            raise ValueError('demo world fields changed')
        for key in wanted:
            if key in numeric:
                a,b = np.asarray(actual[key]),np.asarray(wanted[key])
                if a.shape != b.shape or not np.allclose(a,b,rtol=1e-10,atol=1e-12):
                    raise ValueError('demo numerical values changed: '+key)
            elif actual[key] != wanted[key]:
                raise ValueError('demo metadata changed: '+key)
    return True


def main():
    subprocess.run([sys.executable,str(ROOT/'spike_experiment.py'),'verify'],cwd=ROOT,check=True)
    proof = json.loads((ROOT/'results/spike_verification.json').read_text())
    if not verification_valid(proof):
        raise ValueError('binary inference verification is stale or incomplete')
    payload = build_payload()
    source = (ROOT/'site/spike-data.js').read_text()
    prefix,suffix = javascript_payload({}).split('{}',1)
    if not source.startswith(prefix) or not source.endswith(suffix):
        raise ValueError('browser export wrapper changed')
    stored = json.loads(source[len(prefix):-len(suffix)])
    verify_export(stored,payload)
    print('Binary interface earned:',payload['verification']['binary_interface_earned'])
    print('Joint adaptive gate-and-threshold benefit earned:',payload['verification']['joint_gate_threshold_earned'])


if __name__ == '__main__':
    main()

"""Reproduce the frozen evaluation and reject a stale trained-browser export."""
import json
import subprocess
import sys
from make_spike_site_data import ROOT, build_payload, javascript_payload, verification_valid


def main():
    subprocess.run([sys.executable,str(ROOT/'spike_experiment.py'),'verify'],cwd=ROOT,check=True)
    proof = json.loads((ROOT/'results/spike_verification.json').read_text())
    if not verification_valid(proof):
        raise ValueError('binary inference verification is stale or incomplete')
    payload = build_payload()
    if (ROOT/'site/spike-data.js').read_text() != javascript_payload(payload):
        raise ValueError('browser data no longer matches the sealed weights and receipt')
    print('Binary interface earned:',payload['verification']['binary_interface_earned'])
    print('Joint adaptive gate-and-threshold benefit earned:',payload['verification']['joint_gate_threshold_earned'])


if __name__ == '__main__':
    main()

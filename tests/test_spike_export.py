"""Sealing must reject skipped parity and stale inference/receipt proofs."""
import importlib
import json
from pathlib import Path
import subprocess
import unittest
import numpy as np
from spike_receiver import rollout
from spike_training import model_from_json

ROOT=Path(__file__).resolve().parents[1]


class SpikeExportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('make_spike_site_data'),'trained weight exporter is missing')
        self.e=importlib.import_module('make_spike_site_data')

    def test_export_runs_all_sealed_initializations_on_independent_examples(self):
        payload=self.e.build_payload()
        self.assertEqual(len(payload['models']),24)
        self.assertEqual({m['seed'] for m in payload['models']},{20261012,20261013,20261014})
        self.assertEqual({w['seed'] for w in payload['worlds']},{82000,82001})
        self.assertEqual(len(payload['worlds']),12)
        self.assertEqual(payload['receipt']['gates']['S4']['passed'],False)

    def test_actual_trained_browser_predictions_match_python(self):
        payload=self.e.build_payload()
        code="""
const c=require('./site/spike-core.js');let s='';process.stdin.on('data',v=>s+=v);
process.stdin.on('end',()=>{const d=JSON.parse(s),out=[];for(const m of d.models)for(const w of [d.worlds[0],d.worlds[6]]){
const r=c.startEpisode(m,w);for(let i=0;i<8;i++)c.stepEpisode(r);out.push({predicted_state:c.predictState(m,r.h),observations:r.observations,gates:r.gates,thresholds:r.thresholds});
}process.stdout.write(JSON.stringify(out));});"""
        js=json.loads(subprocess.run(['node','-e',code],input=json.dumps(payload),text=True,cwd=ROOT,capture_output=True,check=True).stdout)
        for i,model in enumerate(payload['models']):
            for j,world in enumerate((payload['worlds'][0],payload['worlds'][6])):
                batch={k:np.asarray([world[k]]) for k in ('states','queries','noise','random_gates','random_thresholds')}
                expected=rollout(model_from_json(model),batch)
                for key in js[2*i+j]:
                    np.testing.assert_allclose(js[2*i+j][key],expected[key][0],atol=1e-11,rtol=1e-9,err_msg=f'{model["method"]} {key}')

    def test_stale_proof_cannot_mark_integrity_passed(self):
        value={'S1_passed':True,'receipt_sha256':'stale','sources':{}}
        self.assertFalse(self.e.verification_valid(value))

    def test_skipped_or_empty_test_runs_cannot_seal(self):
        from make_learned_site_data import checked_test_count
        for output in ('Ran 0 tests\nOK','Ran 10 tests\nOK (skipped=1)'):
            with self.assertRaises(RuntimeError):
                checked_test_count(subprocess.CompletedProcess([],0,'',output))


if __name__=='__main__':
    unittest.main()

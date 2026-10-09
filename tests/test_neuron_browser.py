import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from memory_gate_soma import MemorySetup, trial


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser/reference parity')
class NeuronBrowserTests(unittest.TestCase):
    def test_browser_read_sequences_and_damage_match_reference(self):
        self.assertTrue((ROOT/'site/neuron-core.js').is_file(), 'Browser core is missing')
        code = """
const core=require('./site/neuron-core.js');
const data=require('./site/neuron-data.js');
const results=[];
for(const eta of [0,.25]) for(const method of ['active','random','fixed','open_loop','post_mix','erased']){
 const run=core.begin(data.scenes[0],{...data.config,read_damage:eta,method});
 for(let k=0;k<3;k++)core.step(run);
 results.push({eta,method,actions:run.actions,observations:run.observations,
               posterior:run.posterior,disturbance:core.disturbance(run)});
}
console.log(JSON.stringify(results));
"""
        out = subprocess.check_output(['node','-e',code], cwd=ROOT, text=True)
        for r in json.loads(out):
            reference = trial(8100,r['method'],MemorySetup(read_damage=r['eta']))
            self.assertEqual(r['actions'],reference['actions'])
            np.testing.assert_allclose(r['observations'],reference['observations'],atol=1e-12)
            np.testing.assert_allclose(r['posterior'],reference['posterior'],atol=1e-11)
            self.assertAlmostEqual(r['disturbance'],reference['disturbance'],places=12)

    def test_site_receipt_is_the_exact_frozen_summary(self):
        self.assertTrue((ROOT/'site/neuron-data.js').is_file(), 'Generated browser data is missing')
        code="console.log(JSON.stringify(require('./site/neuron-data.js').receipt))"
        observed=json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True))
        expected=json.loads((ROOT/'results/memory_gate_soma.json').read_text())
        self.assertEqual(observed,expected)

    def test_manual_read_does_not_break_fixed_or_random_strategy(self):
        code = """
const assert=require('node:assert/strict');
const core=require('./site/neuron-core.js');
const data=require('./site/neuron-data.js');
const fixed=[0,4,8,1,5,9,2,6,10,3,7,11,...Array.from({length:12},(_,i)=>i+12)];
for(const scene of data.scenes) for(const method of ['fixed','random']) for(let manual=0;manual<12;manual++){
 const run=core.begin(scene,{...data.config,method});
 core.step(run,manual);
 core.step(run);
 core.step(run);
 const order=method==='fixed'?fixed:scene.random_order;
 assert.deepEqual(run.actions,[manual,...order.filter(i=>i!==manual).slice(0,2)]);
 assert.equal(new Set(run.actions).size,3);
}
"""
        subprocess.check_call(['node','-e',code],cwd=ROOT)


if __name__=='__main__':
    unittest.main()

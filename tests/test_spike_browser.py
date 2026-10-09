"""The browser must run the real command/bit paths, including all controls."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import numpy as np
from spike_data import make_batch
from spike_receiver import METHODS, init_model, rollout
from spike_training import model_to_json

ROOT = Path(__file__).resolve().parents[1]


class SpikeBrowserTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'site/spike-core.js').exists(),'browser binary inference is missing')
        self.assertIsNotNone(shutil.which('node'),'Node is required for browser parity')

    def test_all_methods_and_memory_controls_match_python(self):
        batch = make_batch('demo',954,1,draws=1,query_kind='novel')
        episode = {k:batch[k][0].tolist() for k in ('states','queries','noise','random_gates','random_thresholds')}
        models = [init_model(m,32) for m in METHODS]
        request = {'models':[model_to_json(m) for m in models],'world':episode}
        code = """
const c=require('./site/spike-core.js');
let s='';process.stdin.on('data',x=>s+=x);process.stdin.on('end',()=>{
 const v=JSON.parse(s),out=[];
 for(const m of v.models)for(const mode of ['normal','erased','post_mix']){
  const r=c.startEpisode(m,v.world,mode);for(let i=0;i<8;i++)c.stepEpisode(r);
  out.push({gates:r.gates,thresholds:r.thresholds,observations:r.observations,
            predicted_state:c.predictState(m,r.h),final_state:r.memory});
 }process.stdout.write(JSON.stringify(out));
});"""
        run = subprocess.run(['node','-e',code],input=json.dumps(request),text=True,cwd=ROOT,capture_output=True,check=True)
        js = json.loads(run.stdout)
        for i,model in enumerate(models):
            for j,mode in enumerate(('normal','erased','post_mix')):
                expected = rollout(model,batch,mode=mode)
                for key in ('gates','thresholds','observations','predicted_state','final_state'):
                    np.testing.assert_allclose(js[3*i+j][key],expected[key][0],rtol=1e-10,atol=1e-12,
                                               err_msg=f'{model["method"]} {mode} {key}')

    def test_browser_refuses_ninth_read_and_questions_do_not_change_receiver(self):
        batch = make_batch('demo',961,1,draws=1)
        request = {'model':model_to_json(init_model('adaptive',12)),
                   'world':{k:batch[k][0].tolist() for k in ('states','queries','noise','random_gates','random_thresholds')}}
        code = """
const c=require('./site/spike-core.js');let s='';process.stdin.on('data',x=>s+=x);
process.stdin.on('end',()=>{const v=JSON.parse(s),r=c.startEpisode(v.model,v.world);
for(let i=0;i<8;i++)c.stepEpisode(r);const before=JSON.stringify(r);
c.answerQuestion(r,v.world.queries[0]);c.answerQuestion(r,v.world.queries[1]);
let blocked=false;try{c.stepEpisode(r);}catch(e){blocked=true;}
process.stdout.write(JSON.stringify({blocked,unchanged:before===JSON.stringify(r)}));});"""
        out = subprocess.run(['node','-e',code],input=json.dumps(request),text=True,cwd=ROOT,capture_output=True,check=True)
        self.assertEqual(json.loads(out.stdout),{'blocked':True,'unchanged':True})


if __name__ == '__main__':
    unittest.main()

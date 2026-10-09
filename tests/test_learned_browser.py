import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from learned_data import make_batch
from learned_receiver import init_model, rollout
from learned_training import model_to_json, model_from_json


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is required for browser/reference parity')
class LearnedBrowserTests(unittest.TestCase):
    def test_browser_read_write_and_predictions_match_reference(self):
        self.assertTrue((ROOT/'site/learned-core.js').is_file(), 'Learned browser core is missing')
        batch = make_batch('demo', 89123, 4, query_kind='novel')
        models = [init_model(method, 7) for method in ('adaptive', 'fixed', 'random', 'recurrent')]
        worlds = [{key: batch[key][i].tolist() for key in
                   ('states', 'noise', 'random_gates', 'queries')} for i in range(4)]
        payload = {'models': [model_to_json(m) for m in models], 'worlds': worlds}
        code = """
const assert=require('node:assert/strict'),fs=require('node:fs');
const core=require('./site/learned-core.js'),data=JSON.parse(fs.readFileSync(0,'utf8')),out=[];
for(const model of data.models)for(const mode of ['normal','no_context','erased','post_mix']){
 const rows=data.worlds.map(world=>{
   const run=core.begin(world,model,mode);for(let k=0;k<3;k++)core.step(run);
   const before=JSON.stringify(run);const predictions=world.queries.map(q=>core.answer(run,q).prediction);
   assert.equal(JSON.stringify(run),before,'Final questions must reuse acquired memory');
   assert.throws(()=>core.step(run),/budget/);assert.equal(run.observations.length,4);
   return {gates:run.gates,observations:run.observations,hidden:run.h,
           predicted_state:core.predictState(model,run.h),predictions,final_state:run.actual};
 });out.push({method:model.method,mode,rows});
}console.log(JSON.stringify(out));
"""
        out = subprocess.check_output(['node','-e',code],cwd=ROOT,input=json.dumps(payload),text=True)
        by_method = {m['method']: m for m in models}
        for result in json.loads(out):
            expected = rollout(by_method[result['method']],batch,mode=result['mode'])
            for i, row in enumerate(result['rows']):
                for field in ('gates','observations','predicted_state','predictions','final_state'):
                    np.testing.assert_allclose(row[field],expected[field][i],atol=1e-11)
                np.testing.assert_allclose(row['hidden'],expected['hidden'][i,-1],atol=1e-11)

    def test_actual_exported_weights_and_receipt_are_used(self):
        self.assertTrue((ROOT/'site/learned-data.js').is_file(), 'Trained browser data is missing')
        code = "console.log(JSON.stringify(require('./site/learned-data.js')))"
        data = json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True))
        receipt = json.loads((ROOT/'results/learned_queries.json').read_text())
        self.assertEqual(data['receipt'],receipt)
        weights = json.loads((ROOT/'results/learned_weights.json').read_text())
        selected = [r['model'] for r in weights['models'] if r['model']['seed']==20261009]
        self.assertEqual(data['models'],selected)
        # All exported models also match the real reference on the demo histories.
        js = """
const core=require('./site/learned-core.js'),data=require('./site/learned-data.js');
console.log(JSON.stringify(data.models.map(model=>data.worlds.map(world=>{
 const run=core.begin(world,model);for(let i=0;i<3;i++)core.step(run);
 return core.predictState(model,run.h);
}))));
"""
        predictions = json.loads(subprocess.check_output(['node','-e',js],cwd=ROOT,text=True))
        for index, model in enumerate(data['models']):
            batch = {field:np.array([w[field] for w in data['worlds']]) for field in
                     ('states','noise','random_gates','queries')}
            expected = rollout(model_from_json(model),batch)['predicted_state']
            np.testing.assert_allclose(predictions[index],expected,atol=1e-11)


if __name__ == '__main__':
    unittest.main()

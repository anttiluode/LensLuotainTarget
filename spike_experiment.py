"""Train all controls, seal them, then evaluate the frozen noisy-world test."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from learned_training import paired_interval
from spike_data import make_batch
from spike_receiver import METHODS, resources, rollout
from spike_training import TRAIN_SEEDS, UPDATES, train, model_to_json, model_from_json, model_hash

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('SPIKE_PROTOCOL.md','spike_data.py','spike_receiver.py','spike_training.py',
                'spike_experiment.py','learned_data.py','memory_gate_soma.py')


def source_fingerprints():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def payload_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def train_worker(method,seed,path):
    before = source_fingerprints()
    def log(row):
        if row['step'] % 1000 == 0:
            print(f"{method} {seed} step {row['step']}: validation J {row['validation_objective']:.5f}",flush=True)
    run = train(method,seed,log=log)
    run['model'] = model_to_json(run['model'])
    run.update(method=method,seed=seed)
    if before != source_fingerprints():
        raise ValueError('training source changed during worker')
    run['training_sources'] = before
    Path(path).write_text(json.dumps(run,separators=(',',':'))+'\n')
    return run


def train_all(checkpoint_dir,workers=4):
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True,exist_ok=True)
    before = source_fingerprints()
    runs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for method in METHODS:
            for seed in TRAIN_SEEDS:
                path = directory/f'{method}-{seed}.json'
                if path.exists():
                    runs.append(json.loads(path.read_text()))
                else:
                    futures.append(pool.submit(train_worker,method,seed,str(path)))
        for future in as_completed(futures):
            runs.append(future.result())
    if source_fingerprints() != before:
        raise ValueError('training source changed during run')
    runs.sort(key=lambda r:(METHODS.index(r['method']),r['seed']))
    seal = {'version':1,'sealed_utc':datetime.now(timezone.utc).isoformat(),
            'sources':before,'runs':runs}
    seal['seal_sha256'] = payload_hash(seal)
    validate_seal(seal)
    (ROOT/'results/spike_weights.json').write_text(json.dumps(seal,separators=(',',':'))+'\n')
    return seal


def validate_seal(seal):
    if seal.get('sources') != source_fingerprints():
        raise ValueError('sealed source mismatch')
    expected = {(m,s) for m in METHODS for s in TRAIN_SEEDS}
    runs = seal.get('runs',[])
    if len(runs) != len(expected) or {(r['method'],r['seed']) for r in runs} != expected:
        raise ValueError('all 24 frozen runs are required before evaluation')
    for r in runs:
        if r.get('training_sources') != seal['sources']:
            raise ValueError('resumed run training source mismatch')
        model = model_from_json(r['model'])
        steps = [row['step'] for row in r['trace']]
        selected = min(r['trace'],key=lambda row:row['validation_objective'])
        if (r['updates'] != UPDATES or r['batch_size'] != 96 or r['noise_draws'] != 4 or
            r['validation_seed'] != 11000 or steps != list(range(0,UPDATES+1,100)) or
            r['selected_step'] != selected['step'] or r['selected_value'] != selected['validation_objective'] or
            model['method'] != r['method'] or model['seed'] != r['seed'] or
            model_hash(model) != r['weight_sha256']):
            raise ValueError('training receipt or selected checkpoint mismatch')
    if seal.get('seal_sha256') != payload_hash({k:v for k,v in seal.items() if k != 'seal_sha256'}):
        raise ValueError('seal checksum mismatch')
    return True


def comparison(baseline,current,baseline_j,current_j,require_objective=True):
    baseline,current = np.asarray(baseline),np.asarray(current)
    if baseline.shape != current.shape or baseline.ndim != 2:
        raise ValueError('models by independent histories required')
    interval = paired_interval(baseline.mean(axis=0),current.mean(axis=0),seed=20261012)
    wins = int(np.sum(current.mean(axis=1) < baseline.mean(axis=1)))
    objective_better = bool(np.mean(current_j) < np.mean(baseline_j))
    return {'relative_gain':interval['relative_gain'],'interval':interval,
            'seed_wins':wins,'initializations':len(baseline),
            'independent_histories':baseline.shape[1], 'objective_better':objective_better,
            'passed':bool(interval['relative_gain'] >= .05 and interval['low'] > 0 and
                          wins >= 2 and (objective_better or not require_objective))}


def evaluate(seal):
    validate_seal(seal)
    conditions = {
        'familiar':make_batch('test',62000,512,draws=8),
        'main':make_batch('test',62000,512,draws=8,query_kind='novel'),
        'transfer':make_batch('transfer',72000,512,draws=8,query_kind='novel',history_kind='switching')}
    tables, controls, raw, integrity = {},{}, {},True
    for name,batch in conditions.items():
        table, raw[name] = {},{}
        for method in METHODS:
            outputs = [rollout(model_from_json(r['model']),batch) for r in seal['runs'] if r['method'] == method]
            mse = np.stack([o['mse'].reshape(512,8).mean(axis=1) for o in outputs])
            js = np.stack([o['objective'].reshape(512,8).mean(axis=1) for o in outputs])
            zero = float(np.mean(outputs[0]['zero_mse']))
            table[method] = {'mse':float(np.mean(mse)),'rmse':float(np.sqrt(np.mean(mse))),
                             'normalized_mse':float(np.mean(mse)/zero),'objective':float(np.mean(js)),
                             'mean_squared_displacement':float(np.mean([np.mean(o['squared_damage']) for o in outputs])),
                             'per_seed_mse':mse.mean(axis=1).tolist()}
            raw[name][method] = (mse,js)
            for out in outputs:
                integrity &= bool(np.all(np.isfinite(out['objective'])) and
                                  np.allclose(np.linalg.norm(out['gates'],axis=-1),1) and
                                  np.all(out['gates'] > 0) and np.all(np.abs(out['thresholds']) <= .15))
                if not method.startswith('graded'):
                    integrity &= bool(np.all(np.isin(out['observations'],[-1,1])))
        tables[name] = table
        if name != 'familiar':
            controls[name] = {}
            for mode in ('erased','post_mix'):
                outputs = [rollout(model_from_json(r['model']),batch,mode=mode) for r in seal['runs'] if r['method'] == 'adaptive']
                ratio = float(np.mean([np.mean(o['mse']) for o in outputs])/np.mean(outputs[0]['zero_mse']))
                controls[name][mode] = {'normalized_mse':ratio,'passed':bool(ratio >= .90)}
    for r in seal['runs']:
        resource = resources(model_from_json(r['model']))
        integrity &= resource['parameters'] == 1252 and resource['neural_macs'] == 7376
    gates = {'S1':{'simulation_integrity':bool(integrity),
                   'publication_requires':'numerical tests and browser/Python parity; see SPIKE_RESULTS.md'}}
    for gate,condition,baseline,obj in (('S2','main','fixed',True),('S3','main','gate_only',False),
                                       ('S4','main','threshold_only',False),('S5','transfer','fixed',True)):
        bm,bj = raw[condition][baseline]
        am,aj = raw[condition]['adaptive']
        gates[gate] = {'condition':condition,'baseline':baseline,
                       **comparison(bm,am,bj,aj,require_objective=obj)}
    gates['S6'] = {'passed':all(v['passed'] for c in controls.values() for v in c.values())}
    return {'version':1,'protocol':'SPIKE_PROTOCOL.md','weights_seal_sha256':seal['seal_sha256'],
            'sources':seal['sources'],'histories_per_condition':512,'noise_draws_per_history':8,
            'initialization_seeds':list(TRAIN_SEEDS),'budget':8,'noise_sd':.03,'eta':.25,
            'tables':tables,'memory_controls':controls,'gates':gates,
            'statistical_binary_interface_passed':bool(gates['S2']['passed'] and gates['S5']['passed'] and gates['S6']['passed']),
            'statistical_joint_gate_threshold_passed':bool(all(gates[g]['passed'] for g in ('S2','S3','S4','S5','S6')))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','evaluate','verify'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--checkpoint-dir',default=str(ROOT.parent/'spike-checkpoints'))
    args = parser.parse_args()
    if args.action == 'train':
        seal = train_all(args.checkpoint_dir,args.workers)
        print('All 24 models sealed:',seal['seal_sha256'])
    else:
        seal = json.loads((ROOT/'results/spike_weights.json').read_text())
        result = evaluate(seal)
        path = ROOT/'results/learned_spikes.json'
        if args.action == 'verify':
            stored = json.loads(path.read_text())
            def check(a,b):
                if isinstance(a,dict):
                    return a.keys() == b.keys() and all(check(a[k],b[k]) for k in a)
                if isinstance(a,list):
                    return len(a) == len(b) and all(check(x,y) for x,y in zip(a,b))
                if isinstance(a,float):
                    return np.isclose(a,b,rtol=1e-10,atol=1e-12)
                return a == b
            if not check(result,stored):
                raise ValueError('held-out receipt no longer reproduces')
            print('Sealed weights, training receipt and all held-out outcomes verified.')
        else:
            path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result['gates'],indent=2))


if __name__ == '__main__':
    main()

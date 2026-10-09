"""Frozen training, checkpoint receipts and dictionary-free held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from learned_data import make_batch, NAMESPACES
from learned_receiver import METHODS, resources, rollout
from learned_training import (TRAIN_SEEDS, train, model_to_json, model_from_json,
                              model_hash, paired_interval)


ROOT = Path(__file__).resolve().parent
MODES = {'adaptive': 'normal', 'fixed': 'normal', 'random': 'normal',
         'recurrent': 'normal', 'no_context': 'no_context',
         'erased': 'erased', 'post_mix': 'post_mix'}


def fingerprint():
    source = b''.join((ROOT/p).read_bytes() for p in
                      ('learned_data.py', 'learned_receiver.py', 'learned_training.py'))
    return {'protocol_sha256': hashlib.sha256((ROOT/'LEARNED_PROTOCOL.md').read_bytes()).hexdigest(),
            'learning_source_sha256': hashlib.sha256(source).hexdigest()}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_training_record(record):
    if record['updates'] != 2000 or record['batch_size'] != 96:
        raise ValueError('checkpoint was not trained with the frozen budget')
    if [row['step'] for row in record['trace']] != list(range(0, 2001, 100)):
        raise ValueError('validation checkpoint trace is incomplete')
    selected = min(record['trace'], key=lambda row: row['validation_objective'])
    if record['selected_step'] != selected['step'] or record['selected_value'] != selected['validation_objective']:
        raise ValueError('checkpoint is not the validation-selected minimum')
    model = model_from_json(record['model'])
    if record['weight_sha256'] != model_hash(model):
        raise ValueError('checkpoint weight hash mismatch')
    return model


def train_all(path):
    current = fingerprint()
    output = {'date': '2026-10-09', 'protocol': 'LEARNED_PROTOCOL.md',
              **current, 'training_seeds': list(TRAIN_SEEDS), 'models': []}
    path = Path(path)
    if path.exists():
        output = json.loads(path.read_text())
        if any(output.get(k) != v for k, v in current.items()):
            raise ValueError('existing checkpoint file has a different frozen source/protocol')
        for record in output['models']:
            validate_training_record(record)
    completed = {(r['model']['seed'], r['model']['method']) for r in output['models']}
    for seed in TRAIN_SEEDS:
        for method in METHODS:
            if (seed, method) in completed:
                print('resume:', seed, method, 'already complete', flush=True)
                continue

            def log(row):
                if row['step'] % 500 == 0:
                    print(seed, method, 'step', row['step'],
                          'validation objective', round(row['validation_objective'], 6), flush=True)

            result = train(method, seed, log=log)
            result['model'] = model_to_json(result['model'])
            validate_training_record(result)
            output['models'].append(result)
            save_json(path, output)
            print('selected:', seed, method, result['selected_step'], result['weight_sha256'][:12], flush=True)
    return output


def load_models(path):
    output = json.loads(Path(path).read_text())
    if any(output.get(k) != v for k, v in fingerprint().items()):
        raise ValueError('weights do not match frozen source/protocol')
    models = {}
    for record in output['models']:
        model = validate_training_record(record)
        key = (model['seed'], model['method'])
        if key in models:
            raise ValueError('duplicate training seed/method')
        models[key] = model
    if set(models) != {(s, m) for s in TRAIN_SEEDS for m in METHODS}:
        raise ValueError('all three seeds and four models are required')
    return output, models


def evaluate_condition(name, batch, models):
    rows, summary = {}, {}
    for method, mode in MODES.items():
        by_seed = []
        for seed in TRAIN_SEEDS:
            model = models[(seed, method if method in METHODS else 'adaptive')]
            result = rollout(model, batch, mode=mode)
            by_seed.append(result)
        rows[method] = by_seed
        per_seed = []
        for seed, result in zip(TRAIN_SEEDS, by_seed):
            per_seed.append({'seed': seed, 'mse': float(np.mean(result['mse'])),
                             'nrmse': float(np.sqrt(np.mean(result['mse'])/np.mean(result['zero_mse']))),
                             'objective': float(np.mean(result['objective'])),
                             'relative_disturbance': float(np.mean(result['relative_disturbance']))})
        summary[method] = {'mse': float(np.mean([r['mse'] for r in by_seed])),
                           'nrmse': float(np.sqrt(np.mean([r['mse'] for r in by_seed])/
                                                 np.mean(by_seed[0]['zero_mse']))),
                           'objective': float(np.mean([r['objective'] for r in by_seed])),
                           'relative_disturbance': float(np.mean([r['relative_disturbance'] for r in by_seed])),
                           'per_seed': per_seed}
    comparisons = {}
    active = np.mean([r['mse'] for r in rows['adaptive']], axis=0)
    for index, method in enumerate(('random', 'fixed', 'recurrent')):
        baseline = np.mean([r['mse'] for r in rows[method]], axis=0)
        wins = sum(np.mean(a['mse']) < np.mean(b['mse'])
                   for a, b in zip(rows['adaptive'], rows[method]))
        comparisons[method] = {'mse': paired_interval(baseline, active, 20261009+index),
                               'positive_seed_wins': int(wins),
                               'objective_gain': summary[method]['objective']-summary['adaptive']['objective']}
    compact = {'name': name, 'episodes': len(batch['states']), 'training_replicates': 3,
               'zero_predictor_mse': float(np.mean(rows['adaptive'][0]['zero_mse'])),
               'summary': summary, 'comparisons': comparisons}
    details = {m: [{'seed': s, **{field: r[field].tolist() for field in
                    ('mse', 'objective', 'relative_disturbance')}}
                  for s, r in zip(TRAIN_SEEDS, values)] for m, values in rows.items()}
    return compact, details


def frozen_gates(receipt, contract_passed=None):
    conditions = {c['name']: c for c in receipt['conditions']}

    def beats(c, method, margin, require_objective):
        comparison = c['comparisons'][method]
        return (comparison['mse']['relative_gain'] >= margin and comparison['mse']['low'] > 0
                and comparison['positive_seed_wins'] >= 2
                and (not require_objective or comparison['objective_gain'] > 0))

    main, transfer = conditions['novel_queries'], conditions['unseen_history']
    return {'L1_contract_checks': contract_passed,
            'L2_random': bool(beats(main, 'random', .10, False)),
            'L3_fixed_and_recurrent': bool(all(beats(main, m, .05, True) for m in ('fixed', 'recurrent'))),
            'L4_unseen_history_transfer': bool(all(beats(transfer, m, .05, True) for m in ('fixed', 'recurrent'))),
            'L5_information_loss_controls': bool(all(c['summary'][m]['nrmse'] >= .90
                        for c in (main, transfer) for m in ('erased', 'post_mix')))}


def evaluate(weights_path, output, details=None):
    checkpoints, models = load_models(weights_path)
    conditions, detailed = [], {}
    descriptions = [('fresh_familiar', 'test', 61000, 'familiar', 'mixture'),
                    ('novel_queries', 'test', 61000, 'novel', 'mixture'),
                    ('unseen_history', 'transfer', 71000, 'novel', 'switching')]
    for name, namespace, seed, query_kind, history_kind in descriptions:
        batch = make_batch(namespace, seed, 512, query_kind, history_kind)
        compact, rows = evaluate_condition(name, batch, models)
        compact['data'] = {'namespace': namespace, 'seed': seed,
                           'query_kind': query_kind, 'history_kind': history_kind}
        conditions.append(compact)
        detailed[name] = rows
    receipt = {'date': '2026-10-09', 'protocol': 'LEARNED_PROTOCOL.md', **fingerprint(),
               'evaluation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'config': {'branches': 12, 'history_steps': 32, 'noise': .03, 'eta': .25,
                          'new_readings': 3, 'future_queries': 6, 'updates': 2000, 'batch_size': 96,
                          'training_seeds': list(TRAIN_SEEDS), 'rng_namespaces': NAMESPACES,
                          'state_scale': .1, 'damage_weight': .1, 'query_count_weight': .001},
               'resources': {m: resources(models[(TRAIN_SEEDS[0], m)]) for m in METHODS},
               'weights': [{'seed': r['model']['seed'], 'method': r['model']['method'],
                            'selected_step': r['selected_step'], 'sha256': r['weight_sha256']}
                           for r in checkpoints['models']], 'conditions': conditions}
    receipt['gates'] = frozen_gates(receipt)
    receipt['adaptive_advantage_earned'] = all(v is True for v in receipt['gates'].values())
    receipt['contract_note'] = 'Reference/gradient tests passed; final browser parity remains to be verified.'
    save_json(output, receipt)
    if details:
        save_json(details, detailed)
    for c in conditions:
        print(c['name'], flush=True)
        for m, row in c['summary'].items():
            print(f"  {m:12} NRMSE {row['nrmse']:.4f}  objective {row['objective']:.5f}  displacement {row['relative_disturbance']:.3f}")
    print('Frozen gates:', receipt['gates'], flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'evaluate'))
    p.add_argument('--weights', default='results/learned_weights.json')
    p.add_argument('--output', default='results/learned_queries.json')
    p.add_argument('--details')
    args = p.parse_args()
    if args.command == 'train':
        train_all(args.weights)
    else:
        evaluate(args.weights, args.output, args.details)


if __name__ == '__main__':
    main()

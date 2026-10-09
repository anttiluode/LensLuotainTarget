"""Exact small-model backpropagation and fixed-budget, validation-only training."""
from copy import deepcopy
import hashlib
import json

import numpy as np

from learned_data import make_batch
from learned_receiver import (BUDGET, DAMAGE_WEIGHT, SCALE, ETA, init_model, rollout)


TRAIN_SEEDS = (20261009, 20261010, 20261011)


def loss_and_grad(model, batch):
    result = rollout(model, batch, cache=True)
    cache, p = result['_cache'], model['params']
    grad = {name: np.zeros_like(value) for name, value in p.items()}
    count, questions = result['targets'].shape
    error = (result['predictions']-result['targets'])/SCALE
    de = 2*np.einsum('bk,bkd->bd', error, cache['queries'])/(count*questions)
    h = result['hidden'][:, -1]
    grad['decoder'] = h.T @ de
    grad['decoder_b'] = de.sum(axis=0)
    dh = de @ p['decoder'].T
    dm = 2*DAMAGE_WEIGHT*(result['final_state']-cache['original'])/(count*SCALE**2)
    for step in reversed(range(BUDGET)):
        row = cache['steps'][step]
        gate, memory, value = row['g'], row['m'], row['value']
        delta = dh*(1-row['next_h']**2)
        grad['recurrent'] += row['h'].T @ delta
        grad['gate_input'] += gate.T @ delta
        grad['answer_input'] += np.sum(delta*(row['answer']/SCALE)[:, None], axis=0)
        grad['bias'] += delta.sum(axis=0)
        dh = delta @ p['recurrent'].T
        da = np.sum(delta*p['answer_input'], axis=1)/SCALE
        projection = np.sum(dm*gate, axis=1)
        dg = delta @ p['gate_input'].T+da[:, None]*memory
        dg -= ETA*(value[:, None]*dm+projection[:, None]*memory)
        dm = dm-ETA*projection[:, None]*gate+da[:, None]*gate
        if model['method'] != 'random':
            # g=exp(logit)/||exp(logit)||: include normalization derivative.
            dl = gate*(dg-gate*np.sum(dg*gate, axis=1, keepdims=True))
            grad['gate_logits'][step] += dl.sum(axis=0)
            if model['method'] == 'adaptive':
                grad['policy'] += row['h'].T @ dl
                dh += dl @ p['policy'].T
    delta = dh*(1-cache['initial_h']**2)
    grad['initial_w'] = np.sum(delta*(cache['initial']/SCALE)[:, None], axis=0)
    grad['initial_b'] = delta.sum(axis=0)
    return float(np.mean(result['objective'])), grad


def model_to_json(model):
    return {**{k: v for k, v in model.items() if k != 'params'},
            'params': {k: v.tolist() for k, v in model['params'].items()}}


def model_from_json(value):
    expected = init_model(value['method'], value['seed'])
    if value['hidden'] != expected['hidden'] or set(value['params']) != set(expected['params']):
        raise ValueError('checkpoint architecture mismatch')
    params = {name: np.asarray(v, dtype=float) for name, v in value['params'].items()}
    for name, weight in params.items():
        if weight.shape != expected['params'][name].shape or not np.all(np.isfinite(weight)):
            raise ValueError('invalid checkpoint weight '+name)
    return {**value, 'params': params}


def model_hash(model):
    source = json.dumps(model_to_json(model), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(source).hexdigest()


def train(method, seed, steps=2000, batch_size=96, log=None):
    if not isinstance(steps, int) or steps < 0 or batch_size < 1:
        raise ValueError('invalid training budget')
    model = init_model(method, seed)
    p = model['params']
    first = {name: np.zeros_like(value) for name, value in p.items()}
    second = deepcopy(first)
    validation = make_batch('validation', 10000, 512)
    best_model, best_value, best_step = None, float('inf'), 0
    trace = []

    def checkpoint(step, training_objective=None):
        nonlocal best_model, best_value, best_step
        result = rollout(model, validation)
        value = float(np.mean(result['objective']))
        if not np.isfinite(value):
            raise FloatingPointError('non-finite validation objective')
        row = {'step': step, 'validation_objective': value,
               'validation_mse': float(np.mean(result['mse'])),
               'training_objective': training_objective}
        trace.append(row)
        if value < best_value:
            best_model, best_value, best_step = deepcopy(model), value, step
        if log is not None:
            log(row)

    checkpoint(0)
    for step in range(1, steps+1):
        batch = make_batch('train', int(seed)*10000+step, batch_size)
        loss, gradient = loss_and_grad(model, batch)
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in gradient.values()))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError('non-finite training loss/gradient')
        factor = min(1., 1/max(norm, 1e-30))
        for name in p:
            g = gradient[name]*factor
            first[name] = .9*first[name]+.1*g
            second[name] = .999*second[name]+.001*g*g
            corrected_first = first[name]/(1-.9**step)
            corrected_second = second[name]/(1-.999**step)
            p[name] -= .003*corrected_first/(np.sqrt(corrected_second)+1e-8)
        if step % 100 == 0 or step == steps:
            checkpoint(step, loss)
    return {'model': best_model, 'selected_step': best_step, 'selected_value': best_value,
            'updates': steps, 'batch_size': batch_size, 'trace': trace,
            'weight_sha256': model_hash(best_model)}


def paired_interval(baseline, current, seed=20261009):
    baseline, current = np.asarray(baseline), np.asarray(current)
    if baseline.ndim != 1 or baseline.shape != current.shape or not len(baseline):
        raise ValueError('paired episode losses must be nonempty vectors of equal size')
    differences = baseline-current
    rng = np.random.default_rng(seed)
    samples = np.mean(differences[rng.integers(0, len(differences), size=(2048, len(differences)))], axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {'mean': float(np.mean(differences)), 'low': float(low), 'high': float(high),
            'relative_gain': float(np.mean(differences)/max(float(np.mean(baseline)), 1e-30)),
            'resamples': 2048}

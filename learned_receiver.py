"""Observation-only learned receiver plus a separate, explicitly known simulator.

The four receiver functions below do not take histories, branch states, candidate
predictions, or future questions. The external rollout owns the hidden plant.
"""
import numpy as np

from learned_data import normalized_positive


METHODS = ('adaptive', 'fixed', 'random', 'recurrent')
SCALE = .1
ETA = .25
BUDGET = 3
DAMAGE_WEIGHT = .1
QUERY_WEIGHT = .001


def init_model(method, seed):
    if method not in METHODS:
        raise ValueError('unknown receiver method')
    hidden = 19 if method == 'recurrent' else 16
    rng = np.random.default_rng([int(seed), 98173])
    orthogonal, _ = np.linalg.qr(rng.normal(size=(hidden, hidden)))
    params = {'initial_w': .3*rng.normal(size=hidden), 'initial_b': np.zeros(hidden),
              'recurrent': .7*orthogonal, 'gate_input': .1*rng.normal(size=(12, hidden)),
              'answer_input': .3*rng.normal(size=hidden), 'bias': np.zeros(hidden),
              'decoder': .05*rng.normal(size=(hidden, 12)), 'decoder_b': np.zeros(12)}
    if method == 'adaptive':
        params['policy'] = .1*rng.normal(size=(hidden, 12))
    if method != 'random':
        params['gate_logits'] = np.random.default_rng([int(seed), 72113]).normal(size=(BUDGET, 12))
    return {'method': method, 'hidden': hidden, 'seed': int(seed), 'params': params}


def initial_receiver(model, initial_answer):
    p = model['params']
    return np.tanh(np.asarray(initial_answer)[:, None]/SCALE*p['initial_w']+p['initial_b'])


def select_gate(model, h, step, random_gate=None):
    if not isinstance(step, (int, np.integer)) or not 0 <= step < BUDGET:
        raise ValueError('read index outside three-read budget')
    p = model['params']
    if model['method'] == 'random':
        if random_gate is None:
            raise ValueError('random policy needs its independent gate command')
        return np.asarray(random_gate).copy()
    logits = np.broadcast_to(p['gate_logits'][step], (len(h), 12)).copy()
    if model['method'] == 'adaptive':
        logits += h @ p['policy']
    return normalized_positive(logits)


def receive(model, h, gate, answer):
    p = model['params']
    return np.tanh(h @ p['recurrent']+gate @ p['gate_input']+
                   np.asarray(answer)[:, None]/SCALE*p['answer_input']+p['bias'])


def predict_state(model, h):
    p = model['params']
    return SCALE*(h @ p['decoder']+p['decoder_b'])


def resources(model):
    h = model['hidden']
    return {'parameters': sum(p.size for p in model['params'].values()),
            'neural_macs': BUDGET*(h*h+13*h+(12*h if model['method'] == 'adaptive' else 0))+12*h+h,
            'receiver_values': h, 'sender_values': 12, 'persistent_values': 12+h,
            'prediction_working_values': 12, 'gate_command_values_per_read': 12,
            'new_scalar_readings': BUDGET,
            'excluded_from_macs': 'tanh, exponentials, normalization, sender work and training optimizer'}


def rollout(model, batch, mode='normal', cache=False):
    """Environment orchestration; physical state stays outside receiver functions."""
    if mode not in ('normal', 'no_context', 'erased', 'post_mix'):
        raise ValueError('unknown inference ablation')
    original = batch['states'].copy()
    state = np.zeros_like(original) if mode == 'erased' else original.copy()
    uniform = np.ones_like(state)/np.sqrt(12)
    initial = np.sum(state*uniform, axis=1)+batch['noise'][:, 0]
    h = initial_receiver(model, initial)
    initial_h = h.copy()
    records, observations, gates, hidden = [], [initial], [], [h.copy()]
    for step in range(BUDGET):
        chosen = select_gate(model, np.zeros_like(h) if mode == 'no_context' else h,
                             step, batch['random_gates'][:, step])
        gate = uniform if mode == 'post_mix' else chosen
        value = np.sum(state*gate, axis=1)
        answer = value+batch['noise'][:, step+1]
        next_h = receive(model, h, gate, answer)
        records.append({'h': h, 'm': state, 'g': gate, 'value': value,
                        'answer': answer, 'next_h': next_h})
        state = state-ETA*gate*value[:, None]
        h = next_h
        gates.append(gate.copy())
        observations.append(answer)
        hidden.append(h.copy())
    predicted = predict_state(model, h)
    target = np.einsum('bkd,bd->bk', batch['queries'], original)
    prediction = np.einsum('bkd,bd->bk', batch['queries'], predicted)
    error = np.mean((prediction-target)**2, axis=1)
    squared_damage = np.sum((state-original)**2, axis=1)
    result = {'predicted_state': predicted, 'predictions': prediction, 'targets': target,
              'mse': error, 'zero_mse': np.mean(target**2, axis=1),
              'objective': error/SCALE**2+DAMAGE_WEIGHT*squared_damage/SCALE**2+QUERY_WEIGHT*BUDGET,
              'squared_damage': squared_damage,
              'relative_disturbance': np.sqrt(squared_damage)/np.maximum(np.linalg.norm(original, axis=1), 1e-12),
              'final_state': state, 'gates': np.stack(gates, axis=1),
              'observations': np.stack(observations, axis=1), 'hidden': np.stack(hidden, axis=1)}
    if cache:
        result['_cache'] = {'initial': initial, 'initial_h': initial_h, 'steps': records,
                            'original': original, 'queries': batch['queries']}
    return result

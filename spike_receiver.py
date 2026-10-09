"""An observation-only receiver; the simulator owns the hidden branch memory."""
import numpy as np
from learned_data import normalized_positive
from spike_data import BUDGET, NOISE_SD

METHODS = ('adaptive','fixed','gate_only','threshold_only','zero_threshold','random',
           'graded_adaptive','graded_fixed')
SCALE, THRESHOLD_SCALE, ETA = .1, .15, .25
DAMAGE_WEIGHT, QUERY_WEIGHT = .1, .001


def init_model(method, seed):
    if method not in METHODS:
        raise ValueError('unknown receiver')
    h = 16
    rng = np.random.default_rng([int(seed), 18739])
    orthogonal, _ = np.linalg.qr(rng.normal(size=(h,h)))
    p = {'initial_w': .3*rng.normal(size=h), 'initial_b': np.zeros(h),
         'recurrent': .7*orthogonal,
         'actual_gate_input': .1*rng.normal(size=(12,h)),
         'context_gate_input': .1*rng.normal(size=(12,h)),
         'actual_threshold_input': .1*rng.normal(size=h),
         'context_threshold_input': .1*rng.normal(size=h),
         'answer_input': .3*rng.normal(size=h), 'bias': np.zeros(h),
         'decoder': .05*rng.normal(size=(h,12)), 'decoder_b': np.zeros(12),
         'gate_policy': .1*rng.normal(size=(h,12)),
         'threshold_policy': .1*rng.normal(size=h),
         'gate_logits': rng.normal(size=(BUDGET,12)),
         'threshold_logits': np.zeros(BUDGET)}
    return {'method': method, 'hidden': h, 'seed': int(seed), 'params': p}


def initial_receiver(model, initial_answer):
    p = model['params']
    answer = np.asarray(initial_answer)
    if model['method'].startswith('graded'):
        answer = answer/SCALE
    return np.tanh(answer[:,None]*p['initial_w']+p['initial_b'])


def select_command(model, h, step, random_gate=None, random_threshold=None):
    if not isinstance(step, (int,np.integer)) or not 0 <= step < BUDGET:
        raise ValueError('read outside eight-read budget')
    p, method = model['params'], model['method']
    context_gate = normalized_positive(h @ p['gate_policy']+p['gate_logits'][step])
    context_threshold = THRESHOLD_SCALE*np.tanh(h @ p['threshold_policy']+p['threshold_logits'][step])
    gate = context_gate if method in ('adaptive','gate_only','zero_threshold','graded_adaptive') else np.broadcast_to(normalized_positive(p['gate_logits'][step]), context_gate.shape).copy()
    theta = context_threshold if method in ('adaptive','threshold_only','graded_adaptive') else np.full(len(h),THRESHOLD_SCALE*np.tanh(p['threshold_logits'][step]))
    if method == 'zero_threshold':
        theta = np.zeros(len(h))
    if method == 'random':
        if random_gate is None or random_threshold is None:
            raise ValueError('random receiver needs independent commands')
        gate, theta = np.asarray(random_gate).copy(), np.asarray(random_threshold).copy()
    return {'g': gate, 'theta': theta, 'context_gate': context_gate,
            'context_threshold': context_threshold}


def receive(model, h, command, answer):
    p = model['params']
    reply = np.asarray(answer)/(SCALE if model['method'].startswith('graded') else 1.)
    return np.tanh(h @ p['recurrent']+command['g'] @ p['actual_gate_input']+
                   command['context_gate'] @ p['context_gate_input']+
                   (command['theta']/SCALE)[:,None]*p['actual_threshold_input']+
                   (command['context_threshold']/SCALE)[:,None]*p['context_threshold_input']+
                   reply[:,None]*p['answer_input']+p['bias'])


def predict_state(model, h):
    p = model['params']
    return SCALE*(h @ p['decoder']+p['decoder_b'])


def resources(model):
    h = model['hidden']
    return {'parameters': sum(v.size for v in model['params'].values()),
            'neural_macs': BUDGET*(h*h+40*h)+12*h+h,
            'sender_values': 12, 'receiver_values': h, 'persistent_values': 12+h,
            'new_readings': BUDGET, 'command_values_per_read': 13,
            'excluded_from_macs': 'nonlinearities, normalization, plant and optimizer'}


def rollout(model, batch, mode='normal', cache=False, reads=BUDGET, forced_bits=None):
    """Plant is external to the four pure receiver operations above.

    forced_bits supports exact, small-tree gradient verification. Production
    rollouts always sample actual hard bits from independent Gaussian noise.
    """
    if mode not in ('normal','erased','post_mix') or not isinstance(reads,int) or not 1 <= reads <= BUDGET:
        raise ValueError('invalid rollout')
    original = batch['states'].copy()
    memory = np.zeros_like(original) if mode == 'erased' else original.copy()
    uniform = np.ones_like(memory)/np.sqrt(12)
    graded = model['method'].startswith('graded')
    initial_value = np.sum(memory*uniform, axis=1)+batch['noise'][:,0]
    initial = initial_value if graded else np.where(initial_value > 0,1.,-1.)
    if forced_bits is not None:
        if graded or np.asarray(forced_bits).shape != (len(memory),reads+1) or not np.all(np.isin(forced_bits,[-1,1])):
            raise ValueError('forced replies must be a signed bit trajectory')
        initial = np.asarray(forced_bits)[:,0].astype(float)
    h = initial_receiver(model,initial)
    hidden, observations, gates, thresholds, records = [h.copy()], [initial], [], [], []
    for step in range(reads):
        command = select_command(model,h,step,batch['random_gates'][:,step],batch['random_thresholds'][:,step])
        if mode == 'post_mix':
            command['g'] = uniform.copy()
        gate, theta = command['g'], command['theta']
        value = np.sum(memory*gate,axis=1)
        answer = value+batch['noise'][:,step+1] if graded else np.where(value+batch['noise'][:,step+1] > theta,1.,-1.)
        if forced_bits is not None:
            answer = np.asarray(forced_bits)[:,step+1].astype(float)
        next_h = receive(model,h,command,answer)
        records.append({'h': h, 'm': memory, 'value': value, 'answer': answer,
                        'next_h': next_h, **command})
        memory = memory-ETA*gate*value[:,None]
        h = next_h
        hidden.append(h.copy()); observations.append(answer.copy())
        gates.append(gate.copy()); thresholds.append(theta.copy())
    predicted = predict_state(model,h)
    target = np.einsum('bkd,bd->bk',batch['queries'],original)
    predictions = np.einsum('bkd,bd->bk',batch['queries'],predicted)
    mse = np.mean((predictions-target)**2,axis=1)
    damage = np.sum((memory-original)**2,axis=1)
    result = {'predicted_state': predicted, 'predictions': predictions, 'targets': target,
              'mse': mse, 'zero_mse': np.mean(target**2,axis=1),
              'objective': mse/SCALE**2+DAMAGE_WEIGHT*damage/SCALE**2+QUERY_WEIGHT*reads,
              'squared_damage': damage, 'final_state': memory,
              'gates': np.stack(gates,axis=1), 'thresholds': np.stack(thresholds,axis=1),
              'observations': np.stack(observations,axis=1), 'hidden': np.stack(hidden,axis=1)}
    if cache:
        result['_cache'] = {'initial': initial, 'steps': records, 'original': original,
                            'queries': batch['queries']}
    return result

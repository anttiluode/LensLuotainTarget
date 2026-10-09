"""Hard-bit likelihood-score gradients, exact plant derivatives and Adam."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from scipy.special import log_ndtr
from spike_data import make_batch, BUDGET, NOISE_SD
from spike_receiver import SCALE, THRESHOLD_SCALE, ETA, DAMAGE_WEIGHT, init_model, rollout

TRAIN_SEEDS = (20261012,20261013,20261014)
UPDATES = 4000


def probit_score(z, signed_bit):
    """d log P(bit | z)/dz, including extremely improbable observed bits."""
    z, signed_bit = np.asarray(z), np.asarray(signed_bit)
    return signed_bit*np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(signed_bit*z))


def independent_baseline(losses, draws):
    losses = np.asarray(losses)
    if draws < 2:
        return np.zeros_like(losses)
    grouped = losses.reshape(-1,draws)
    return ((grouped.sum(axis=1,keepdims=True)-grouped)/(draws-1)).ravel()


def gate_derivative(g, dg):
    return g*(dg-g*np.sum(g*dg,axis=1,keepdims=True))


def loss_and_grad(model,batch,reads=BUDGET,forced_bits=None):
    result = rollout(model,batch,cache=True,reads=reads,forced_bits=forced_bits)
    p, c = model['params'], result['_cache']
    grad = {k:np.zeros_like(v) for k,v in p.items()}
    count, questions = result['targets'].shape
    graded = model['method'].startswith('graded')
    error = (result['predictions']-result['targets'])/SCALE
    de = 2*np.einsum('bk,bkd->bd',error,c['queries'])/(count*questions)
    grad['decoder'] = result['hidden'][:,-1].T @ de
    grad['decoder_b'] = de.sum(axis=0)
    dh = de @ p['decoder'].T
    dm = 2*DAMAGE_WEIGHT*(result['final_state']-c['original'])/(count*SCALE**2)
    advantage = (result['objective']-independent_baseline(result['objective'],batch['draws']))/count
    for step in reversed(range(reads)):
        row = c['steps'][step]
        h, gate, memory, value = row['h'],row['g'],row['m'],row['value']
        delta = dh*(1-row['next_h']**2)
        grad['recurrent'] += h.T @ delta
        grad['actual_gate_input'] += gate.T @ delta
        grad['context_gate_input'] += row['context_gate'].T @ delta
        grad['actual_threshold_input'] += np.sum(delta*(row['theta']/SCALE)[:,None],axis=0)
        grad['context_threshold_input'] += np.sum(delta*(row['context_threshold']/SCALE)[:,None],axis=0)
        grad['answer_input'] += np.sum(delta*(row['answer']/(SCALE if graded else 1.))[:,None],axis=0)
        grad['bias'] += delta.sum(axis=0)
        dh = delta @ p['recurrent'].T
        dg = delta @ p['actual_gate_input'].T
        dcg = delta @ p['context_gate_input'].T
        dt = np.sum(delta*p['actual_threshold_input'],axis=1)/SCALE
        dct = np.sum(delta*p['context_threshold_input'],axis=1)/SCALE
        if graded:
            dv = np.sum(delta*p['answer_input'],axis=1)/SCALE
        else:
            # No derivative through the bit. Advantage is a detached coefficient.
            dv = advantage*probit_score((value-row['theta'])/NOISE_SD,row['answer'])/NOISE_SD
            dt -= dv
        projection = np.sum(dm*gate,axis=1)
        dg += dv[:,None]*memory-ETA*(value[:,None]*dm+projection[:,None]*memory)
        dm = dm-ETA*projection[:,None]*gate+dv[:,None]*gate
        if model['method'] in ('adaptive','gate_only','zero_threshold','graded_adaptive'):
            dcg += dg
        elif model['method'] != 'random':
            grad['gate_logits'][step] += gate_derivative(gate,dg).sum(axis=0)
        if model['method'] in ('adaptive','threshold_only','graded_adaptive'):
            dct += dt
        elif model['method'] not in ('random','zero_threshold'):
            grad['threshold_logits'][step] += np.sum(dt*THRESHOLD_SCALE*(1-(row['theta']/THRESHOLD_SCALE)**2))
        dl = gate_derivative(row['context_gate'],dcg)
        dlt = dct*THRESHOLD_SCALE*(1-(row['context_threshold']/THRESHOLD_SCALE)**2)
        grad['gate_logits'][step] += dl.sum(axis=0)
        grad['threshold_logits'][step] += dlt.sum()
        grad['gate_policy'] += h.T @ dl
        grad['threshold_policy'] += h.T @ dlt
        dh += dl @ p['gate_policy'].T+dlt[:,None]*p['threshold_policy']
    delta = dh*(1-result['hidden'][:,0]**2)
    grad['initial_w'] = np.sum(delta*(c['initial']/(SCALE if graded else 1.))[:,None],axis=0)
    grad['initial_b'] = delta.sum(axis=0)
    return float(np.mean(result['objective'])),grad


def model_to_json(model):
    return {**{k:v for k,v in model.items() if k != 'params'},
            'params':{k:v.tolist() for k,v in model['params'].items()}}


def model_from_json(value):
    expected = init_model(value['method'],value['seed'])
    if value['hidden'] != expected['hidden'] or set(value['params']) != set(expected['params']):
        raise ValueError('checkpoint architecture mismatch')
    params = {k:np.asarray(v,dtype=float) for k,v in value['params'].items()}
    for k,v in params.items():
        if v.shape != expected['params'][k].shape or not np.all(np.isfinite(v)):
            raise ValueError('invalid checkpoint '+k)
    return {**value,'params':params}


def model_hash(model):
    return hashlib.sha256(json.dumps(model_to_json(model),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def train(method,seed,steps=UPDATES,batch_size=96,log=None):
    if not isinstance(steps,int) or steps < 0 or not isinstance(batch_size,int) or batch_size < 1:
        raise ValueError('invalid training budget')
    model = init_model(method,seed)
    first = {k:np.zeros_like(v) for k,v in model['params'].items()}
    second = deepcopy(first)
    validation = make_batch('validation',11000,512,draws=4)
    best,best_value,best_step,trace = None,float('inf'),0,[]
    def checkpoint(step,loss=None):
        nonlocal best,best_value,best_step
        out = rollout(model,validation)
        value = float(np.mean(out['objective']))
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite validation')
        row = {'step':step,'validation_objective':value,
               'validation_mse':float(np.mean(out['mse'])),'training_objective':loss}
        trace.append(row)
        if value < best_value:
            best,best_value,best_step = deepcopy(model),value,step
        if log is not None:
            log(row)
    checkpoint(0)
    for step in range(1,steps+1):
        batch = make_batch('train',int(seed)*10000+step,batch_size,draws=4)
        loss,gradient = loss_and_grad(model,batch)
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in gradient.values()))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError('nonfinite loss or gradient')
        factor = min(1.,1/max(norm,1e-30))
        for k,p in model['params'].items():
            g = gradient[k]*factor
            first[k] = .9*first[k]+.1*g
            second[k] = .999*second[k]+.001*g*g
            p -= .003*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        if step % 100 == 0 or step == steps:
            checkpoint(step,loss)
    return {'model':best,'selected_step':best_step,'selected_value':best_value,
            'updates':steps,'batch_size':batch_size,'noise_draws':4,
            'validation_seed':11000,'trace':trace,'weight_sha256':model_hash(best)}

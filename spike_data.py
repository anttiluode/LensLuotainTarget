"""New independent streams; shared histories, independent noisy trajectories."""
import numpy as np
from learned_data import make_batch as original_batch, normalized_positive, NAMESPACES

BUDGET = 8
NOISE_SD = .03


def make_batch(namespace, seed, size=96, draws=4, query_kind='familiar', history_kind='mixture'):
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('positive integer noise draws required')
    source = original_batch(namespace, int(seed)+1_000_000_000, size, query_kind, history_kind)
    rng = np.random.default_rng([7000+NAMESPACES[namespace], int(seed), 21])
    gate_rng = np.random.default_rng([7000+NAMESPACES[namespace], int(seed), 22])
    return {'namespace': namespace, 'seed': int(seed), 'worlds': size, 'draws': draws,
            'histories': source['histories'], 'families': source['families'],
            'states': np.repeat(source['states'], draws, axis=0),
            'queries': np.repeat(source['queries'], draws, axis=0),
            'noise': NOISE_SD*rng.normal(size=(size*draws, BUDGET+1)),
            'random_gates': normalized_positive(gate_rng.normal(size=(size*draws,BUDGET,12))),
            'random_thresholds': .15*np.tanh(gate_rng.normal(size=(size*draws,BUDGET)))}

"""Independent, continuous history/measurement/target streams; no candidate list."""
import numpy as np

from memory_gate_soma import MemorySetup, decay_constants, uniform_gate


NAMESPACES = {'train': 100, 'validation': 200, 'test': 300, 'transfer': 400, 'demo': 500}
FAMILY_NAMES = ('white', 'autoregressive', 'impulses', 'sinusoid', 'switching')


def normalized_positive(logits):
    weights = np.exp(logits-np.max(logits, axis=-1, keepdims=True))
    return weights/np.linalg.norm(weights, axis=-1, keepdims=True)


def make_batch(namespace, seed, size=96, query_kind='familiar', history_kind='mixture'):
    if namespace not in NAMESPACES or query_kind not in ('familiar', 'novel'):
        raise ValueError('unknown split or query family')
    if history_kind not in ('mixture', 'switching') or not isinstance(size, int) or size < 1:
        raise ValueError('invalid history family or batch size')
    key = [NAMESPACES[namespace], int(seed)]
    stimulus_rng = np.random.default_rng(key+[1])
    sensor_rng = np.random.default_rng(key+[2])
    query_rng = np.random.default_rng(key+[3])
    gate_rng = np.random.default_rng(key+[4])
    shape = (size, 32)
    if history_kind == 'mixture':
        families = stimulus_rng.integers(0, 4, size=size)
        white = stimulus_rng.normal(size=shape)
        ar = np.zeros(shape)
        for t in range(32):
            ar[:, t] = (.85*ar[:, t-1] if t else 0)+stimulus_rng.normal(size=size)
        impulses = np.zeros(shape)
        positions = stimulus_rng.integers(0, 31, size=(size, 4))
        heights = stimulus_rng.normal(size=(size, 4))
        np.add.at(impulses, (np.arange(size)[:, None], positions), heights)
        frequency = stimulus_rng.uniform(.04, 1.2, size=(size, 1))
        phase = stimulus_rng.uniform(0, 2*np.pi, size=(size, 1))
        sine = np.sin(frequency*np.arange(32)+phase)+.05*stimulus_rng.normal(size=shape)
        histories = np.stack([white, ar, impulses, sine])[families, np.arange(size)]
    else:
        families = np.full(size, 4)
        histories = np.zeros(shape)
        for row in range(size):
            t, sign = 0, stimulus_rng.choice([-1., 1.])
            while t < 32:
                width = int(stimulus_rng.integers(3, 10))
                histories[row, t:t+width] = sign*stimulus_rng.uniform(.5, 1.)
                sign = -sign
                t += width
        histories += .1*stimulus_rng.normal(size=shape)
    setup = MemorySetup()
    decays = decay_constants(setup)
    filters = (1-decays[:, None])*decays[:, None]**(31-np.arange(32))[None, :]
    ambiguous = uniform_gate(setup) @ filters
    ambiguous[-1] = 0
    histories[:, -1] = 0
    histories -= (histories @ ambiguous)[:, None]*ambiguous/(ambiguous @ ambiguous)
    histories /= np.max(np.abs(histories), axis=1, keepdims=True)
    states = histories @ filters.T
    if query_kind == 'novel':
        queries = query_rng.normal(size=(size, 6, 12))
    else:
        count = query_rng.integers(2, 7, size=(size, 6, 1))
        count = np.where(query_rng.random(size=(size, 6, 1)) < .5, 1, count)
        ranks = np.argsort(np.argsort(query_rng.random(size=(size, 6, 12)), axis=-1), axis=-1)
        queries = (ranks < count).astype(float)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
    return {'namespace': namespace, 'seed': int(seed), 'histories': histories,
            'families': families, 'states': states, 'queries': queries,
            'noise': .03*sensor_rng.normal(size=(size, 4)),
            'random_gates': normalized_positive(gate_rng.normal(size=(size, 3, 12)))}

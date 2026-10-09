"""Abstract history traces and selected scalar readout, not a spiking neuron.

The observer knows a finite candidate dictionary. It never receives the true
branch state. Candidate propagation uses the issued gate and known read rule.
"""
from dataclasses import dataclass, asdict

import numpy as np

from lens_luotain_target import posterior_update


METHODS = ('active', 'random', 'fixed', 'open_loop', 'post_mix', 'erased')


@dataclass(frozen=True)
class MemorySetup:
    branches: int = 12
    history_steps: int = 32
    candidates: int = 8
    noise: float = 0.03
    read_damage: float = 0.25

    def __post_init__(self):
        if self.branches != 12 or self.history_steps != 32 or self.candidates != 8:
            raise ValueError('The first frozen gate uses 12 branches, 32 steps and 8 candidates')
        if not np.isfinite(self.noise) or self.noise <= 0:
            raise ValueError('noise must be finite and positive')
        if not np.isfinite(self.read_damage) or not 0 <= self.read_damage <= 1:
            raise ValueError('read_damage must be between zero and one')


def decay_constants(setup):
    return np.linspace(.35, .98, setup.branches)


def uniform_gate(setup):
    return np.ones(setup.branches) / np.sqrt(setup.branches)


def trace_history(history, decays):
    state = np.zeros(len(decays))
    for stimulus in history:
        state = decays * state + (1-decays) * stimulus
    return state


def candidate_histories(seed, setup):
    """Independent histories in a deliberately ambiguous initial-read family."""
    decays = decay_constants(setup)
    powers = setup.history_steps-1-np.arange(setup.history_steps)
    filters = (1-decays[:, None]) * decays[:, None]**powers[None, :]
    initial_read = uniform_gate(setup) @ filters
    initial_read[-1] = 0.  # constrain the current stimulus separately
    histories = np.random.default_rng(seed).normal(size=(setup.candidates, setup.history_steps))
    histories[:, -1] = 0.
    histories -= (histories @ initial_read)[:, None] * initial_read / (initial_read @ initial_read)
    histories /= np.max(np.abs(histories), axis=1)[:, None]
    states = np.array([trace_history(h, decays) for h in histories])
    return histories, states


def gate_catalog(setup):
    rng = np.random.default_rng(20261009)
    gates = [row for row in np.eye(setup.branches)]
    for _ in range(12):
        mask = np.zeros(setup.branches)
        count = int(rng.integers(2, 7))
        mask[rng.choice(setup.branches, count, replace=False)] = 1/np.sqrt(count)
        gates.append(mask)
    return np.array(gates)


def forecast_gates(setup):
    gates = np.random.default_rng(20271009).uniform(size=(8, setup.branches))
    return gates / np.linalg.norm(gates, axis=1)[:, None]


def choose_gate(posterior, predictions, available):
    """Only beliefs and predicted messages enter, never the hidden label."""
    available = np.asarray(available, dtype=int)
    if not len(available):
        raise ValueError('no unused gate remains')
    means = predictions @ posterior
    scores = ((predictions-means[:, None])**2) @ posterior
    return int(available[np.argmax(scores[available])])


def read_state(state, gate, eta):
    value = float(state @ gate)
    return value, state-eta*gate*value


def scene(seed, setup):
    histories, states = candidate_histories(seed, setup)
    gates = gate_catalog(setup)
    noise_rng = np.random.default_rng([int(seed), 91821])
    return {
        'seed': int(seed), 'histories': histories, 'states': states,
        'truth': int(np.random.default_rng([int(seed), 71239]).integers(setup.candidates)),
        'initial_noise': float(noise_rng.normal()),
        'noise': noise_rng.normal(size=(len(gates), len(gates))),
        'random_order': np.random.default_rng([int(seed), 61673]).permutation(len(gates)),
    }


def trial(seed, method, setup=None, budget=3, truth=None):
    setup = setup or MemorySetup()
    if method not in METHODS:
        raise ValueError('unknown policy')
    gates = gate_catalog(setup)
    if not isinstance(budget, int) or not 0 <= budget <= len(gates):
        raise ValueError('budget must be an integer between zero and catalogue size')
    world = scene(seed, setup)
    truth = world['truth'] if truth is None else int(truth)
    if not 0 <= truth < setup.candidates:
        raise ValueError('invalid truth index')
    original = world['states'].copy()
    candidates = np.zeros_like(original) if method == 'erased' else original.copy()
    state = candidates[truth].copy()
    original_true = original[truth].copy()
    uniform = uniform_gate(setup)
    prior = np.full(setup.candidates, 1/setup.candidates)
    initial_prediction = candidates @ uniform
    observations = [float(state @ uniform + setup.noise*world['initial_noise'])]
    posterior = posterior_update(prior, initial_prediction[:, None],
                                 np.array([observations[0]]), setup.noise)
    fixed_order = sorted(range(setup.branches), key=lambda i: (i % 4, i // 4))
    fixed_order += list(range(setup.branches, len(gates)))
    actions = []
    for step in range(budget):
        available = [i for i in range(len(gates)) if i not in actions]
        predictions = gates @ candidates.T
        if method in ('active', 'erased'):
            index = choose_gate(posterior, predictions, available)
        elif method == 'open_loop':
            index = choose_gate(prior, predictions, available)
        elif method == 'random':
            index = int(world['random_order'][step])
        else:
            index = fixed_order[step]
        gate = uniform if method == 'post_mix' else gates[index]
        prediction = candidates @ gate
        value, state = read_state(state, gate, setup.read_damage)
        observation = float(value+setup.noise*world['noise'][step, index])
        posterior = posterior_update(posterior, prediction[:, None],
                                     np.array([observation]), setup.noise)
        candidates -= setup.read_damage*np.outer(prediction, gate)
        actions.append(index)
        observations.append(observation)
    # Predict fresh gate answers of the ORIGINAL state, including erased runs.
    # Otherwise forgetting everything would make predicting zero look perfect.
    future = original @ forecast_gates(setup).T
    forecast = posterior @ future
    mse = float(np.mean((forecast-future[truth])**2))
    scale = float(np.mean(np.var(future, axis=0)))
    norm = max(float(np.linalg.norm(original_true)), 1e-12)
    return {
        'seed': int(seed), 'method': method, 'truth': truth,
        'actions': actions, 'observations': observations,
        'posterior': posterior.tolist(),
        'correct': bool(np.argmax(posterior) == truth),
        'logloss': float(-np.log(max(float(posterior[truth]), 1e-300))),
        'entropy': float(-np.sum(posterior[posterior>0]*np.log2(posterior[posterior>0]))),
        'forecast_mse': mse, 'forecast_scale': scale,
        'disturbance': float(np.linalg.norm(state-original_true)/norm),
        'initial_separation_over_noise': float(np.ptp(original @ uniform)/setup.noise),
        'soma_drift': float((state-original_true) @ uniform),
    }


def benchmark(seeds, setup=None, budget=3):
    setup = setup or MemorySetup()
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError('at least one seed is required')
    runs = {method: [trial(seed, method, setup, budget) for seed in seeds] for method in METHODS}
    summary = {}
    for method, rows in runs.items():
        summary[method] = {
            'correct': sum(r['correct'] for r in rows), 'total': len(rows),
            'accuracy': float(np.mean([r['correct'] for r in rows])),
            'mean_logloss': float(np.mean([r['logloss'] for r in rows])),
            'mean_entropy_bits': float(np.mean([r['entropy'] for r in rows])),
            'forecast_nrmse': float(np.sqrt(np.mean([r['forecast_mse'] for r in rows]) /
                                           max(np.mean([r['forecast_scale'] for r in rows]), 1e-12))),
            'mean_disturbance': float(np.mean([r['disturbance'] for r in rows])),
        }
    active, random, open_loop = [summary[m] for m in ('active','random','open_loop')]
    sequence_difference = float(np.mean([a['actions']!=o['actions']
                                        for a,o in zip(runs['active'],runs['open_loop'])]))
    g1 = max(r['initial_separation_over_noise'] for r in runs['active']) < 1e-6
    g2 = random['mean_logloss']-active['mean_logloss'] >= .10 and active['accuracy']-random['accuracy'] >= .05
    g3 = open_loop['mean_logloss']-active['mean_logloss'] >= .03 and sequence_difference >= .20
    g4 = all(.08 <= summary[m]['accuracy'] <= .18 and
             abs(summary[m]['mean_logloss']-np.log(8)) < 1e-8 for m in ('post_mix','erased'))
    g5 = (active['mean_disturbance'] == 0 if setup.read_damage == 0 else active['mean_disturbance'] > 0)
    return {
        'config': {'setup': asdict(setup), 'budget': budget, 'seeds': seeds,
                   'catalogue_size': len(gate_catalog(setup)), 'retained_branch_values': setup.branches,
                   'observer_candidate_state_values': setup.branches*setup.candidates,
                   'observer_belief_values': setup.candidates},
        'summary': summary,
        'comparisons': {
            'active_minus_random_accuracy': active['accuracy']-random['accuracy'],
            'random_minus_active_logloss': random['mean_logloss']-active['mean_logloss'],
            'open_loop_minus_active_logloss': open_loop['mean_logloss']-active['mean_logloss'],
            'active_open_loop_sequence_difference_fraction': sequence_difference,
        },
        'gates': {'G1_initial_ambiguity': bool(g1), 'G2_active_vs_random': bool(g2),
                  'G3_retained_context_vs_open_loop': bool(g3), 'G4_information_loss_controls': bool(g4),
                  'G5_read_disturbance': bool(g5)},
        'runs': runs,
    }

"""A reproducible active-query toy for computational periscopy.

This is a 1-D Lambertian shadow model, not the 2-D D11 camera experiment.
No ground-truth candidate is accessible to the sensor selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class Setup:
    patches: int = 16
    wall_pixels: int = 64
    subsources: int = 5
    distance: float = 1.03
    mask_depth: float = 0.55
    screen_halfwidth: float = 0.19
    wall_halfwidth: float = 0.45
    mask_halfwidth: float = 0.26
    mask_cells: int = 11
    noise: float = 0.03


def transport(setup: Setup, mask: np.ndarray | None = None, shift: float = 0.0) -> np.ndarray:
    """Integrate a 1-D patch's light at each wall pixel, blocking intersecting rays.

    D^2/r^4 Lambertian intensity, finite sub-sources for penumbra.  All
    exposures use the SAME reference calibration (unoccluded uniform screen).
    """
    if not (0 < setup.mask_depth < setup.distance):
        raise ValueError('mask depth must lie between screen and wall')
    if mask is not None and np.asarray(mask).shape != (setup.mask_cells,):
        raise ValueError('mask shape mismatch')
    width = 2 * setup.screen_halfwidth / setup.patches
    centers = np.linspace(-setup.screen_halfwidth + width / 2,
                          setup.screen_halfwidth - width / 2, setup.patches)
    offsets = (np.arange(setup.subsources) + 0.5) / setup.subsources - 0.5
    sources = centers[:, None] + width * offsets[None, :]
    wall = np.linspace(-setup.wall_halfwidth, setup.wall_halfwidth, setup.wall_pixels)
    delta = wall[:, None, None] - sources[None, :, :]
    r2 = setup.distance**2 + delta**2
    intensity = setup.distance**2 / r2**2
    if mask is not None:
        interception = sources[None, :, :] + (setup.mask_depth / setup.distance) * delta
        fraction = (interception - shift + setup.mask_halfwidth) / (2 * setup.mask_halfwidth)
        indices = np.floor(fraction * setup.mask_cells).astype(int)
        inside = (indices >= 0) & (indices < setup.mask_cells)
        blocked = inside & np.asarray(mask, dtype=bool)[np.clip(indices, 0, setup.mask_cells - 1)]
        intensity = intensity * (~blocked)
    matrix = intensity.mean(axis=2)
    # Fixed readout gain chosen using the no-mask response to a uniform screen.
    calibration = _calibration(setup, wall, sources)
    return matrix / calibration[:, None]


def _calibration(setup: Setup, wall: np.ndarray, sources: np.ndarray) -> np.ndarray:
    r2 = setup.distance**2 + (wall[:, None, None] - sources[None, :, :])**2
    return (setup.distance**2 / r2**2).mean(axis=2).sum(axis=1)


def catalog(setup: Setup, seed: int = 20261009, count: int = 30):
    """Predetermined mask+shift catalogue, independent of test scenes."""
    rng = np.random.default_rng(seed)
    options = []
    seen = set()
    while len(options) < count:
        cells = rng.integers(0, 2, size=setup.mask_cells, dtype=np.int64)
        if not (3 <= cells.sum() <= setup.mask_cells - 3):
            continue
        shift = float(rng.choice([-0.06, -0.03, 0.0, 0.03, 0.06]))
        key = (tuple(cells.tolist()), shift)
        if key in seen:
            continue
        seen.add(key)
        options.append((cells.astype(bool), shift))
    return options


@lru_cache(maxsize=16)
def transport_catalog(setup: Setup):
    a0 = transport(setup)
    ops = catalog(setup)
    masked = np.stack([transport(setup, mask, shift) for mask, shift in ops])
    return a0, masked


def twins(unmasked: np.ndarray, seed: int, n: int = 8):
    """Candidates deliberately constructed from the least observed directions.

    This is a synthetic ambiguity stress test, not a natural-image benchmark.
    The candidate set is supplied to every strategy; the true index is hidden.
    """
    rng = np.random.default_rng(seed)
    _, _, vh = np.linalg.svd(unmasked, full_matrices=False)
    basis = vh[-4:]
    # Rotate the difficult-to-see subspace and choose a different collection
    # of sign combinations for each seed.  No true-label information is used.
    patterns = np.array([[1.0 if (i >> bit) & 1 else -1.0 for bit in range(4)]
                         for i in range(16)])
    selected = patterns[rng.choice(len(patterns), size=n, replace=False)]
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    candidates = selected @ rotation @ basis
    candidates /= np.max(np.abs(candidates), axis=1)[:, None]
    return 0.5 + 0.4 * candidates


def posterior_update(p: np.ndarray, prediction: np.ndarray, observation: np.ndarray, sigma: float):
    """Exact categorical Bayes update under iid Gaussian camera noise."""
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    logp = np.log(np.maximum(p, np.finfo(float).tiny))
    logp -= 0.5 * np.sum(((prediction - observation[None, :]) / sigma)**2, axis=1)
    logp -= logp.max()
    new = np.exp(logp)
    return new / new.sum()


def choose_probe(p: np.ndarray, predicted: np.ndarray, available: np.ndarray):
    """Maximize expected pairwise Gaussian KL (variance / sigma^2).

    predicted shape: (n_probes, n_hypotheses, n_wall).  Equal sigma cancels
    from the argmax.  Does not see observations, true index or future noise.
    """
    means = np.einsum('h,phw->pw', p, predicted)
    scores = np.einsum('h,phw->p', p, (predicted - means[:, None, :])**2)
    return int(available[np.argmax(scores[available])])


def entropy(p: np.ndarray) -> float:
    positive = p[p > 0]
    return float(-(positive * np.log2(positive)).sum())


def trial(seed: int, setup: Setup | None = None, budget: int = 3, method: str = 'active'):
    """One independent trial, with common potential noise across methods."""
    if method not in {'active', 'random', 'repeat'}:
        raise ValueError('unknown method')
    if budget < 0:
        raise ValueError('negative budget')
    setup = setup or Setup()
    rng = np.random.default_rng(seed)
    a0, matrices = transport_catalog(setup)
    images = twins(a0, seed)
    truth = int(rng.integers(len(images)))
    candidates = catalog(setup)
    estimates = np.einsum('pws,hs->phw', matrices, images)
    start = images @ a0.T
    # One common no-mask initial observation. Each mask has pre-generated
    # potential noise per step, shared across policies, independent of choice.
    initial_observation = start[truth] + rng.normal(0, setup.noise, setup.wall_pixels)
    potential_noise = rng.normal(0, setup.noise, (budget, len(candidates), setup.wall_pixels))
    repeat_noise = rng.normal(0, setup.noise, (budget, setup.wall_pixels))
    random_order = rng.permutation(len(candidates))
    p = posterior_update(np.full(len(images), 1 / len(images)), start, initial_observation, setup.noise)
    prior_p = p.copy()
    actions = []
    for step in range(budget):
        if method == 'repeat':
            prediction = start
            observation = start[truth] + repeat_noise[step]
            actions.append('unmasked')
        else:
            available = np.array([j for j in range(len(candidates)) if j not in actions], dtype=int)
            if len(available) == 0:
                raise ValueError('budget exceeds catalogue')
            index = choose_probe(p, estimates, available) if method == 'active' else int(random_order[step])
            prediction = estimates[index]
            observation = prediction[truth] + potential_noise[step, index]
            actions.append(index)
        p = posterior_update(p, prediction, observation, setup.noise)
    return dict(seed=seed, truth=truth, method=method, actions=actions,
                correct=bool(np.argmax(p) == truth), truth_probability=float(p[truth]),
                logloss=float(-np.log(np.maximum(p[truth], 1e-300))),
                entropy=entropy(p), initial_entropy=entropy(prior_p),
                initial_truth_probability=float(prior_p[truth]),
                initial_ambiguity=float(np.max(np.linalg.norm(start[:, None] - start[None], axis=2)) /
                                        (setup.noise * np.sqrt(setup.wall_pixels))),
                final_posterior=p.tolist())


def benchmark(seeds=range(4100, 4164), budget=3, setup=None):
    setup = setup or Setup()
    runs = {method: [trial(int(seed), setup, budget, method) for seed in seeds]
            for method in ['active', 'random', 'repeat']}
    summary = {}
    for method, items in runs.items():
        summary[method] = {
            'accuracy': float(np.mean([r['correct'] for r in items])),
            'mean_logloss': float(np.mean([r['logloss'] for r in items])),
            'median_truth_probability': float(np.median([r['truth_probability'] for r in items])),
            'mean_entropy': float(np.mean([r['entropy'] for r in items]))}
    return {'config': {'seeds': [int(s) for s in seeds], 'budget': budget,
                       'setup': vars(setup), 'catalog_size': 30,
                       'hypotheses': 8}, 'summary': summary, 'runs': runs}

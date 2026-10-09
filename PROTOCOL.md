# Frozen held-out gate — 2026-10-09

## Goal

Use an independently generated wall measurement to distinguish visually confusable hidden images. The observer knows eight candidate hidden-screen patterns and the forward model, but not which candidate generated the observations. Choose occluder patterns from a predetermined catalogue without peeking at the true image. Compare three measurements after the same no-mask initial photograph: actively chosen masks, uniformly random masks, and repeats of the unmasked view. Equal measurement counts; same Gaussian noise distribution and same candidate scenes for all strategies.

This is **not** a claim of general image reconstruction. Candidate identification is easier than unconstrained inversion. The simulator uses 1-D screen/wall geometry, fixed mask depth, fixed calibrated light transport, Gaussian noise independent of brightness and no model mismatch; it is not a replication of the 2019 Nature camera measurements.

## Design and separation

- Geometry: `Setup()` in source, 16 hidden screen patches, 64 wall pixels, 5 point-sources per patch, D=1.03 m, mask at y=0.55 m, mask cells=11, noise sigma=0.03 in units of no-mask wall irradiance.
- Forward model: Lambertian D²/r⁴ with ray-occlusion in a binary mask. 30 predetermined mask/offset probes (catalog seed 20261009), available to both active and random policies. No camera-position or light-exposure tuning after evaluation.
- Each seed produces eight hypotheses with seed-varying rotations and sign combinations of the last four right singular vectors of its no-mask transport matrix. These hypotheses have nearly identical initial unmasked observations by construction. Their true label is drawn uniformly; no label is available to the policy.
- Both policies start from exactly the same noisy unmasked measurement; subsequent noise is drawn from the same independent distribution, keyed by seed/step/mask to enable paired comparisons. Random chooses a no-replacement ordering of masks. The repeat policy only retakes the initial unmasked configuration.
- Active chooses the mask that maximizes the current posterior-weighted variance of predicted wall images, the expected pairwise KL of equal-variance Gaussian observation models up to a constant factor. This choice sees *only candidate predictions and the posterior*, never the hidden label or future noise. All strategies update the categorical posterior using exact Gaussian likelihoods.
- **Exploratory tuning** was carried out on seeds 4100..4163 over several noise levels (0.008 to 0.2) and budgets (1 to 3). Low noise saturated both coded policies. The frozen held-out choice is sigma=0.03, a budget of **two additional views**, 64 fresh corrected seeds **6100..6163**. No result from the corrected seeds was inspected when choosing it. A first check at seeds 5100..5163 (90.6% / 75.0% / 20.3% accuracy) was **reclassified as a pilot** after discovering that the intended seed-specific sign patterns had been overwritten by a fixed array. That check varied noise and true labels but effectively reused one candidate family. We corrected the generator *before* evaluating the independent 6100-series seeds and retained the same thresholds. The old pilot is not counted as the corrected final gate.

## Gates (set before opening the held-out run)

- G1: initial no-mask image differences are much smaller than typical noise (median pair-separation / per-pixel noise norm < 0.05).
- G2: active mean negative log probability of the true candidate is **at least 0.10 nats lower** than random after two additional masks, and active classification accuracy exceeds random by **at least 0.05** on these 64 seeds.
- G3: each coded policy beats unmasked repeat in accuracy by at least 0.20. Negative results are retained; the small sample gives noisy accuracy estimates.
- G4: all core deterministic and mathematical tests pass; the result is reproducible from published source and seed list.

The held-out run is not proof of novelty. Tests on natural images, real camera input, depth/occluder miscalibration, photon noise, fixed computation budget, and real-world geometry remain future experiments. We report runtime-independent scalar model costs, not hardware speedup. Selection evaluates all 30 possible mask predictions at each step, while random selection does not—this computation is a real cost.

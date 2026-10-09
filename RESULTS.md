# LensLuotainTarget — outcome ledger (2026-10-09)

## Frozen corrected test

Defined in [PROTOCOL.md](PROTOCOL.md). Noise sigma = 0.03; 64 independent new truth/noise/rotated-candidate seeds 6100..6163; eight finite candidate images; 30 preselected mask/position probes; each strategy starts with the same noisy no-mask observation and takes two more images. The active strategy maximizes current posterior-weighted predicted image variance. Random uses a fixed-per-seed permutation of the same mask list. Repeat photographs without a mask. Forward model is known and correct by construction.

| Metric | Active | Random | Repeat |
|---|---:|---:|---:|
| Top-1 accuracy | 54/64 = 84.375% | 48/64 = 75% | 7/64 = 10.9375% |
| Mean true-label negative log probability (nats) | 0.293596 | 0.641515 | 2.079442 |
| Median true-label probability | 0.909297 | 0.682568 | 0.125 |
| Mean final entropy (bits) | 0.626940 | 1.018179 | 3.000000 |

Active improves top-1 accuracy over random by 9.375 percentage points, and true-label log loss by 0.347919 nats (lower is better). Median maximum pairwise no-mask wall-response separation divided by the noise norm = 3.08e-15, so the first view conveys no label information at this precision.

## Frozen gates

- G1 (initial ambiguities below noise): PASS. Median 3.08e-15 < 0.05.
- G2 (active log loss at least 0.10 below random, and accuracy at least 0.05 higher): PASS. 0.3479 nats and 0.09375 accuracy.
- G3 (coded policies at least 0.20 accuracy better than no-mask repeats): PASS. Differences 0.734375 and 0.640625.
- G4 (tests and reproduction): PASS locally on Python + NumPy; see source/tests and rerun command.

These are 64 trials in a deliberately constructed finite hypothesis family. No statistical significance claim was predeclared, and the policies do **not** use equal computation: active evaluates a 30-mask catalogue each step. Noise is fixed-variance Gaussian, independent of blocked light (not photon-noise accurate). Real cameras require calibration of geometry, gain, ambient light and occluder shape.

## The failed-first-implementation lesson

Initial exploratory work on seeds 4100..4163 revealed that sigma=0.008 and two additional measurements made both active and random reach 100% top-1 accuracy. That setting offered little discrimination between policies; the harder sigma=0.03/two-view setting was chosen on exploratory seeds and frozen for held-out analysis.

The first 5100..5163 evaluation showed 90.6% active vs 75.0% random vs 20.3% repeat. A subsequent code review caught a generator defect: intended random sign patterns had been overwritten with a deterministic eight-pattern array. Different seeds thus shared effectively one candidate family, while varying true label/noise. That run is a **pilot, not an independent scene-generalization result**. Before running the 6100-series, we fixed the candidate generator to randomly select eight sign patterns and rotate the four-dimensional weakly observed subspace for each seed. The 6100-series gate reused the frozen thresholds; no post-run adjustment was made.

## Why it's interesting, and why it is not a discovery of inverse optics

An occluder changes the measurement operator, so rival explanations that were invisible under one operator can become distinguishable. This is a standard coded-aperture/active-sensing principle. The concrete result is that an online observer, with the hypothesis dictionary and geometry but **without the truth**, chose more informative operators than random sampling in this controlled experiment.

This does **not** establish general recovery of unseen image content or robustness to wrong forward models; Varjoluotain's own real-photo occluder-location failures are a reminder that model mismatch matters. The next experiment should preserve a fixed camera exposure, introduce position/background uncertainty, and test candidate sets the selector has never been calibrated on, ideally with actual photographs.

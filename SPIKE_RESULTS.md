# Learned binary querying — frozen outcome ledger

The learned adaptive binary interface passed its predeclared gates on 2026-10-09:
**36.2% lower answer MSE than a trained fixed schedule**, and **36.5% lower MSE
on withheld switching histories**. All three initialization seeds improved.
The stronger joint gate-and-threshold benefit **was not earned**: adaptive branch
selection adds only **0.44%** over fixed gates with adaptive thresholds, below the
5% criterion, with an interval crossing zero.

[Run all 24 actual trained receivers](https://anttiluode.github.io/LensLuotainTarget/site/spikes.html).
The [frozen protocol](SPIKE_PROTOCOL.md), [numerical receipt](results/learned_spikes.json),
[all selected weights and validation traces](results/spike_weights.json) and
[inference verification](results/spike_verification.json) preserve the result.

## What was frozen before testing

The protocol was committed locally before training. All eight controls received
4,000 Adam updates for each of three initializations. Every checkpoint was
selected by minimum full loss on validation data. All 24 checkpoints were
committed and sealed before the held-out evaluation. Publication collects this
history; no earlier public preregistration is claimed. No seeds, thresholds,
budgets or success criteria were retuned after seeing these outcomes.

There are 512 independent histories per condition. Each has eight independent
sensor-noise trajectories and is evaluated by all three models of each approach.
Noise and initializations are averaged within a history before paired bootstrap
resampling of whole histories (2,048 resamples). Intervals are conditional on
these three trained models, not a substitute for more independent training runs.

The receiver has no history dictionary, candidate posterior or exact branch
state. Training does have supervised targets and a known differentiable plant.
It learns from actual hard bits using exact probit likelihood-score derivatives
and backpropagation through command-dependent memory movement. The independent
leave-one-trajectory-out baseline reduces variance without differentiating
through the bit. Enumerated short bit trees agree with numerical expected-loss
derivatives; no straight-through sigmoid is substituted for binary inference.

## All aggregate outcomes

NRMSE is sqrt(MSE / zero-predictor MSE); 1 matches predicting zero.
Full J is MSE/0.1² + 0.1 × squared original-state displacement/0.1² + 0.001 × 8.
The read-count penalty is constant. Graded references transmit scalar answers
and are not equal-bit competitors. Their better absolute accuracy is preserved.

### Fresh histories, familiar questions

| Receiver | Answer NRMSE | Full J | Mean squared displacement |
|---|---:|---:|---:|
| Adaptive gate + threshold | 0.3849 | 0.05305 | 0.0008136 |
| Trained fixed schedule | 0.4813 | 0.07397 | 0.0008236 |
| Adaptive gates / fixed thresholds | 0.4211 | 0.06036 | 0.0008169 |
| Fixed gates / adaptive thresholds | 0.3871 | 0.05345 | 0.0008105 |
| Adaptive gates / threshold zero | 0.5256 | 0.08131 | 0.0004461 |
| Random commands | 0.7924 | 0.17145 | 0.0006969 |
| Graded adaptive reference | 0.2653 | 0.03157 | 0.0006028 |
| Graded fixed reference | 0.2817 | 0.03424 | 0.0006457 |

### Main: fresh histories, new signed questions

| Receiver | Answer NRMSE | Full J | Mean squared displacement |
|---|---:|---:|---:|
| Adaptive gate + threshold | 0.3856 | 0.05901 | 0.0008136 |
| Trained fixed schedule | 0.4829 | 0.08346 | 0.0008236 |
| Adaptive gates / fixed thresholds | 0.4201 | 0.06705 | 0.0008169 |
| Fixed gates / adaptive thresholds | 0.3865 | 0.05917 | 0.0008105 |
| Adaptive gates / threshold zero | 0.5282 | 0.09292 | 0.0004461 |
| Random commands | 0.7875 | 0.19376 | 0.0006969 |
| Graded adaptive reference | 0.2626 | 0.03391 | 0.0006028 |
| Graded fixed reference | 0.2795 | 0.03698 | 0.0006457 |

### Transfer: withheld switching histories, new signed questions

| Receiver | Answer NRMSE | Full J | Mean squared displacement |
|---|---:|---:|---:|
| Adaptive gate + threshold | 0.3865 | 0.07980 | 0.0012034 |
| Trained fixed schedule | 0.4848 | 0.11418 | 0.0012107 |
| Adaptive gates / fixed thresholds | 0.4285 | 0.09360 | 0.0012104 |
| Fixed gates / adaptive thresholds | 0.3817 | 0.07792 | 0.0011609 |
| Adaptive gates / threshold zero | 0.4954 | 0.11352 | 0.0007314 |
| Random commands | 0.7588 | 0.24796 | 0.0009540 |
| Graded adaptive reference | 0.2677 | 0.04574 | 0.0009051 |
| Graded fixed reference | 0.2764 | 0.04813 | 0.0009563 |

## Frozen criteria, including the failure

| Gate | Outcome | Evidence |
|---|---|---|
| S1: integrity, numerical gradients, observation-only receiver and actual-weight browser parity | PASS | Complete suite passed without skips; separate verification record pins the code, tests and receipt. |
| S2: adaptive beyond trained fixed schedule | PASS | Main MSE gain 36.2%; 3/3 wins; positive paired difference interval; lower full J. |
| S3: threshold contribution beyond adaptive gates / fixed thresholds | PASS | Main MSE gain 15.7%; 3/3 wins; positive paired interval. |
| S4: branch contribution beyond fixed gates / adaptive thresholds | **FAIL** | Main MSE gain 0.44%; below 5%; interval crosses zero; 2/3 wins. |
| S5: unseen-history transfer beyond trained fixed | PASS | MSE gain 36.5%; 3/3 wins; positive paired interval; lower full J. |
| S6: erasure and common-sum controls lose recoverability | PASS | MSE / zero-predictor MSE exceeds 0.90 in both controls on main and transfer. |

The main absolute MSE difference between fixed and adaptive is 0.000243564,
95% paired interval [0.000205623, 0.000292945]. Transfer difference is
0.000343016, interval [0.000308984, 0.000378752]. Adaptive versus gate_only is
0.0000801132, interval [0.0000550316, 0.000110445]. Adaptive versus threshold_only
is 0.00000191455, interval [-0.0000128747, 0.0000159265]. These are absolute MSE
intervals, not relative-gain confidence intervals.

The scoped **learned adaptive binary-interface** claim is earned. The stronger
**joint adaptive gate-and-threshold** claim fails. The measurements point to the
threshold as the useful adaptive control in this task. They do not prove that
branch adaptation never helps, nor that the learned threshold implements exact
Bayesian bisection. Fixed gates with learned adaptive thresholds nearly match the
fully adaptive model, so a simpler interface deserves the next replication.

## Controls and limits

Every control has the same 16-state recurrent receiver and internal nonlinear
gate-and-threshold head, even when the physical commands are fixed. This removes
the simpler comparison's extra-computation advantage. All have 1,252 trainable
parameters, 7,376 counted neural MACs, and 28 sender-plus-receiver persistent
values. Equal counts are not equal effective capacity: actual/context commands
coincide in the fully adaptive receiver, while the fixed control can use distinct
inputs. Adaptive outbound commands cost thirteen coefficients per read; a fixed
schedule can be shared in advance. Nonlinear functions, normalization, plant and
optimizer work are excluded from MAC counts. No energy or speed win is claimed.

Branch erasure and forcing every read to the original common uniform sum both
remove recoverability of the original targets. These are inference ablations,
not separately retrained competitors. Their zero-predictor-normalized MSE is
1.067/1.227 on main and 1.047/1.173 on transfer (erased/common-sum).

The history prior remains structured, correlated and non-Gaussian: leaky branch
filters, common-sum projection and per-history normalization substantially
constrain it. Eight bits do not reconstruct an arbitrary twelve-dimensional
state. New final questions are new linear vectors in a designed decoder interface,
not new semantics or a changing future environment. One withheld history family
is a narrow transfer test. The Gaussian covariance theorem for graded, exact
Bayesian estimation is not contradicted by these non-Gaussian, trained receivers
and the state-displacement penalty.

The physical write at eta=0.25 is invertible. It moves the retained state and
changes later answers but does not erase a mathematical dimension. The threshold
is externally controllable by assumption. There are no membrane dynamics, spike
waveforms, timing, refractory periods or biological measurements. This is not a
new neuron theory or demonstrated practical AI architecture advantage.

## The thread this preserves

[Kuulustelu](https://github.com/anttiluode/Kuulustelu) supplied the threshold
comparison; that repository was not changed. The
[original ecg.json geometric-neuron accident](https://github.com/anttiluode/GeometricNeuronOriginReview)
placed observation geometry inside a feedback loop.
[AnttisBrain2's moons](https://github.com/anttiluode/AnttisBrain2),
[MovingTarget2](https://github.com/anttiluode/MovingTarget2),
[Varjoluotain's masks](https://github.com/anttiluode/Varjoluotain) and
[BrainAsInverseModeler](https://github.com/anttiluode/BrainAsInverseModeler) sharpened
which retained differences can reach another observer. Here a learned threshold
changes the question asked of retained memory. The mechanisms have different
equations; their common question is experimentally useful without making them
the same physical system.

## Reproduce without retraining

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_spike_receipt.py
python make_spike_site_data.py --verify
```

NumPy and SciPy run the reference; Node runs actual-weight browser parity.
`spike_experiment.py verify` reproduces the numerical receipt from the sealed
weights. Browser integrity is sealed separately after the full suite, and a stale
receipt or code/test fingerprint cannot be presented as S1 passed. The earlier
optical, dictionary and graded experiment sources and scientific receipts remain
unchanged. CI reproduces all receipts without retraining.

The browser runs all three selected initializations for all eight models on
12 independent demonstration worlds (seeds 82000/82001). These examples are not
the held-out test. It enforces eight additional reads and exposes new final
questions only after the budget. Changing a final question reuses the same
acquired receiver state.

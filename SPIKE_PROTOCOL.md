# Learned gates and spike thresholds — frozen protocol

Frozen 2026-10-09 before training or held-out evaluation. This extends the
dictionary-free experiment without changing its source, checkpoints or receipt.
Kuulustelu is a read-only motivation, not an implementation dependency.

## Question and scope

Can a receiver learn to choose useful branch gates **and** spike thresholds from
previous one-bit replies, on continuous histories and future questions that were
not supplied to it? Does physical adaptation help beyond the same computation
used internally by a receiver with a jointly trained fixed query schedule?

The 12 leaky branches retain a 32-step stimulus history. The generator from
`learned_data.py` is reused unchanged with an additional seed offset of
1,000,000,000. Its projection makes the initial uniform sum zero; per-history
normalization makes this a non-Gaussian, correlated prior. It is not a test of
the exact Gaussian covariance theorem. Four training history families are
continuous; the switching family is withheld. No candidate-history catalogue,
posterior particles, exact branch state or future question enters the receiver.
Simulation states, labels and a differentiable known forward model are available
to the trainer. Generalization to other branch dynamics is not tested.

## Physical channel and loss

There are eight additional reads after a common initial uniform read. Initial
read is nondestructive, threshold zero. All later gates are positive and unit
length; repeated directions are allowed. Gaussian sensor noise has SD 0.03.

```
bit_t = 1[g_t · m_t + noise_t > theta_t]
m_(t+1) = m_t - 0.25 g_t (g_t · m_t)
theta_t = 0.15 tanh(threshold_logit_t)
```

The receiver sees signed bits (-1,+1), its actual commands and its own recurrent
context. The threshold changes the observation, not the write. The write uses
the noiseless projection. This invertible update moves memory; it does not
destroy a mathematical dimension. The target is the **original** state, scored
on six unit future question vectors revealed only after the reads.

`J = mean future squared error / 0.1² + 0.1 ||m_final-m_original||² / 0.1² + 0.001*8`.

Read cost is constant at this budget. It does not establish an optimal budget.
Graded references return the noisy scalar, with the same write and read count.
They communicate more than one bit per read and are accuracy references, not
equal-bit competitors.

## Matched architecture and controls

Every model has hidden size 16 and the same parameters, internal context gate
and threshold head. Each recurrent update consumes the actual 13-value command,
the internal 13-value command, its previous hidden state and the reply. The
context command is computed even when it is not applied to the plant. Fixed
controls therefore retain the extra computation that confounded a simpler
comparison. The final decoder predicts 12 branch values; questions are applied
externally. Initial conditions are shared within each initialization.

| Model | Physical gate | Physical threshold | Reply |
|---|---|---|---|
| adaptive | context head | context head | one bit |
| fixed | jointly trained step schedule | jointly trained step schedule | one bit |
| gate_only | context head | trained step schedule | one bit |
| threshold_only | trained step schedule | context head | one bit |
| zero_threshold | context head | zero | one bit |
| random | independent positive direction | independent bounded threshold | one bit |
| graded_adaptive | context head | no effect on scalar reading | scalar |
| graded_fixed | trained step schedule | no effect on scalar reading | scalar |

All models: 1,252 trainable parameters, 7,376 counted neural multiply-accumulates
per episode, 28 persistent state values (12 sender, 16 receiver). Count excludes
nonlinear functions, normalization, plant work and optimizer. Outbound adaptive
commands contain 12 gate coefficients plus a threshold per read; fixed commands
can be stored in advance. No energy, speed or total communication advantage is
claimed. Identical parameter counts do not guarantee identical effective
capacity: the adaptive actual/context inputs coincide.

## Training and sealed evaluation

Three initialization seeds: **20261012, 20261013, 20261014**. All eight models
receive **4,000 Adam updates**, learning rate .003, betas .9/.999, epsilon 1e-8,
global gradient norm clipped at 1. Each batch has 96 independent histories and
four independent sensor trajectories per history. Training uses fresh batch
seed `initialization_seed*10000 + update`.

For binary replies, train with the exact probit likelihood-score gradient plus
direct differentiation of the realized trajectory. Do not differentiate through
the hard bit. Subtract the other three trajectories' mean loss as a detached,
independent baseline. Stable Gaussian log-CDF computations avoid clipping the
likelihood. Short, enumerated bit trees must agree with numerical derivatives
of expected loss, including the write's effect on later readings. Graded
references use ordinary pathwise gradients.

Select minimum full J on a fixed validation set: seed 11000, 512 histories,
four sensor trajectories each, familiar questions. Checkpoint at step zero
and every 100 updates. No test-based checkpoint, early stopping or retuning.
Seal all 24 selected models and complete traces, with source fingerprints,
before any held-out evaluation.

Held-out sets: 512 histories each, eight independent sensor trajectories each.
Test seed **62000** (familiar questions and novel dense signed questions on the
same worlds); transfer seed **72000** (withheld switching histories and novel
questions). Demo seeds 82000/82001 are separate. Average errors over the eight
noise draws and three models within each world. Paired 95% bootstrap intervals
use 2,048 resamples of whole histories, seed 20261012. These are 512 independent
histories, not 4,096 independent histories or replications of the architecture.

## Predeclared gates

Main condition means fresh mixture histories with novel signed questions.
Relative gain is `(baseline MSE - adaptive MSE)/baseline MSE`.

1. **S1 / integrity:** observation-only receiver, binary replies, gate and
   threshold bounds, budget, exact expected-gradient checks, resource equality,
   sealed-source/weight integrity and browser/Python inference agreement pass.
2. **S2 / physical adaptation:** main adaptive versus head-matched fixed MSE
   gain >=5%, paired interval's difference lower bound >0, at least two of
   three initialization wins, and lower mean full J.
3. **S3 / threshold contribution:** main adaptive versus gate_only MSE gain
   >=5%, positive paired lower bound and at least two initialization wins.
4. **S4 / branch contribution:** main adaptive versus threshold_only MSE gain
   >=5%, positive paired lower bound and at least two initialization wins.
5. **S5 / transfer:** adaptive versus fixed on switching histories and novel
   questions meets the S2 criteria.
6. **S6 / hidden-memory controls:** erase memory, or force all reads to the
   common uniform sum, retaining original targets. The trained adaptive model
   must have MSE / zero-predictor MSE >=0.90 on main and transfer in both controls.

S1+S2+S5+S6 earns the scoped claim **learned adaptive binary interface**.
The stronger **joint adaptive gate-and-threshold benefit** additionally requires
S3+S4. Report every gate separately, including failures. Random and zero-threshold
controls, familiar questions, graded accuracy and read displacement are reported
without additional success criteria. Do not upgrade criteria after the run.

## Interpretation boundary

Kuulustelu motivates threshold adaptation under a restricted query family.
This experiment tests learned commands without its supplied posterior or gate
catalogue, while still supplying simulation dynamics to training. A learned
threshold is assumed physically controllable. No spike timing, membrane voltage,
refractory period, biological soma, new neuron theory or practical AI superiority
is established by these results. The original geometric-neuron ECG accident,
MovingTarget attention and Varjoluotain masks motivate changing the measurement
operator; they are not identical physical mechanisms.

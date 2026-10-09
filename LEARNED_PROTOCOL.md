# Learning to ask without a history dictionary

Frozen on 2026-10-09 before training or opening the held-out sets. This extends the existing memory-to-gate-to-soma experiment; it does not revise its receipt.

## Question and boundary

Can a trained receiver choose three useful scalar reads of retained branch memory, then answer new linear questions about the original memory, without a supplied catalogue of possible histories?

The receiver sees one common initial noisy sum, its own issued gate coefficients and three noisy answers. It never receives stimulus histories, true branch values, a history label, candidate predictions, future questions while choosing gates, or the forward/write equations as inference inputs. Training uses supervised future-answer labels and gradients through the exact simulator. That simulator knowledge is a real training advantage, not a discovery of physical laws.

The sender retains twelve linear traces, with the original decays 0.35..0.98 and 32 stimulus steps. Histories end at input zero and are projected to have the same initial uniform soma sum, then bounded to [-1,1]. This deliberately imposed ambiguity and the learned training distribution remain priors. Between instantaneous reads, traces are stationary.

Every issued gate is nonnegative and has Euclidean norm one. The plant reads y=g·m+epsilon, epsilon~N(0,0.03²), then writes m'=m-0.25*g*(g·m). The initial sum does not write. The requested future answers are q·m_original, not the easily predicted erased state or future environmental dynamics. Read disturbance at eta=0.25 is invertible; the erasure control is genuinely irreversible.

## Models and resource comparison

All trainable models use a tanh recurrent receiver and a linear 12-coordinate prediction head. Future queries are applied to that predicted vector only after the reads; this linear-query interface is built in.

| Model | Receiver state | Gate rule | Trainable parameters | Neural multiply-accumulates per episode |
|---|---:|---|---:|---:|
| Learned adaptive | 16 | learned function of retained receiver state and read index | 944 | 2176 |
| Optimized fixed | 16 | three jointly trained gates independent of answers | 752 | 1600 |
| Random | 16 | shared, independent random normalized-positive gates | 716 | 1600 |
| Matched recurrent | 19 | three jointly trained fixed gates | 941 | 2071 |

Counts include initial encoding, recurrent updates, prediction head and adaptive gate head. Sender trace/filter/read/write work is shared, and softmax, normalization, tanh and optimizer operations are reported separately rather than hidden in these MAC counts. Persistent sender+receiver values are 28 for adaptive and 31 for matched recurrent. This is a comparable-resource, specific tanh recurrent baseline, not a comparison against all RNNs, GRUs or transformers.

A continuous adaptive gate commands twelve coefficients per read in addition to receiving one scalar. That outbound communication is larger than a shared fixed schedule. No communication-efficiency or energy advantage is claimed.

## Training, validation and freezing

- NumPy float64; no GPU or additional Python dependency.
- Four training history families: white noise, AR(0.85), four sparse impulses, and a sinusoid with small white noise. Family identity is not an input.
- Three independent training seeds: 20261009, 20261010, 20261011. Every model gets the same training batches, noise and supervised query targets within a seed; initialization streams are separate from data streams.
- Exactly 2000 Adam updates per model, batch size 96, learning rate 0.003, beta=(0.9,0.999), epsilon=1e-8, global gradient norm clip 1.0. Initial recurrent matrix is orthogonal with gain 0.7; gate index logits have standard deviation 1.0. No post-test hyperparameter search.
- Six future questions per training episode: unit branches or normalized positive subsets of 2..6 branches. Validation uses 512 separately seeded histories and questions of these familiar kinds.
- Select the lowest validation objective checkpoint among initialization and every 100 updates, breaking ties toward the earlier step. Do not select seeds or models using held-out performance.
- Separate RNG namespaces for train, validation, test and browser examples; stimulus, sensor noise, random gate and future-question streams are independent.
- Named batch seeds: validation 10000, familiar-family test 61000, switching-family test 71000, demonstrations 81000/81001. Unit-test fixtures use other seeds or batch sizes and are not the 512-episode evaluation sets.
- The loss is mean squared future-answer error / 0.1² + 0.1 * squared final displacement norm / 0.1² + 0.001 * number of new reads. There are exactly three new reads, so the query-count penalty is constant. This experiment learns what to ask, not when to stop.

## Held-out evaluation

512 episodes per condition, with all model seeds and methods sharing each episode's history, noise and queries. No gradient update or checkpoint selection is permitted during evaluation.

1. Fresh histories from the four familiar families; fresh familiar question vectors.
2. The same fresh histories; new dense signed normalized question vectors. This is the main unseen-question condition.
3. Unseen switching-block histories; new dense signed questions. This tests transfer beyond the trained history families.

Report future-answer NRMSE relative to the zero predictor, the full objective, mean relative branch-state displacement, and each training seed separately. Paired improvements use 2048 deterministic bootstrap resamples of the 512 per-episode errors averaged across the three training seeds. These intervals describe episode variation conditional on three trained models, not broad training-seed uncertainty.

Also evaluate the trained adaptive receiver with selection context zeroed, with branch memory erased, and with gates applied only to the already mixed uniform sum. Erasure still scores against the original history. These are inference ablations, not separately trained competitors.

## Predeclared gates

- L1: tested inference boundary, unit gate gain, three-read budget, gradient correctness, independent splits and browser/reference parity.
- L2: in condition 2, adaptive reduces future MSE versus random by at least 10% on the seed-averaged score, wins at least two of three seeds, and the paired 95% improvement interval is above zero.
- L3: in condition 2, adaptive reduces future MSE versus both optimized fixed and matched recurrent by at least 5%, wins at least two seeds against each, and both paired improvement intervals are above zero. Its full objective must also improve against both.
- L4: all L3 requirements also hold in the unseen-history condition 3. Failure is preserved; same-family success alone cannot claim this transfer.
- L5: erased and post-mix controls have future NRMSE at least 0.90 in conditions 2 and 3. Tiny floating-point initial differences cannot count as recovered information.

A learned interface earns an advantage here only if L1..L5 pass. A random-baseline win with L3/L4 failures establishes trained readout, not an adaptive architectural advantage. No threshold, model size, noise, loss weight, train family or seed may be changed after these held-out results. Implementation defects may be repaired with a written invalidation and complete rerun; failed scientific gates may not be patched away.

## Deliverables

Publish the training/evaluation code, gradient and boundary tests, all selected weights, training/validation traces, held-out receipt and limits. Add a dedicated browser page running the actual exported weights, with Python/JavaScript parity tests. Demonstration worlds are separate from training/validation/test; changing the final question must reuse the same acquired readings. Link the page from the current site, keeping both previous experiments accessible.

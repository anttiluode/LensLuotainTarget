# Learned querying — frozen outcome ledger

On 2026-10-09, the dictionary-free learned interface passed L1–L5 in the frozen [protocol](LEARNED_PROTOCOL.md). This is a positive result for this small simulator and these specific receivers; it does not establish a new neuron theory or general architecture superiority.

[Run the actual weights in the browser](https://anttiluode.github.io/LensLuotainTarget/site/learned.html). The [full receipt](results/learned_queries.json) includes every condition, control, seed and paired interval. [All twelve selected checkpoints](results/learned_weights.json) include validation traces and coefficient hashes.

## What was frozen, and what was learned

The protocol was committed locally before full training and held-out evaluation; all twelve validation-selected weight sets were committed locally before opening the 512-episode tests. Publication collects those records together; no earlier public preregistration is claimed. The held-out gates, model sizes, seeds and training budgets were not retuned after seeing their outcomes.

Each model received exactly 2000 Adam updates, batch size 96. Checkpoints were selected only by the objective on a separate 512-episode validation batch. Three training seeds were retained, with four models per seed. Every comparison shares test histories, sensor noise and final questions. There are 512 distinct episodes per condition, each evaluated by all three trained models per approach; they are not 1536 independent worlds.

The sender has twelve linear branch traces. Histories are continuous samples, end at zero input and are deliberately projected to the same initial uniform sum. Four history families train the receivers; a switching-block family is withheld. The trained selector receives its own retained state and read index, never the stimulus history, true branch values, a label, an eight-history catalogue or the final question.

Training still receives supervised future-answer labels and exact gradients through the simulator. The decoder predicts twelve original branch coordinates; the linear final-question interface is designed in advance. Final questions at test include dense signed vectors, unlike training's individual branches and positive subsets. Changing the final question cannot change the acquired observations or issue another read.

## Complete aggregate outcomes

NRMSE is root MSE divided by the zero predictor's root MSE. Smaller is better; 1 matches that baseline. The full objective is answer MSE/0.1² + 0.1 × squared final displacement/0.1² + 0.001 × 3. State change is the mean relative Euclidean displacement, a different quantity from the squared loss penalty.

### Fresh histories, familiar question kinds

| Receiver / control | Answer NRMSE | Full objective | Relative state change |
|---|---:|---:|---:|
| Learned adaptive | 0.2717 | 0.03095 | 14.3% |
| Optimized fixed | 0.3000 | 0.03603 | 13.1% |
| Random | 0.7016 | 0.15149 | 7.8% |
| Matched recurrent | 0.2993 | 0.03615 | 13.5% |
| Adaptive selection context zeroed | 0.6551 | 0.13196 | 8.4% |
| Branch memory erased | 1.0181 | 0.74064 | 100.0% |
| Gate after mixing | 1.0312 | 0.31592 | 0.0% |

### Main test: fresh histories, new dense signed questions

| Receiver / control | Answer NRMSE | Full objective | Relative state change |
|---|---:|---:|---:|
| Learned adaptive | 0.2509 | 0.03288 | 14.3% |
| Optimized fixed | 0.2806 | 0.03913 | 13.1% |
| Random | 0.7044 | 0.19296 | 7.8% |
| Matched recurrent | 0.2792 | 0.03905 | 13.5% |
| Adaptive selection context zeroed | 0.6494 | 0.16407 | 8.4% |
| Branch memory erased | 1.0214 | 0.82743 | 100.0% |
| Gate after mixing | 1.0327 | 0.40351 | 0.0% |

### Transfer: withheld switching histories, new signed questions

| Receiver / control | Answer NRMSE | Full objective | Relative state change |
|---|---:|---:|---:|
| Learned adaptive | 0.2753 | 0.04634 | 13.7% |
| Optimized fixed | 0.2923 | 0.05094 | 13.2% |
| Random | 0.6831 | 0.22052 | 7.9% |
| Matched recurrent | 0.2900 | 0.05064 | 13.6% |
| Adaptive selection context zeroed | 0.6201 | 0.18302 | 9.0% |
| Branch memory erased | 1.0112 | 1.02281 | 100.0% |
| Gate after mixing | 1.0307 | 0.48907 | 0.0% |

Adaptive reads cause more displacement than optimized fixed reads. The answer improvement is large enough that the full penalized objective is lower in all three conditions. The read-count cost is constant, so this experiment does not learn when to stop.

## Paired comparisons and all training seeds

The relative improvements below are in **MSE**, not NRMSE. Intervals are 2048 paired bootstrap resamples of the 512 episode errors, averaged across the three training seeds. They describe episode uncertainty conditional on those trained models, not broad training-seed uncertainty.

| Condition | Baseline | Adaptive MSE reduction | 95% interval: baseline minus adaptive MSE | Adaptive seed wins | Full-objective improvement |
|---|---|---:|---|---:|---:|
| fresh_familiar | Random | 85.0% | [0.00104172, 0.00145308] | 3/3 | 0.12054 |
| fresh_familiar | Optimized fixed | 18.0% | [0.00003476, 0.00006219] | 3/3 | 0.00508 |
| fresh_familiar | Matched recurrent | 17.6% | [0.00003366, 0.00005915] | 3/3 | 0.00520 |
| novel_queries | Random | 87.3% | [0.00128269, 0.00205833] | 3/3 | 0.16008 |
| novel_queries | Optimized fixed | 20.1% | [0.00004623, 0.00007278] | 3/3 | 0.00625 |
| novel_queries | Matched recurrent | 19.2% | [0.00004285, 0.00007038] | 3/3 | 0.00618 |
| unseen_history | Random | 83.8% | [0.00154570, 0.00207135] | 3/3 | 0.17418 |
| unseen_history | Optimized fixed | 11.2% | [0.00002794, 0.00006056] | 3/3 | 0.00460 |
| unseen_history | Matched recurrent | 9.9% | [0.00002168, 0.00005409] | 3/3 | 0.00430 |

| Condition | Training seed | Adaptive NRMSE | Fixed NRMSE | Random NRMSE | Matched recurrent NRMSE |
|---|---:|---:|---:|---:|---:|
| fresh_familiar | 20261009 | 0.2685 | 0.2824 | 0.7388 | 0.2855 |
| fresh_familiar | 20261010 | 0.2720 | 0.2914 | 0.6886 | 0.2874 |
| fresh_familiar | 20261011 | 0.2744 | 0.3244 | 0.6759 | 0.3235 |
| novel_queries | 20261009 | 0.2471 | 0.2679 | 0.7440 | 0.2680 |
| novel_queries | 20261010 | 0.2506 | 0.2739 | 0.6890 | 0.2695 |
| novel_queries | 20261011 | 0.2549 | 0.2990 | 0.6783 | 0.2989 |
| unseen_history | 20261009 | 0.2756 | 0.2821 | 0.7029 | 0.2788 |
| unseen_history | 20261010 | 0.2744 | 0.2856 | 0.6740 | 0.2808 |
| unseen_history | 20261011 | 0.2761 | 0.3084 | 0.6719 | 0.3095 |

The receipt retains the per-seed control scores, MSE, objective and displacement as well. The browser shows seed 20261009's actual selected weights on ten independent demonstration worlds; neither demonstration seeds nor example performance are used for the test verdict.

## Frozen gate outcomes

| Gate | Outcome | Evidence |
|---|---|---|
| L1: inference boundary, gradients, budgets, splits, parity | PASS | Full contract suite includes finite-difference gradients and Python/JavaScript parity for actual exported coefficients. |
| L2: beyond random | PASS | Main-test MSE improves 87.3%; positive paired interval; all three seed wins. |
| L3: beyond optimized fixed and matched recurrent | PASS | Main-test MSE improves 20.1% and 19.2%; positive paired intervals; all seed wins; both full objectives improve. |
| L4: unseen-history transfer | PASS | Transfer MSE improves 11.2% and 9.9%; positive paired intervals; all seed wins; both full objectives improve. |
| L5: lost-information controls | PASS | Erased and post-mix NRMSE exceed 0.90 on both main and transfer tests. |

Contract sealing is separate from scientific evaluation: `learned_experiment.py evaluate` leaves L1 pending. `make_learned_site_data.py --verify` runs the complete test suite before sealing L1 and re-exporting the exact receipt. `verify_learned_receipt.py` recomputes the held-out metrics from saved weights and checks every scientific outcome and weight hash. GitHub Actions runs both the tests and the receipt reproduction, without expensive retraining.

The fresh final review caught an export guard that could accept a successful test exit with Node parity checks skipped. The guard now rejects skipped or empty runs. All parity checks actually ran in the published evaluation; this repair changes verification metadata, not trained weights, scientific metrics or frozen thresholds.

## Limits that stay attached to the result

- **Known simulator during training.** Exact read/write gradients and supervised target answers are substantial training advantages. There is no learned physical law, camera calibration or biological evidence here.
- **Structured memory prior.** Three scalar answers cannot identify a general twelve-dimensional state. Common-sum projection, correlated trace geometry and the sampled history families make this a structured inference problem. Transfer covers one withheld family, not arbitrary worlds.
- **Designed linear question interface.** Dense signed questions test unseen vectors within an explicitly linear task. The system does not discover arbitrary question semantics or predict a changing future environment.
- **Specific resource comparison.** Adaptive uses 944 parameters, 2176 neural MACs and 28 persistent sender-plus-receiver values; matched recurrent uses 941, 2071 and 31. Counts exclude nonlinear operations, plant work, optimizer work, prediction workspace and command buffers. This is one tanh recurrent baseline, not an exhaustive GRU/RNN/transformer comparison.
- **Additional receiver computation.** The adaptive gate head feeds a state-dependent nonlinear command back into the receiver's next update. The positive comparison therefore tests the complete implemented interface. It does not isolate every benefit of physical adaptive measurement from that additional nonlinear receiver computation. A future head-matched control could separate them.
- **Outbound cost.** Adaptive commands twelve coefficients per read; fixed gates can be shared beforehand. There is no demonstrated communication, latency or energy advantage.
- **Ablations are not retrained baselines.** Context-zeroing retains answers for decoding but changes trained selection inputs; post-mixing also changes the gate command available to the receiver. These reveal failure under removed interfaces, without isolating every source of the normal model's gain.
- **Displacement is not irreversible forgetting.** At eta=0.25 the physical update is invertible: `(I-0.25*g*gᵀ)^(-1)=I+(1/3)*g*gᵀ` for a unit gate. Genuine erasure is a separate control.
- **Limited replication.** All three predeclared training seeds improved, but conditional episode bootstrap intervals do not replace larger independent training-seed replication.

The original shadow and supplied-history receipts remain intact. The ecg.json observation-in-feedback accident, moons, optical masks and branch gates motivate a precise common question; they have different equations and are not asserted to be the same physical mechanism.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_learned_receipt.py
python make_learned_site_data.py --verify
# Optional full training in a separate output; preserves the frozen publication:
python learned_experiment.py train --weights /tmp/retrained-weights.json
python learned_experiment.py evaluate --weights /tmp/retrained-weights.json --output /tmp/retrained-receipt.json
```

Only NumPy is required for training. Browser/reference tests require Node.js. No test-time learning or catalogue search runs in the browser.

## Verification and live demonstration

The final review found no remaining publication blockers after the verification guard repair. All **42 tests ran and passed without skips**, including actual-weight browser parity. The frozen receipt reproduced from saved weights, all browser JavaScript parsed, and the data/page export was deterministic. Both unit-tests and Pages deployment succeeded for release commit `aa491bd629032940ad4f667294b505ef61282bdc`.

On the live desktop page, all four trained receiver choices ran; each information-loss control forced the adaptive ablation as labelled. Three reads disabled the read button; reset restored zero new reads and zero displacement. Changing the final question preserved the acquired readings, gate display and closed read budget while changing the prediction. Transfer results matched the frozen receipt. The older supplied-history and shadow views and their new return link remained accessible. No page-origin console errors or horizontal desktop overflow were observed.

[Verified preview of the learned gate and reusable final questions](docs/images/learned-queries.jpg).

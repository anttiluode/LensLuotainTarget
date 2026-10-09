# Memory → gate → soma: outcome ledger

The protocol was frozen in [NEURON_PROTOCOL.md](NEURON_PROTOCOL.md) before opening seeds 9100..9611. This is an abstract linear trace/selected-readout model with a supplied eight-history dictionary and correct dynamics. The output is a noisy scalar, not an action-potential waveform.

## Reads with eta = 0.0

| Strategy | Correct / 512 | Log loss, nats | Original-answer NRMSE | Mean relative state change |
|---|---:|---:|---:|---:|
| Memory-guided | 294 | 1.062063 | 0.398363 | 0.000000 |
| Random | 208 | 1.463286 | 0.615548 | 0.000000 |
| Fixed branches | 216 | 1.382811 | 0.573496 | 0.000000 |
| Selection context discarded | 272 | 1.127017 | 0.440602 | 0.000000 |
| Gate after mixing | 54 | 2.079442 | 1.008698 | 0.000000 |
| Branch memory erased | 60 | 2.079442 | 1.008698 | 1.000000 |

Frozen G1–G5 outcomes: G1 PASS, G2 PASS, G3 PASS, G4 PASS, G5 PASS.

Active improves accuracy over random by 16.797 percentage points. It improves log loss over the stronger open-loop selector by 0.064954 nats. Active and open-loop choose different gate sequences in 87.50% of trials.

## Reads with eta = 0.25

| Strategy | Correct / 512 | Log loss, nats | Original-answer NRMSE | Mean relative state change |
|---|---:|---:|---:|---:|
| Memory-guided | 290 | 1.088227 | 0.411323 | 0.164329 |
| Random | 209 | 1.466270 | 0.615331 | 0.102125 |
| Fixed branches | 216 | 1.382811 | 0.573496 | 0.121542 |
| Selection context discarded | 269 | 1.135279 | 0.441734 | 0.160754 |
| Gate after mixing | 58 | 2.079442 | 1.008698 | 0.000000 |
| Branch memory erased | 60 | 2.079442 | 1.008698 | 1.000000 |

Frozen G1–G5 outcomes: G1 PASS, G2 PASS, G3 PASS, G4 PASS, G5 PASS.

Active improves accuracy over random by 15.820 percentage points. It improves log loss over the stronger open-loop selector by 0.047052 nats. Active and open-loop choose different gate sequences in 84.96% of trials.

## What survived

Both conditions pass the frozen active-versus-random and retained-context-versus-open-loop gates. Erasing the branch traces or applying the gate only after the common sum leaves log loss at log(8) and original-answer error near the uninformed prediction. Retained answers therefore earn a modest role in selecting later gates, beyond knowing the candidate dictionary. The three-read adaptive gain over the strong open-loop baseline is 22/512 additional correct histories without disturbance and 21/512 with disturbance.

The read-write trade-off is separate: eta=0.25 changes active branch state by mean relative norm 0.164 while accuracy stays close to the eta=0 condition. The policy does not optimize this damage cost or restore the original memory. This validates only the deliberately imposed linear read-write rule.

G6 is checked by the reference tests and cross-runtime tests: the browser uses the same trace, observation-only selection, posterior update and state-write equations for frozen example histories. Its noise slider is illustrative; the benchmark always uses sigma=0.03.

## Limits preserved

- The same initial answer is imposed by projection; this is an ambiguity stress test, not natural neuronal activity.
- A known finite dictionary and exact forward/read model are supplied. No gate, encoder or decoder was learned from observations; unknown-history and model-mismatch generalization were not tested.
- The branch memories are linear leaky traces, held stationary between instantaneous reads. The write rule is an abstract projection, not ion-channel dynamics, reconsolidation or a spike-width mechanism.
- All gates have norm 1, but selected branches, selector evaluations and dictionary storage are not equal-cost physical hardware. Active and open-loop each evaluate 24 gates; random and fixed do not need that scoring.
- Twelve branch-state numbers are accompanied by 96 observer candidate-state numbers, eight belief numbers and shared histories/filter/gate machinery. This is not a 12-number complete AI.
- Original-answer NRMSE targets unseen gates of the pre-read memory, not future environmental dynamics. The erased control still predicts that original target, avoiding a trivial perfect-zero score.
- Chance-control classification can depend on machine-precision argmax ties. Their near-chance accuracy and log(8) log loss are the information-boundary checks; tiny confidence differences are not information.
- The 64-seed exploratory pilot used the same frozen values. Its no-damage context gate failed; the separate 512-seed held-out gate passed. No threshold or noise was changed after the pilot or held-out results.
- The original ecg.json is a checkerboard aliasing/variance-controller loop. It motivates observation inside feedback; it does not share these equations or establish this neural mechanism.

## Reproduce

```bash
python -m unittest discover -s tests -v
python neuron_experiment.py --output results/memory_gate_soma.json
python make_neuron_site_data.py
```

Use `--details /tmp/neuron-details.json` for all per-seed actions, observations, beliefs and metrics. Browser parity tests require Node; NumPy is the only Python dependency. The live page is a visual demonstration over three exploratory example worlds, not a rerun of all held-out trials.

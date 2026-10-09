# Memory → gate → soma: frozen first gate

## Question and authorization

Extend the existing LensLuotainTarget page with the user-approved test: different histories have the same present soma output; compare fixed, random and context-selected gates at equal observation budgets; measure downstream recovery and read disturbance. This is a bounded synthetic extension of the existing candidate/probe experiment, not a conductance-based neuron or an action-potential model.

## The two memories

Twelve branches retain different leaky traces of one 32-step signed stimulus history: m_i(t+1)=lambda_i*m_i(t)+(1-lambda_i)*u(t), with lambda evenly spaced from 0.35 to 0.98. Eight seed-specific histories end with the same input, zero, and are projected into the null direction of the initial un-gated soma sum. Histories are bounded by max absolute input 1. The traces persist between instantaneous reads; natural inter-read decay is intentionally omitted.

The observer retains an eight-probability belief over the supplied candidate histories. It knows the trace/gate model and candidate dictionary, not which history occurred. Its active policy selects the unused gate maximizing belief-weighted variance of predicted scalar soma answers. This is programmed experimental design, not a learned biological gate or proof of optimal mutual information.

## Instrument and controls

- All physical gates have Euclidean norm 1. The catalogue has 12 individual-branch gates and 12 deterministic coded branch subsets, seed 20261009. Un-gated soma uses the unit-norm uniform vector.
- A read emits y=g^T*m plus independent Gaussian noise, sigma 0.03. It then writes m'=m-eta*g*(g^T*m). eta=0 and eta=0.25 are separate conditions. All candidate predictions undergo the same known issued action, without access to the true state.
- One common initial un-gated observation plus three new scalar outputs for every policy. Selection computation is disclosed rather than claimed equal.
- Active: posterior-guided gate. Random: random unused catalogue gate. Fixed: branch order 0,4,8,1,5,9,... followed by coded gates. Open-loop: same candidate dictionary and selector, but uniform selection weights every step; the final decoder still retains all observations.
- Post-mix: gate acts only on the already-combined scalar, with gain 1. Erased: replace branch traces by zero before reading, preserving the receiver's original candidate dictionary and prior. These are controlled information-loss ablations, not matched biological implementations.
- All decoders receive only gate identity and noisy scalar output. The dictionary is privileged prior information and its storage is charged separately. Truth, branch traces and noise are unavailable to gate selection.
- Measure classification, true-label log loss, posterior entropy, prediction of eight held-out gate answers of the original pre-read memory, and relative retained-state disturbance. Forecast queries are not used for calibration or selection.

## Separation and gates (frozen before the held-out run)

Exploration uses seeds 8100..8163 only. Confirmatory evaluation uses fresh seeds 9100..9611 (512 per eta condition). The noise, budget, filters, gate catalogue and thresholds above will not change after evaluation. Any failure stays a failure.

- G1: initial candidate soma differences / noise below 1e-6, and all last stimulus samples zero.
- G2: active mean log loss at least 0.10 nats below random, and accuracy at least 0.05 higher, in each eta condition.
- G3: active mean log loss at least 0.03 nats below open-loop in each eta condition; at least 20% of trials have an active sequence different from open-loop. This isolates the value of retained observer context.
- G4: post-mix and erased accuracy each in [0.08,0.18], log loss equal to log(8) within 1e-8, and their forecast uncertainty remains. Chance-band classification is a check, not a claim of significance.
- G5: eta=0 leaves the state unchanged; eta=0.25 produces nonzero mean active disturbance. This verifies the imposed read-write rule, not a biological finding.
- G6: tests, deterministic receipts and Python/JavaScript parity pass.

Passing G1/G4 establishes the constructed information boundary. G2/G3 must pass separately to claim a useful active or adaptive-policy advantage. No learned gate, realistic waveform channel, unknown-history generalization, model-mismatch robustness or neural implementation is established. The original PerceptionLab ecg.json is an aliasing/variance-controller ancestor of the question, not this model's equations.

## Site contract

Keep the shadow explainer working. Add a Memory → gate → soma tab, a view of two stimulus histories and their retained traces, selectable pre-soma gates, a noisy scalar-response timeline, observer belief and read-disturbance meters. Show the frozen benchmark and gates directly from the generated receipt. Label it an abstract simulation throughout. The visible hidden traces are explanatory diagnostics; the policy never consumes them. The demo uses stored example histories with the same equations, not a rerun of all 512 trials.

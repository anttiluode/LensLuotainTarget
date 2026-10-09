# Memory → gate → soma implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Test the approved memory-to-gate-to-soma mechanism and publish an interactive, accurately labeled extension of the existing site.

**Architecture:** A NumPy reference model traces histories into branch memory, selects norm-matched gates from retained observation context, and decodes noisy scalar output. A small browser implementation uses frozen example histories and the same equations. Generated receipts supply the result display.

**Tech stack:** Python 3.12+, NumPy, plain JavaScript, existing GitHub Pages.

**Spec:** NEURON_PROTOCOL.md.

## Global constraints

- Eight candidate histories; 12 branches; 32 stimulus steps; common final input zero.
- Three additional reads; Gaussian noise sigma 0.03; eta=0 and 0.25 reported separately.
- Norm-1 gates; true state unavailable to selector and decoder; known dictionary costs explicit.
- Seeds 8100..8163 exploratory; 9100..9611 held out. Preserve failed gates.
- Keep the original optical test and page intact; no new dependencies.

## Review focus

- Floating-point initial differences must remain negligible relative to noise.
- Candidate-state propagation may use issued gates, never the hidden truth.
- Erased-state forecasting targets original memory, so trivial zero prediction cannot masquerade as success.
- Browser reset, changing history/noise/damage and reaching the read budget must remain consistent.
- The existing shadow controls, mobile layout and keyboard navigation must keep working.

## Task 1: reference model and held-out receipt

Files: memory_gate_soma.py, neuron_experiment.py, tests/test_memory_gate_soma.py, results/memory_gate_soma.json, NEURON_RESULTS.md.

Interfaces: MemorySetup; trace_history(history, decays); candidate_histories(seed, setup); gate_catalog(setup); choose_gate(posterior,predictions,available); read_state(state,gate,eta); trial(seed, method, setup, budget=3, truth=None); benchmark(seeds, setup, budget=3).

- [x] Write behavioral tests for trace recurrence, matched initial soma, seed diversity, context-dependent selection, post-mix and erased information loss, hidden-label separation, equal measurement counts and read disturbance.
- [x] Run tests and observe the missing-model failure.
- [x] Implement the reference model and CLI; run the full suite.
- [x] Run exploration without changing the frozen values; run 512 held-out seeds per eta; export summary and preserve every gate outcome.
- [x] Verify receipt regeneration and commit this task.

## Task 2: browser mechanism and site

Files: site/neuron-core.js, site/neuron-data.js, site/neuron-ui.js, site/neuron.css, site/index.html, tests/test_neuron_browser.py, README.md, .github/workflows/tests.yml.

Interfaces: Browser core exposes trace, posterior update, chooseGate, readState, begin and step. Generated data contains reference fixtures and the exact benchmark summary; the UI consumes those interfaces.

- [x] Write cross-runtime tests comparing history traces, observation-only gate sequences, post-mix/erased controls and state disturbance on fixed fixtures; observe failure before JavaScript exists.
- [x] Implement the pure browser core and generated data. Confirm parity.
- [x] Add the mechanism tab, responsive visualization, controls, receipts and limitations; retain the original shadow section and IDs.
- [x] Run the full 20-test suite, JavaScript syntax checks and byte-identical receipt/fixture regeneration. Review the responsive stylesheet. A local browser is unavailable; interaction verification follows deployment on live Pages.
- [x] Complete a fresh code/model review and fix its manual-then-automatic gate finding with a failing-then-passing regression over 72 combinations.
- [x] Publish the verified files using guarded updates of main; check reset, policies and the optical controls on live Pages. Keep current diagram sums distinct from received noisy readings.

## Verification record

- 20 reference/browser tests pass; the manual-then-automatic regression covers 72 combinations.
- A fresh reviewer independently checked 288 scene/truth/policy/damage combinations and byte-identical held-out receipt regeneration. No Critical or Important findings remain.
- Live Chrome controls passed: all strategies, manual then automatic, three-read budget, erasure, post-mix, damage switch, noise reset, history reset, another world, reset and keyboard tabs. The original shadow mask chooser improves displayed separation from 0.31 to 2.89 at fixed noise.
- Desktop page width fits its 1348-pixel viewport. Responsive CSS was reviewed; a separate mobile viewport was not available in this browser API.
- Unit-test and Pages workflows passed for the mechanism deployment. The original optical model files and historical receipt are preserved.

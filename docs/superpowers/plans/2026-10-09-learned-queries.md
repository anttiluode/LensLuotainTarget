# Learned queries implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox syntax for tracking.

**Goal:** Train and test a useful query policy without a supplied history dictionary, and publish actual learned inference with an honest frozen result.

**Architecture:** Separate deterministic data generation, observation-only recurrent inference, differentiable simulator/training, and held-out scoring. Export all weights as JSON; a small browser core runs those same recurrent equations.

**Tech Stack:** NumPy, unittest, plain JavaScript, existing GitHub Pages.

**Spec:** LEARNED_PROTOCOL.md.

## Global constraints

- 12 branches, 32 stimulus steps, three new readings, noise 0.03, eta=0.25.
- Adaptive/ordinary receivers have 16 states; matched recurrent has 19. Three independent training seeds.
- Exactly 2000 updates, batch 96, Adam lr=0.003, clip=1.0; validation every 100 updates.
- No dictionary, history, branch value or future question in the selector inputs. Preserve existing model and receipts.
- All L1..L5 outcomes and all seeds are published. No tuning after test.

## Review focus

- Gradient paths must include both reading and state-writing, and gate normalization.
- Future questions must not affect gate selection or consume new readings.
- Erasure and post-mix ablations must still forecast the original memory.
- A selected checkpoint must be chosen exclusively on validation data; bootstrap must pair common episodes across seeds/methods.
- Browser inference must use exported coefficients and equal histories/noise, without reconstructing a candidate catalogue.

### Task 1: data and observation-only inference

Files: learned_data.py, learned_receiver.py, tests/test_learned_receiver.py.

Interfaces: make_batch(namespace,seed,size,query_kind,history_kind) -> arrays; init_model(method,seed) -> model; initial_receiver(model,initial_answer), select_gate(model,h,step,random_gate=None), receive(model,h,gate,answer), predict_state(model,h); rollout(model,batch,mode='normal') -> received/actions/forecast/state metrics.

- [x] Write and observe failing tests for seed separation, continuous history diversity, common initial sum, gate norm, observation-only selection, final-question independence, read budget and erased original target.
- [x] Implement the generators and inference; pass the whole suite.
- [x] Commit the component and record verification.

### Task 2: gradients, training and frozen evaluation

Files: learned_training.py, learned_experiment.py, tests/test_learned_training.py, results/learned_queries.json, results/learned_weights.json, LEARNED_RESULTS.md.

Interfaces: loss_and_grad(model,batch) -> (loss,gradients); train(method,seed,steps=2000) -> selected weights and validation trace; evaluate(models,batches) -> metrics/per-episode losses; paired_interval(left,right,seed) -> interval; frozen_gates(receipt) -> L1..L5.

- [x] Write and observe failing finite-difference tests for decoder, recurrence, gate head and write effect; test small-training loss decrease and validation-only selection.
- [x] Implement exact vectorized backpropagation and Adam; pass the whole suite.
- [x] Train all 12 models under the frozen protocol; select validation checkpoints and freeze weight hashes.
- [x] Open all held-out conditions once, preserve every outcome, write the result ledger; commit.

### Task 3: actual trained browser inference and publication

Files: site/learned.html, site/learned-core.js, site/learned-data.js, site/learned-ui.js, site/learned.css, make_learned_site_data.py, tests/test_learned_browser.py, site/index.html, README.md, MATH.md, .github/workflows/tests.yml.

Interfaces: browser initialReceiver/selectGate/receive/predictState match Task 1; generated data holds seed-20261009 selected weights, separate demonstration worlds and the exact receipt. UI acquires three readings and permits new final queries without resetting the receiver.

- [x] Write and observe failing reference/JS parity and exact receipt tests; include all controls and final-query reuse.
- [x] Implement and verify the core, data and responsive page; label resource limits and failed gates from the receipt.
- [x] Run complete tests, syntax and deterministic export checks; complete one fresh final review and resolve material findings.
- [x] Publish by guarded GitHub ref update, verify workflows and live controls, and save a verified preview.

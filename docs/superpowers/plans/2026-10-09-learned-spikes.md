# Learned binary querying implementation plan

Binding spec: `SPIKE_PROTOCOL.md`. Baseline: `397b8ae`.
Worktree: LensLearnedWork, branch learned-spike-thresholds. Execute inline.
The user authorized continuing the experiment and publication. Kuulustelu stays
read-only. Reuse this already isolated linked worktree. Preserve old protocols,
training sources and receipts.

1. Freeze the protocol. Add separate data/receiver modules. Test disjoint splits,
   observation-only interfaces, actual hard bits, command limits and head-matched
   physical controls. Expected: missing functions fail first, then full suite green.
2. Verify gradients against exact expected-loss bit-tree enumeration and graded
   numerical gradients. Implement training, sealing and evaluation. Test stable
   tails, architecture validation and fresh control rollouts. Freeze 24 checkpoints
   before held-out access. Run exactly the declared budget, report all outcomes.
3. Export actual weights into a new browser instrument and result page. Test
   Python/JS episode parity, command paths, budget and future-question-only decoder.
   Verify sealed receipt and exports; link existing instrument pages and document
   result scope. Update CI for dependencies and receipt verification.
4. One fresh whole-branch review using the most capable available reviewer, then
   one verified fix pass. Publish through guarded GitHub plugin updates and verify
   actions and the live page. Save a reviewed live screenshot.

Review focus: score-function gradient unbiasedness and baseline independence;
control capacity matching; bit/command leakage; disjoint noisy worlds; selection
before testing; frozen criteria and receipt integrity; threshold-specific versus
gate-specific conclusions; browser uses real sealed weights; graded comparisons
are not presented as equal-bit; old experiments and Kuulustelu remain untouched.

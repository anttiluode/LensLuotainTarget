# The measurement, its invisible directions, and the next probe

## 1. A known forward operator, not a magic inverse

For an unknown screen brightness vector x, measurement p gives y_p=A_p x+epsilon_p. Each row of A_p corresponds to a wall pixel, each column to a hidden screen patch. The 1-D transport is integrated over five subpixel sources and uses parallel-plane D²/r⁴ attenuation. An opaque intermediate mask blocks intersecting rays. A single unmasked uniform-screen exposure fixes an intensity normalization for every subsequent mask; blocked rays reduce total light.

This is a geometry-based but reduced model. The real 2019 periscopy paper works in two dimensions and includes camera/occluder calibration, backgrounds, and photon noise. The 1-D model does not reproduce its real-world reconstruction evidence.

## 2. Distinguishability and nullspaces

Two hidden images x and x' are exact twins for p if A_p(x-x')=0. After p_1,...,p_t, the remaining exact ambiguity is the intersection of kernels of their transport matrices. With additive Gaussian sensor noise epsilon ~ N(0,sigma² I), the KL divergence between observation distributions for two images under probe p is:

    KL(p(y|x,p) || p(y|x',p)) = ||A_p(x-x')||² / (2 sigma²).

Near-zero KL means the images cannot be reliably distinguished through that measurement at the available exposure/noise, even if a fancy inverse algorithm invents a sharper-looking answer.

The experiment manufactures eight candidates from a four-dimensional subspace of the smallest right singular vectors of A_0. Their initial predictions agree essentially to numerical precision. Importantly, those *are adversarial synthetic alternatives,* not real images drawn from a natural distribution. The hidden label is never supplied to the selector.

## 3. Probe selection without seeing the answer

Given categorical posterior weights w_i over the candidate screens, define predicted wall signal mu_{p,i}=A_p x_i. The expected pairwise KL averaged over two independently drawn candidates is:

    E_{i,j~w} KL(N(mu_i, sigma² I)||N(mu_j, sigma² I))
      = (1/sigma²) sum_i w_i ||mu_{p,i} - sum_j w_j mu_{p,j}||².

Select the currently unused p maximizing this posterior-weighted variance. This is **not** the exact mutual information between the hidden label and the next observation; it is a convenient pairwise-KL surrogate. It is fully computable from the posterior, candidate library and known forward operators, without the unknown true label.

Upon observing y at that mask:

    w'_i = [w_i exp(-||y-A_p x_i||²/(2 sigma²))]
           / sum_j [w_j exp(-||y-A_p x_j||²/(2 sigma²))].

The code uses a log-sum-exp-style normalization to avoid underflow.

## 4. What changes when we ask a different question

A fixed 12-number MovingTarget2 code can give excellent weak answers but need additional higher moments to predict what an interaction changes. Here a low-observability measurement similarly compresses distinct hidden states into nearly identical answers. Neither result implies information reappears from nowhere. New probes couple to previously inaccessible directions, so the next answer distinguishes what the earlier query couldn't.

Connection to active sensing: the world is not just an answer oracle. The observer is permitted to **choose A_p**. If the additional measurement makes rival predictions diverge, it can settle an uncertainty that passive repetition cannot.

## 5. A bound and a caveat

If every allowed measurement has A_p(x-x')=0, no policy using those measurements can distinguish the two images above chance under equal priors, because the entire observation distributions agree. More computation cannot change that. If there exists a p with A_p(x-x') != 0, the difference becomes statistically identifiable with sufficient independent repetitions under the stated known-model and stationary-scene assumptions. The finite-budget performance still depends on signal/noise and policy.

The claim does not transfer automatically to nonlinear perception, neural memories, language-model hallucinations or uncalibrated cameras. Those settings require evidence that the proposed query family is realizable and informative, and that their observation model is calibrated.

## History traces, gated sum and read disturbance

The new [memory/gate/soma gate](NEURON_PROTOCOL.md) uses branch traces
`m_i(t+1)=lambda_i*m_i(t)+(1-lambda_i)*u(t)`.
Its history-to-state map F is linear. Candidate histories are projected onto
`ker(g_0^T F)` with a separate shared last-input constraint; they therefore
agree in the initial un-gated read but differ in their retained traces.

A norm-1 gate emits `y=g^T m + epsilon` and applies the imposed write
`m'=m-eta*g*(g^T m)`. The observer propagates **all** candidates under the
issued gate; it does not access the true state. Previous received outputs
update its categorical belief p. The active choice maximizes
`sum_h p_h*(g^T m_h - sum_j p_j*g^T m_j)^2` over unused gates.
The open-loop comparison substitutes uniform selection weights while keeping
the same dictionary, state propagation and downstream evidence accumulation.

In the post-mix control every choice can only act on the already-combined
scalar `g_0^T m`; gain 1 is used to keep this a pure information-loss control.
The erased control replaces the retained physical state by zero. Both retain
the receiver's original dictionary, so prior knowledge alone is insufficient
to identify the unknown history. Their forecast target is the original
pre-read state's answers on independent held-out gates, not the trivial
zero state after erasure.

These equations are a constructed observability test. They do not identify
the original ECG-loop thermostat with a neuron, or derive dendritic biology.

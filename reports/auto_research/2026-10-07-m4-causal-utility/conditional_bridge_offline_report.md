# Conditional M4 bridge: frozen offline gate

Date: 2026-10-07

Parent commit: `849037409`

Protocol: `conditional_bridge_offline_protocol.json`

Canonical result: `conditional_bridge_offline_result.json`

## Question

Can the already measured M4 signal define a reachable treatment that differs from
the registered reward-variance scheduler while holding service-relevant token load
and reward variance fixed? This is a development replay. It does not update a model
or test learning quality.

## Frozen policy

The deployment score predicts exact token-normalized group-gradient magnitude from
log token count, reward mean, reward variance, and log M4. Zero-M4 groups receive a
zero prediction because every zero-M4 group in the authenticated acquisition also
had zero exact gradient.

At each archived eight-group frontier, the comparator is the registered
reward-variance proposal. The conditional policy searches all four-group subsets
inside the comparator's FIFO-relative 0.98--1.02 token band, rejects any subset
with lower total reward variance, and selects the feasible subset with the largest
predicted gradient-norm sum. It actuates only on a strict predicted improvement.

## Authenticated inputs

- Gradient-utility ledger: 256 groups, SHA-256
  `a226a6de068256fca55a48a8be931cd5ae33896cdfa4e92d6c55d13b414124d6`.
- Controlled-frontier source archive: 64 decisions with eight candidates each,
  SHA-256
  `c4e8be7c84d3899ca220c1a50858c87024b2e572f8f636ed481cee279f6a4617`.
- Ledger extracted from that archive: SHA-256
  `ccd493e4077ad538ded6b510b2ce2b69affbf39a30fbe006968c1618f58b9d22`.

## Result

All frozen offline gates passed.

- Primary positive-M4/nonzero-gradient population: 214 groups.
- Baseline cross-fitted R2: 0.7600.
- M4-augmented cross-fitted R2: 0.9362; incremental R2: 0.1762.
- Strict treatment contrasts: 16 of 64 archived decisions (25%).
- Median predicted gradient-magnitude gain among interventions: 20.61%; mean:
  26.36%.
- Interventions with at least 10% predicted gain: 12.
- Minimum reward-variance retention: 1.0000.
- Selected/comparator token ratio range: 0.98015--1.01990.
- Mean comparator overlap on interventions: 2.375 of four groups.

The canonical replay hash is
`a95a84b25d56451e7ee3e870d7bf44da021433954aa5a14e9c0cb5c48add4d77`.
Two independent executions produced byte-identical result files with SHA-256
`15fcabfcc3636985fef93858938bbf6fb8d278ddc2453a0b734776db326ac326`.

## Decision and boundary

The result authorizes implementation and outcome-excluded qualification of a
version-zero capsule collector. It does **not** establish that this treatment
improves an update, held-out loss, terminal accuracy, or production behavior.

A causal one-update experiment remains gated on all of the following:

1. the treatment and comparator are selected from one shared frozen candidate
   frontier without consulting learning outcomes;
2. both arms use byte-identical retained training tensors and matched fresh model
   initialization, configuration, and RNG state;
3. a disjoint held-out surrogate-loss reader is qualified without optimizer or
   scheduler mutation; and
4. the selected treatment contrast is authenticated before either arm's outcome is
   opened.

Separate live jobs are not an acceptable substitute for the shared capsule:
historical runs did not reliably reproduce an identical candidate frontier even
under the same seed.

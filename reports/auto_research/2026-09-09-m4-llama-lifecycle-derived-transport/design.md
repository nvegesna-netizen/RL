# Llama 3.2 1B lifecycle-derived M4 transport design

Status: `FROZEN_LOCAL_SUCCESSOR_PROTOCOL_PENDING_IMPLEMENTATION`.

## Separation from the closed study

This is a new, separately preregistered successor study. The preceding Llama
family-transport study remains terminally closed at commit `59a0f88c8` after
GSM8K missed its unchanged 1% synchronous-observer-duty gate. Nothing here is
a retry, amendment, threshold relaxation, or completion of that protocol.

The successor keeps the scientific treatment, estimands, model, workloads,
training geometry, causal windows, analysis rules, and 1% support ceiling. It
changes the measurement architecture prospectively: opportunity is reconstructed
after the active run from arm-blind pre-treatment lifecycle primitives, and the
synchronous per-group opportunity callback is disabled.

## Scientific question

Does the accepted M4 opportunity-loss effect transport from Qwen3 to
`meta-llama/Llama-3.2-1B-Instruct`, and does the OpenMath-versus-GSM8K contrast
retain the direction observed in the completed Qwen grid?

The primary cross-family anchor is Llama/OpenMath. The Llama/GSM8K cell and the
within-Llama workload contrast are prospectively registered supporting tests.
The design cannot identify a pure model-family effect because tokenizer, chat
template, exact parameter count, and model history also differ.

## Lifecycle-derived instrument

For every prompt group, the frozen opportunity quantity remains the absolute
token-level GRPO coefficient mass before release:

`Q = sum_i |A_i| * T_i`,

where `A_i` is the production-equivalent float32 leave-one-out normalized GRPO
scalar advantage and `T_i` is the valid assistant-token count for sibling `i`.
Rewards, sibling indices, trajectory identities, truncation, token counts, and
start versions come only from `sibling_done` lifecycle events. Sample identity
comes only from the matching `group_ready` lifecycle event. Completed learner
steps come only from `removed(reason=selected)` events paired with the matching
`learner_version_advanced` transition.

The derivation code is forbidden from reading `release_arm`,
`release_delay_seconds`, assignment draws, nonces, masses, or any downstream
outcome/disposition when computing Q. It runs only after the active window has
closed. Its output records the maximum controller sequence of the source
primitives; it does not receive or fabricate a pre-delay computation timestamp.
The analysis join establishes pre-treatment timing by requiring every primitive
source event to precede `release_delay_started` for that group.

The synchronous `gradient_opportunity_audit` callback must be disabled. The
only measurement observer allowed during the active run is the common lifecycle
recorder. Its total synchronous `record()` body is timed across every lifecycle
event, in both arms, with no branch on arm label. Corrected lifecycle-recorder
duty must remain at or below `0.01`. Post-run serialization and reconstruction
elapsed time are reported separately and cannot alter release or training.

## Instrument bridge gates

Before any training launch, all of the following must pass:

1. Bit-exact float32 scalar-advantage equivalence to the accepted production
   estimator on exhaustive binary eight-sibling reward vectors and randomized
   supported fixtures.
2. Aggregate and sibling opportunity equivalence to the accepted synchronous
   instrument on compact historical fixtures, with exact integer fields,
   bit-exact scalar advantages, and `1e-12` absolute tolerance for float64
   aggregates.
3. Exact recovery of group/sample identity and completed-step membership from
   lifecycle topology.
4. Treatment-blindness metamorphism: arbitrary changes to all release-arm and
   release-dose fields leave every derived opportunity byte unchanged.
5. Fail-closed rejection of incomplete siblings, invalid ordering, duplicate
   identities, nonfinite/unsupported rewards, missing selected-step evidence,
   or incompatible estimator/loss settings.
6. Total lifecycle-recorder duty instrumentation and post-run elapsed-time
   reporting are present and independently validated.

Any bridge or no-training preflight failure closes this successor without
qualification, retry, repair, substitution, or threshold change.

## Fixed execution geometry

- balanced randomized arms: `control=0s`, `d5=5s`;
- burn-in start versions: 0--7;
- primary start versions: 8--407;
- terminal guard start versions: 408--447;
- 448 trainer steps, four prompts per step, eight generations per prompt;
- windowed FIFO, maximum staleness one, 16 in-flight prompts, buffer 64;
- one generation H100 plus one trainer H100 per cell;
- OpenMath one epoch; GSM8K two epochs solely to reach the fixed window;
- four wall-hours and eight GPU-hours maximum per cell;
- no checkpointing, automatic retry, extension, rerandomization, or cell
  substitution.

## Registered analysis and conclusions

The accepted adjusted cell estimand remains

`(mean_d5(Q*D)-mean_control(Q*D))/pooled_pre_delay_mean_Q`.

Each cell uses eight-fold generalized-regression adjustment with Q and
`1[Q=0]`, version-clustered four-lag HAC, and 20,000 circular-block bootstrap
draws of eight start versions. The reported 95% interval is the outer
HAC/bootstrap envelope. Terminal missingness uses the already frozen endpoint
bounds; missing Q is red.

- OpenMath is `MATERIAL` only if the lower envelope exceeds 0.20 and the
  materiality test rejects at 0.05; `NOT_MATERIAL` requires an upper endpoint
  at or below 0.20; otherwise it is `INCONCLUSIVE`.
- GSM8K is `POSITIVE` only if its whole envelope exceeds zero, `NEGATIVE` only
  if wholly below zero, otherwise `INCONCLUSIVE`.
- The workload contrast is `SAME_DIRECTION` only if the entire envelope for
  `Llama(OpenMath)-Llama(GSM8K)` exceeds zero, `OPPOSITE_DIRECTION` only if it
  is wholly below zero, otherwise `INCONCLUSIVE`.

Mechanism replication, lifecycle completeness, and lifecycle duty are mandatory
support conditions but cannot override the causal conclusions.

## One-shot execution sequence

1. Commit and push the frozen protocol and implementation package.
2. Submit exactly one shared credential-free no-training EOS preflight with
   `runllm.py --no_wait`. It may run tests, equivalence checks, package/hash
   validation, and metadata-only model access, but no training or qualification.
3. Only if it passes, freeze and submit both neutral 32-step qualifications
   before inspecting either. Qualification observations never enter a causal
   estimator.
4. Both cells must pass every registered qualification gate. Either miss closes
   the successor permanently without retry, repair, or acquisition.
5. Only if both pass, freeze both acquisition packages and submit both before
   inspecting either causal outcome.
6. After both terminal artifacts are authenticated and frozen, run the joint
   registered analysis exactly once.

The user's instruction to execute the sequence authorizes these conditional
steps; eligibility gates remain binding and do not authorize retries.

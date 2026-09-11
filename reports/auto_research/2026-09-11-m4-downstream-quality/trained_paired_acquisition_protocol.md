# Prospective M4 downstream-quality paired acquisition

Status: `FROZEN_DESIGN_PENDING_IMPLEMENTATION_AND_EXPLICIT_TRAINING_AUTHORIZATION`

This is a new successor study. It does not reopen, reclassify, or contribute
outcomes to any completed M4 acquisition. No training, EOS qualification, or
scientific acquisition is authorized by this document.

## Question and causal interpretation

Does running the accepted 1:1 control/d5 controlled-release policy during
training change terminal held-out exact-answer accuracy relative to an
otherwise identical all-immediate policy after 448 learner updates?

The randomized unit is an independently seeded training run. The causal
estimand is the intention-to-treat effect of the release policy. M4 opportunity
loss is a measured intermediate outcome, not the randomized treatment itself.
Consequently, a quality difference may be attributed to the release policy,
but not exclusively to the M4 mediator: ordering, staleness, congestion, and
other learning-path changes may also mediate the effect.

## Why this design

The V8 no-training preflight authenticated terminal export, conversion, vLLM
reload, deterministic prompt selection, and scoring. It produced 369 successes
among 1,024 prompts (36.04%), so the endpoint is not at floor or ceiling. It did
not train a model and cannot serve as a causal control.

The existing M4 result establishes that the 1:1 d5/control policy causes
material opportunity loss in this Qwen3-0.6B/OpenMath setting. Retaining that
exact policy avoids introducing an unvalidated delay dose or changing the
accepted assignment primitive. Comparing it with an all-immediate run asks
whether the measurement intervention is large enough to change the trained
model.

The accepted anchor is source commit
`47dc1a713cbf0e07813d31c4ba70a8262214e098`, protocol SHA-256
`6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c`,
and result artifact SHA-256
`07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6`.
Its adjusted M4 estimate was 0.3054 with conservative envelope
[0.2759, 0.3349]. These values motivate the setting and policy; none enters
the new final-quality estimator.

Sixteen matched seed blocks are fixed. Prompt rows are repeated measurements,
not causal replicates. Four pairs would make inference depend heavily on one
seed and cannot support a useful exact randomization test. Sixteen pairs allow
all 65,536 within-block sign flips, while remaining feasible for the smallest
tested Qwen model.

## Frozen cells

Each block contains two runs with the same training seed, split, prompt order,
model initialization, optimizer, batch geometry, and 448-update boundary:

- `immediate`: controlled release enabled with a single zero-second arm;
- `mixed_d5`: the accepted equal-mass zero-second and five-second arms.

The dataset split seed is fixed at `20260911` and is decoupled from the 16
training seeds. Mixed-policy assignment has a fresh frozen domain and seed in
each block. Within-block submission order is the SHA-256-derived order recorded
in the JSON protocol; execution may be concurrent, but queue order may not be
chosen using outcomes or system conditions.

Every run uses Qwen/Qwen3-0.6B, OpenMathInstruct-2, four prompts per update,
eight sibling generations per prompt, and the accepted asynchronous
SingleController geometry. The only intended causal difference within a block
is the release policy.

## Terminal endpoint

After exactly 448 completed learner updates and learner version 448, each run
must atomically export terminal policy weights without optimizer state. An
independent post-training job converts that immutable export to Hugging Face
format and evaluates exactly the V8 prompt manifest under greedy decoding:

- prompt count: 1,024;
- prompt-manifest SHA-256:
  `469a34a176febe9d15c5c6d967ff21f135a43cb20a7cb7d5ae1a53066e6b3a0a`;
- one generation per prompt;
- temperature 0, top-p 1, top-k -1, max new tokens 1,792;
- primary run score: mean binary exact-answer reward.

No in-loop validation, best-checkpoint selection, adaptive horizon, or
evaluation feedback into training is allowed.

## Primary estimand and decisions

For block `b`, let `D_b` be terminal accuracy under `mixed_d5` minus terminal
accuracy under `immediate`. The primary estimate is the equal-weight mean of
the 16 block differences.

The practical margin is fixed at 0.02 absolute accuracy. This is about 20.5
answers on the 1,024-prompt endpoint and 5.55% of the V8 base accuracy. It is a
prespecified study decision threshold, not an externally derived business
value.

Using the two-sided 95% paired Student interval:

- `MATERIALLY_WORSE` if its upper endpoint is below -0.02;
- `MATERIALLY_BETTER` if its lower endpoint is above 0.02;
- `PRACTICALLY_EQUIVALENT` if the entire interval is inside [-0.02, 0.02];
- `DETECTABLE_BUT_NOT_MATERIAL` if the interval excludes zero but satisfies
  none of the preceding rules;
- otherwise `INCONCLUSIVE`.

The exact 65,536-sign-flip test and a 20,000-draw paired bootstrap are
sensitivities. Prompt-level paired differences and a block/prompt mixed model
are precision diagnostics only and may not replace the run-level primary
analysis.

## Mechanism and systems outcomes

Every run retains the lifecycle and opportunity ledgers. The mixed runs must
report delay compliance, realized control/d5 opportunity loss, direct-chain
contrast, version advance, and observer duty. Immediate runs provide a
contemporaneous zero-dose reference. Throughput, wall time, tokens, response
length, collapse indicators, and time-to-fixed-quality are secondary.

Mechanism failure never licenses exclusion or as-treated substitution. The
intention-to-treat quality contrast remains primary, with interpretation
qualified by measured compliance.

## Fail-closed execution rules

- Freeze all 32 resolved configurations and their hashes before the first
  training submission.
- Submit all authorized runs without inspecting any regime-labeled terminal
  score. Outcome release occurs only after the 16-pair completion gate is
  resolved.
- Require exact source, dependency, image, protocol, config, prompt-manifest,
  terminal-export, converted-weight, and evaluation-output hashes.
- Require all 1,024 terminal prompt scores for every run.
- No automatic retry, replacement, extension, seed substitution, run-count
  change, model/workload change, or endpoint change.
- A failed or incomplete run makes the primary 16-pair decision
  `INCONCLUSIVE_INCOMPLETE`. Any operational repair requires evidence, a
  prospective amendment made without unblinding terminal scores, and separate
  authorization.
- Individual scheduler cap: four wall-hours and eight H100 GPU-hours per run.
  Aggregate hard cap: 128 wall-hours summed across runs and 256 H100 GPU-hours.

## Next boundary

The next permissible local step is implementation: build the two config
templates, materialize and hash the 32 run manifests, implement the detached
paired analyzer and outcome embargo, and test them without training. A launch
requires a separate explicit authorization that names training or the paired
acquisition.

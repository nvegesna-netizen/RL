# Prospective downstream-quality study design

Status: `DRAFT_LOCAL_NO_LAUNCH_AUTHORITY`

This is a new successor hypothesis. It does not reopen or modify any completed
M4 acquisition or decision.

## Scientific question

Does the controlled-delay training regime that produces measured M4 opportunity
loss change final held-out model quality relative to immediate release at the
same learner-update budget?

The randomized exposure is the run-level delay regime. The primary causal
estimand is its intention-to-treat effect on final quality. Opportunity loss is a
measured intermediate outcome. The study will not claim that it is the exclusive
mediator because delay can also change ordering, staleness, congestion, or other
learning dynamics.

## Anchor configuration

The recommended anchor is Qwen3-0.6B on OpenMathInstruct-2:

- it is the canonical prospectively confirmed M4 setting;
- its M4 estimate is 0.3054 with envelope [0.2759, 0.3349];
- the model is the smallest tested Qwen configuration;
- the existing 448-step geometry can be retained.

This selection is provisional until a no-training configuration audit confirms
that the exact SingleController path can load the model, construct one fixed
train/evaluation split, export terminal policy weights, and reload them in an
independent evaluator. SingleController deliberately rejects its generic
validation and resumable-checkpoint settings, so those facilities cannot be
assumed from the synchronous GRPO path. It may not be changed after any
regime-labeled final-quality outcome is inspected.

## Run-level intervention

The training run is the independent unit. Runs are organized into matched seed
blocks. Within each block, one run is assigned to each regime before submission:

1. `immediate`: every eligible group receives zero added release delay;
2. `mixed_d5`: a frozen SHA-256 assignment maps exactly half of eligible groups
   in expectation to d5 and half to immediate release, matching the accepted M4
   exposure policy.

The paired runs use the same model, fixed train/validation split, prompt order,
learner-update limit, batch geometry, optimizer, and evaluation protocol. They
use an independent training seed for each block. The dataset split seed must be
decoupled from the training seed and held constant across every run; otherwise
the final scores would refer to different held-out prompt sets.

The mixed regime is preferred over an all-d5 regime for the anchor because it
preserves the accepted intervention policy and supports a contemporaneous
within-run M4 mechanism audit. It dilutes any final-quality effect relative to
all-d5, so the power calculation must use the run-level outcome variance rather
than the historical group-assignment count.

## Training and evaluation geometry

- Primary resource axis: exactly 448 completed learner updates.
- Training groups per update: 4 prompts × 8 sibling generations = 32 samples.
- Evaluation split: one fixed 5% OpenMath split, seeded independently of training.
- Candidate evaluation size: 1,024 fixed prompts with one generation each. This
  is subject to a measured runtime and score-variance gate in the no-training
  preflight.
- Terminal policy export: mandatory after step 448 and only after the bounded
  run reaches both its train-step and learner-version boundary. It contains
  weights and the resolved run configuration, not optimizer state or a
  resumable training checkpoint. The base tokenizer identity remains frozen in
  that resolved configuration.
- Evaluation is a separate post-training job over the immutable export. It must
  use the same frozen prompt IDs and decoding settings for every run and must not
  feed traffic or state back into training.
- Primary score: exact-answer evaluation accuracy for the terminal export.
- Primary contrast: `mixed_d5 - immediate`, averaged across independent seed
  blocks.
- Secondary learning outcomes: baseline-adjusted final score, collapse
  incidence, and response length. There is no learning-curve or best-checkpoint
  claim in this minimal design.
- Secondary systems outcomes: wall time, samples and tokens processed,
  throughput, realized M4 loss, version advance, and hold compliance.

Final accuracy at equal update count is primary because it addresses learning
efficacy. Equal-wall-clock quality and time-to-quality are secondary systems-
utility views and may not replace the primary endpoint.

## Decisions and inference

Let `delta_q` be the smallest practically meaningful absolute accuracy change.
It is an open human scientific gate and must be set before any regime-labeled
pilot outcome is inspected. It must not be derived from the eventual point
estimate.

For paired seed-block differences `Y_mixed - Y_immediate`:

- report the mean difference and a two-sided 95% interval at the run level;
- use a paired studentized interval as primary and an exact sign-flip or paired
  bootstrap analysis as sensitivity;
- classify `MATERIALLY_WORSE` only if the entire interval is below `-delta_q`;
- classify `MATERIALLY_BETTER` only if the entire interval is above `delta_q`;
- classify `PRACTICALLY_EQUIVALENT` only if the entire interval lies inside
  `[-delta_q, delta_q]`;
- otherwise classify `INCONCLUSIVE`.

Evaluation prompts and rollout groups reduce measurement noise but are not
independent training replicates. No analysis may use their count as the degrees
of freedom for the final-quality effect.

## Blinded feasibility and variance stage

The first stage is not a free-form retry loop. Before it begins, freeze:

- `delta_q`;
- four initial matched seed blocks;
- the regime assignment for every block;
- a maximum number of blocks and GPU-hour cap;
- a nuisance-variance-only sample-size recalculation rule;
- floor, ceiling, terminal-export/reload, score-variance, and runtime gates;
- a rule for whether initial blocks enter the final estimator.

The preferred design retains the initial blocks and uses a preprogrammed blinded
sample-size recalculation that emits only the required block count and gate
status, not arm means or a treatment contrast. If that procedure cannot be
validated to preserve the intended error rate, designate the first four blocks
as a nonconfirmatory pilot and exclude them from the fresh confirmatory analysis.

No claim is permitted from the initial four blocks alone unless a separately
validated sequential boundary was frozen before acquisition.

## Stop rules

Stop before scientific acquisition if any of these occurs:

- the evaluation set is not identical across paired runs;
- terminal export, reload, or final evaluation cannot complete and be authenticated;
- the endpoint is at a prespecified floor or ceiling;
- the required run count exceeds the frozen compute cap;
- treatment changes a supposedly fixed training input other than release timing;
- regime assignment or outcome labels are exposed before the blinded rule runs.

There is no automatic retry, sample-size extension, model substitution, workload
substitution, or endpoint change after outcome inspection.

## Remaining gates before a no-training preflight

1. Human approval of `delta_q` and the maximum compute budget.
2. Static config overlays for immediate and mixed-d5 regimes with unique output
   identities and fixed dataset split.
3. Validation of seed separation, run assignment, output isolation, terminal
   export settings, evaluator reload, and exact resolved configuration.
4. A local analyzer for run-level results and the blinded sample-size rule.
5. A credential-free package whose embedded authority stops before training.

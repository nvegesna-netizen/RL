# Qwen3-0.6B/GSM8K M4 grid-completion design

Status: `FROZEN_LOCAL_PACKAGE_PENDING_NO_TRAINING_PREFLIGHT`.

## Scientific question

Does a fixed five-second release delay produce a material M4 opportunity-loss
effect for Qwen3-0.6B on GSM8K under the accepted instrument and the frozen 0.20
threshold?

This is the missing fourth cell in the existing Qwen3 model-scale by workload
grid. Relative to the completed Qwen3-1.7B/GSM8K R2 study, the intended
scientific change is model and tokenizer identity only. The dataset, GRPO loss,
two-epoch capacity rule, control:d5 intervention, queue geometry, opportunity
instrument, estimator, inference, missingness semantics, fixed window, topology,
and resource ceiling remain unchanged.

No prior OpenMath, Qwen3-1.7B, GSM8K R1, or qualification observation may enter
the new primary estimator. Existing evidence is used only to select this grid
cell and plan its fixed information target.

## Primary endpoint

The primary estimand remains

`(mean_d5(Q×D) - mean_control(Q×D)) / pooled_pre_delay_mean_Q`,

estimated by the accepted eight-fold cross-fitted generalized regression using
only pre-delay `Q` and `1[Q=0]`. Inference retains four-lag version HAC,
eight-version circular blocks, 20,000 bootstrap draws, sharp terminal bounds,
and the outer HAC/bootstrap confidence envelope.

The fixed decision rule is:

- `MATERIAL` when the lower envelope endpoint exceeds 0.20 and the registered
  material p-value is at most 0.05;
- `NOT_MATERIAL` when the upper envelope endpoint is at most 0.20;
- `INCONCLUSIVE` otherwise;
- `INSUFFICIENT_TERMINAL_COVERAGE` when either primary arm exceeds 1% missing
  terminal dispositions.

The mechanism and observer-duty conclusions are support conditions and cannot
override the primary endpoint.

## Fixed primary geometry

- burn-in start versions: 0–7;
- primary start versions: 8–507;
- terminal guard start versions: 508–557;
- trainer steps: 558;
- minimum primary assignments: 7,395;
- preferred primary assignments: 7,500;
- two epoch-specific GSM8K prompt exposures permitted;
- one generation GPU and one trainer GPU;
- four wall-hours and eight GPU-hours maximum;
- no automatic retry, extension, rerandomization, or outcome-guided stopping.

Planning scales the completed Qwen3-1.7B/GSM8K HAC standard error to 7,395
assignments and inflates it by 25% for model transport, giving `0.0155332`.
Under a normal planning approximation, classification power is `0.89598` at
both 0.15 and 0.25, above the 0.80 gate. Power is only `0.36298` at 0.175 and
0.225; near-threshold outcomes may legitimately remain inconclusive.

## Secondary grid synthesis

The secondary analysis uses common start versions 8–407 in every cell. It will
report each common-window adjusted cell estimate, both within-scale workload
contrasts, both within-workload model-scale contrasts, and

`(0.6B OpenMath - 0.6B GSM8K) - (1.7B OpenMath - 1.7B GSM8K)`.

Each cell is independently randomized. Joint uncertainty therefore combines
independent cell-specific four-lag HAC variances and independently seeded
eight-version circular block-bootstrap shifts. The 95% outer HAC/bootstrap
envelope is reported for the interaction. An interaction claim requires that
envelope to exclude zero. This secondary result cannot override or relabel any
cell's registered primary conclusion, and no family-wide generalization follows
from two models and two workloads.

The interaction requires complete terminal scoring in the common window for all
four cells. If any common-window terminal disposition is missing, the grid
interaction is reported as unavailable; the fourth cell's registered primary
missingness analysis remains valid and is not altered.

Historical 500-version results will be reanalyzed over the common window before
the fourth-cell outcome exists. The historical Qwen3-0.6B/OpenMath primary
window already equals the common window. These reanalyses are secondary grid
inputs; they do not replace the preserved terminal results.

The planning reconstruction now passes. Over versions 8–407,
Qwen3-1.7B/OpenMath has 6,955 assignments, estimate `0.2982263`, and HAC standard
error `0.0145328`; Qwen3-1.7B/GSM8K has 7,613 assignments, estimate `0.1430635`,
and HAC standard error `0.0125265`. The common-window OpenMath-minus-GSM8K
difference at Qwen3-1.7B is `0.1551628`. These checks used 199 bootstrap draws
only to validate reconstruction; they deliberately report no confidence
interval. The frozen terminal secondary analysis still requires 20,000 draws
per independently seeded cell.

## Gate sequence

1. Validate the exact protocol, common-window contract, config delta, and
   outcome-blind power calculation locally.
2. Freeze a source commit and run a pinned no-training preflight. It may resolve
   configuration and execute tests but may not train or submit an acquisition.
3. If preflight is green, separately freeze and authorize one 32-step neutral
   qualification. Its observations are excluded from all causal estimators.
4. Use qualification only to confirm topology, reward support, assignment yield,
   observer duty, and completion within the fixed resource cap.
5. Freeze and review the acquisition package. A fresh, exact authorization is
   required before one `runllm.py --no_wait` submission.

This design creates no EOS or acquisition authority.

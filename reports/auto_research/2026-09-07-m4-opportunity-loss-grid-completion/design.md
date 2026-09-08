# Qwen3-0.6B/GSM8K M4 grid-completion design

Status: `STUDY_COMPLETE_PRIMARY_MATERIAL_SECONDARY_INTERACTION_DETECTED`.

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

## No-training preflight result

The frozen package passed its single authorized containerized no-training
preflight. Parent pipeline `66657829` and downstream pipeline `66657895` both
succeeded. Slurm job `5989387` exited zero. The pinned environment resolved the
configuration, safely unpacked all 1,950 source-archive members, passed 150
selected tests, passed Ruff check and format validation, compiled the selected
sources, and built the fail-closed lock.

The terminal artifact SHA-256 is
`d938bffc23c4eb3d14d22ceec0dd777202543fe5a64bf0e13d30aff7163e1e26`.
The lock SHA-256 is
`fb8d0ee532bc8fc688154a9ce19bc9747be9ad5872254ebb9143311be99914ce`.
All 49 locked files were reconciled byte-for-byte to source archive
`6152ed6c879241a68ccb0081ed7b886ca23e6b52aba52017400fa9327c3c0e36`
at source commit `3f2f959167ed4a4be1d353d8dafb45321aad95c3`.

The preflight trained nothing and acquired no scientific observations. It
authorizes neither qualification nor acquisition. The next gate is to build,
review, and separately authorize one neutral 32-step qualification package.

## Neutral qualification design

The neutral qualification was separately authorized for exactly one 32-step
submission. Its resolved configuration differs from the frozen acquisition
configuration only in the step limit, fresh neutral assignment identity, sole
zero-second neutral arm, and isolated lifecycle, opportunity, observer-duty,
and metrics paths. Qualification observations remain excluded from every
primary and grid estimator. This authorization does not extend to the 558-step
scientific acquisition.

## Neutral qualification result

The single authorized qualification passed. Parent pipeline `66666794`,
downstream pipeline `66667017`, and Slurm job `5989573` completed successfully.
It completed 32 trainer steps, yielded 565 opportunity groups over 32 observed
start versions, observed both binary reward values, and measured corrected
observer duty `0.002934918347246835`. With the registered 1.25 safety factor,
the 558-step projection is 1.538276939 wall-hours and 3.076553878 two-GPU-hours,
below the frozen four-wall-hour and eight-GPU-hour caps.

The terminal artifact SHA-256 is
`bc3931a37c8e9b510392806d552e4043c2af55bc3ed91fd3f864f4f768213677`.
Independent reconciliation from the raw lifecycle and opportunity streams
matched the emitted summary and all embedded artifact hashes.

The qualification remains non-causal: it used only the neutral zero-second arm,
produced no causal estimate, and its observations cannot enter any primary or
grid estimator. The next gate is acquisition-package freeze and review. A fresh
exact authorization is required before the single 558-step submission.

## Acquisition result

The single authorized acquisition completed all 558 trainer steps and produced
9,573 primary assignments with zero missing terminal dispositions in either
arm. Its adjusted primary estimate is `0.2323419953`, with registered 95% outer
envelope `[0.2107693490, 0.2532757032]` and material p-value `0.0015999200`.
The registered conclusion is `MATERIAL`.

The supporting unadjusted result is `INCONCLUSIVE` and does not override the
primary endpoint. The d5-vs-control mechanism conclusion is `REPLICATED`, and
corrected observer duty `0.0032864855` supports portability under the registered
ceiling.

Pipelines `66675448` and `66675506` are red only because the manifest invoked
the generic analyzer after acquisition; that entrypoint rejects the grid
protocol identity. The already frozen workload-transport analyzer explicitly
registered for this protocol recovered the exact 20,000-draw result from the
immutable ledgers. No reacquisition, protocol change, retry, extension, or
qualification observation entered the terminal result.

The preregistered secondary four-cell synthesis over common start versions
8–407 followed this primary analysis. It cannot override this cell's primary
conclusion.

## Secondary grid-synthesis result

The preregistered four-cell synthesis completed with 20,000 independently
seeded draws per cell and complete terminal scoring over common start versions
8-407. The Qwen3-0.6B OpenMath-minus-GSM8K contrast is `0.0721577766`; the
Qwen3-1.7B contrast is `0.1551627789`. Their registered difference is
`-0.0830050022`, with 95% outer HAC/bootstrap envelope
`[-0.1365688691, -0.0289398546]`. Because the envelope excludes zero, the
secondary conclusion is `INTERACTION_DETECTED`.

This interaction result does not override the Qwen3-0.6B/GSM8K primary
`MATERIAL` conclusion or relabel any historical cell. It supports heterogeneity
only across the four tested cells and does not establish generalization beyond
the tested models, workloads, instruments, runtimes, or environments.

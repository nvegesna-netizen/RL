# NuminaMath-1.5 paired M4 generalization design

Status: `PAIRED_QUALIFICATION_GREEN_PENDING_ACQUISITION_PACKAGE_FREEZE`.

## Scientific question

Does the previously observed model-by-workload heterogeneity replicate on a
clean third math workload when the accepted M4 opportunity-loss instrument is
run as a matched Qwen3-0.6B/Qwen3-1.7B pair?

The registered primary interaction is

`(0.6B NuminaMath - 0.6B GSM8K) - (1.7B NuminaMath - 1.7B GSM8K)`.

The GSM8K terms are immutable historical inputs over common start versions
8–407. Both filtered NuminaMath-1.5 cells are prospective acquisitions. They
retain the accepted intervention, instrument, estimator, inference method,
queue geometry, two-GPU topology, common window, and terminal semantics.

## Outcome-blind workload selection

DAPOMath17K and DeepScaleR were rejected for duplicated prompts before any
candidate causal outcome. NuminaMath-1.5 is pinned to revision
`1b05109f9e5c1ad06c0663519502416c30b300f8`. The registered valid/verifiable
filter requires nonblank problems and yields 680,786 rows and 680,786 unique
problems across three exact
SHA-256-bound parquet shards.

This is an outcome-blind amendment after the first no-training preflight found
one empty problem string admitted by the original filter. That preflight created
no lock and ran no training, qualification, acquisition, or causal analysis. No
scientific design parameter changed.

## Fixed primary geometry

- burn-in start versions: 0–7;
- primary start versions: 8–407;
- terminal guard start versions: 408–447;
- trainer steps: 448 per cell;
- minimum projected primary assignments: 6,900 per cell;
- one generation GPU and one trainer GPU per cell;
- four wall-hours and eight GPU-hours maximum per cell;
- no automatic retry, extension, rerandomization, or outcome-guided stopping.

The frozen planning model retains completed GSM8K variances and inflates the
scale-matched OpenMath standard errors by 15% as conservative proxies for the
new cells. The planned combined interaction standard error is `0.0295832`, and
normal-approximation classification power at interaction magnitude `0.0830050`
is `0.80118`, above the 0.80 gate.

## Analysis and decision rule

Each cell retains the accepted adjusted M4 cell estimand and 0.20 materiality
rule as supporting evidence. The interaction above is primary. Each cell uses
four-lag version HAC and 20,000 independently seeded eight-version circular
block-bootstrap draws. Joint uncertainty is the outer HAC/bootstrap envelope.

- `SAME_DIRECTION_GENERALIZATION` if the entire 95% envelope is below zero;
- `OPPOSITE_DIRECTION_GENERALIZATION` if the entire envelope is above zero;
- `INCONCLUSIVE_GENERALIZATION` if the envelope includes zero;
- `UNAVAILABLE` if either new cell lacks complete common-window scoring.

Both prospective packages must be frozen and submitted before either causal
outcome is inspected. A failed cell cannot trigger outcome inspection, an
outcome-guided retry, or an extension. Supporting cell materiality, mechanisms,
and observer duty cannot override the interaction.

## Gate sequence

1. Commit and push the audit, exact loader, paired configs, power plan, and
   frozen protocol without training.
2. Under separate authority, run one shared pinned no-training preflight.
3. If green, freeze and separately authorize neutral 32-step qualifications;
   qualification observations cannot enter a causal estimator.
4. Require both qualifications to pass topology, reward-support, information,
   observer-duty, and four-hour capacity gates.
5. Freeze both acquisition packages and submit both before inspecting either
   outcome, only under fresh exact acquisition authority.
6. Reconcile immutable artifacts and run the registered paired analysis.

This local design creates no EOS, qualification, or acquisition authority.

## Repaired preflight result

The repaired shared no-training preflight succeeded as parent pipeline
`66799474` and downstream pipeline `66799605`. The terminal artifact SHA-256 is
`f2de4a664c6ed15048fc13d009edc9dc65da897bdc48ab018c8211501818f58e`.
All 121 selected tests passed, the 680,786-row dataset mapped successfully, and
all 50 lock entries match the frozen source archive. No training,
qualification, acquisition, or causal analysis occurred. The frozen protocol
and its SHA-256 remain unchanged.

## Neutral qualification design

The paired qualification gate was separately authorized for exactly two
32-step submissions: one for each prospective model cell. Each qualification
inherits its frozen acquisition config and changes only the step limit, a fresh
neutral assignment identity, the sole zero-second neutral arm, and isolated
lifecycle, opportunity, observer-duty, and metrics paths. Each submission has
one attempt, uses two GPUs, and has a 90-minute scheduler limit within the
frozen four-hour per-cell ceiling. Automatic retry and extension are forbidden.

Qualification observations remain excluded from every causal estimator. This
authorization does not permit either 448-step scientific acquisition.

## Neutral qualification result

Both authorized qualifications passed. The 0.6B cell completed 32 steps with
552 opportunity groups, corrected observer duty `0.0023387151`, and a 448-step
projection of 1.866 wall-hours and 3.731 two-GPU-hours. The 1.7B cell completed
32 steps with 535 opportunity groups, corrected observer duty `0.0017940087`,
and a projection of 2.095 wall-hours and 4.190 two-GPU-hours. Both observed
binary reward support and passed every registered operational gate.

These results are non-causal and remain excluded from all estimators. The next
gate is paired acquisition-package freeze. Both 448-step packages must be
frozen before either submission, and fresh explicit acquisition authority is
required.

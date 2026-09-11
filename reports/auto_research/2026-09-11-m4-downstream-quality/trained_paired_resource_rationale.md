# M4 downstream-quality resource and precision rationale

Status: `DESIGN_RATIONALE_NOT_COMPUTE_AUTHORITY`

## Endpoint information

The V8 preflight observed base accuracy 369/1,024 = 0.36035. Its simple
binomial standard error is 0.0150 for one run score. Reusing the identical
prompts within each block allows prompt pairing, but training-seed variation is
unknown and remains the reason that independent run blocks are required.

At 16 pairs, a paired standard deviation of 0.03 gives a standard error of
0.0075 and an approximate Student 95% half-width of 0.0160. A paired standard
deviation of 0.04 would instead give a half-width near 0.0213. These are design
scenarios, not empirical power claims: no independent trained-run variance has
yet been observed.

The fixed 16-pair design is preferable to an outcome-guided internal pilot. It
avoids exposing arm means before the run count is fixed, supports a complete
65,536-sign-flip randomization distribution, and is materially more credible
than four training pairs. There is no optional sample-size extension.

## Compute envelope

The accepted Qwen3-0.6B/OpenMath cadence was 11.878 seconds per learner update.
At 448 updates this corresponds to roughly 1.48 training wall-hours and 2.96
two-GPU hours per run before terminal conversion and evaluation. V8 exercised
the export/conversion/evaluation path in roughly thirteen elapsed minutes; this
is only a planning observation because its zero-step path differs from a
trained terminal export.

For 32 runs, the central planning total is therefore roughly 102 H100 GPU-hours
after a modest allowance for terminal evaluation. The hard fail-closed cap is
four wall-hours and eight GPU-hours per run: 128 summed wall-hours and 256 H100
GPU-hours over all runs. Queue time is excluded from compute accounting but
recorded.

These caps make the study bounded; they do not authorize resource use. Before
launch, a local manifest audit must prove that every run inherits the cap and
that no retrier or automatic extension is enabled.

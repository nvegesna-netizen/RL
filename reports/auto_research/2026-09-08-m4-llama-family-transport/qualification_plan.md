# Llama 3.2 1B paired neutral qualification plan

Status: `AUTHORIZED_PAIRED_QUALIFICATION_DESIGN_PENDING_PACKAGE_FREEZE`.

The next gate consists of exactly two neutral 32-step qualifications: one for
OpenMathInstruct-2 and one for GSM8K. Both packages must be frozen before either
submission, and both must be submitted before either terminal result is used to
alter the design. Each uses one generation H100 and one trainer H100, a
90-minute Slurm limit, and no automatic retry or extension.

The sole controlled-release arm is `neutral=0s`. These runs therefore estimate
no control-versus-delay contrast, produce no causal conclusion, and are
permanently barred from every family-transport estimator.

Each cell passes only if it completes all 32 trainer steps, preserves unique
group identities and the neutral zero-dose lifecycle, observes both binary
reward values, yields at least 15 opportunity groups per observed start
version, projects at least 6,900 assignments into the frozen 400-version
primary window, keeps corrected observer duty at or below 0.01, and projects
the full 448-step run below four wall-hours and eight two-GPU-hours using the
registered 1.25 safety factor.

This authorization covers only these two qualification submissions through
`runllm.py --no_wait`. It does not authorize either 448-step acquisition.
